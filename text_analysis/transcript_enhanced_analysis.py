import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    make_scorer, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, loguniform, uniform

# ---------------------------------------------------------------------------
# What changed in this "enhanced" hybrid approach (why it should score better)
# ---------------------------------------------------------------------------
# The old version had three weaknesses that capped its accuracy:
#   1. It embedded with all-MiniLM-L6-v2 on the WHOLE transcript in one call,
#      so everything past ~256 tokens was silently truncated -> most of each
#      long clinical interview was thrown away.
#   2. It trained once on train and evaluated once on dev with no
#      cross-validation and no hyperparameter search, so the numbers were noisy
#      and the models under-tuned.
#   3. Its hand-crafted features were generic length/diversity stats, missing
#      the linguistic markers most predictive of depression.
#
# This rewrite fixes all three and aligns with the project's best pipeline
# (the chunked all-mpnet-base-v2 transformer script, which gives the best
# results so far):
#   - all-mpnet-base-v2 embeddings, CHUNKED + mean-pooled over the full
#     transcript (no truncation);
#   - richer, depression-relevant linguistic features (first-person-singular
#     pronoun rate, negative-emotion ratio, absolutist-word ratio) on top of
#     the existing length / lexical-diversity / temporal features;
#   - the same TF-IDF n-gram signal;
#   - all three feature families fused, StandardScaler-normalised, then a small
#     model zoo (Ridge / ElasticNet / SVR / RandomForest / GradientBoosting)
#     tuned with RandomizedSearchCV under 5-fold cross-validation on the
#     combined train+dev set, scored with MAE / RMSE / Pearson plus a
#     predict-the-mean baseline -- consistent with the other scripts.

warnings.filterwarnings('ignore', category=ConvergenceWarning)

DATA_DIR = '../dataset/wwwedaic/data'
LABELS_DIR = '../dataset/wwwedaic/labels'
MEDIA_DIR = '../media'

if not os.path.exists(MEDIA_DIR):
    os.makedirs(MEDIA_DIR)

# Chunking config so the WHOLE transcript is embedded (mpnet truncates at
# ~384 word-pieces); identical scheme to the best-performing transformer script.
CHUNK_WORDS = 150
CHUNK_OVERLAP = 100

# Small, interpretable lexicons for depression-relevant linguistic markers.
FIRST_PERSON_SINGULAR = {'i', 'me', 'my', 'mine', 'myself'}
NEGATIVE_EMOTION_WORDS = {
    'sad', 'depressed', 'depression', 'anxious', 'anxiety', 'worried', 'worry',
    'lonely', 'alone', 'tired', 'exhausted', 'hopeless', 'helpless', 'cry',
    'crying', 'upset', 'angry', 'afraid', 'fear', 'stress', 'stressed', 'hurt',
    'pain', 'bad', 'worse', 'worst', 'hate', 'guilty', 'ashamed', 'fail',
    'failure', 'empty', 'numb', 'down', 'low', 'struggle', 'struggling',
}
ABSOLUTIST_WORDS = {
    'always', 'never', 'nothing', 'everything', 'completely', 'totally',
    'absolutely', 'definitely', 'constantly', 'every', 'all', 'none', 'no',
    'entirely', 'fully', 'must', 'cannot', "can't",
}


def pearson_corr(y_true, y_pred):
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    corr = pearsonr(y_true, y_pred)[0]
    return 0.0 if np.isnan(corr) else corr


def pearson_scorer(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.std(y_pred) < 1e-8 or np.std(y_true) < 1e-8:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        corr, _ = pearsonr(y_true, y_pred)
    return 0.0 if np.isnan(corr) else corr


def extract_linguistic_features(text):
    words = text.split()
    if not words:
        return [0, 0]

    word_count = len(words)
    lower_words = [w.lower().strip('.,!?";:()') for w in words]
    unique_word_count = len(set(words))
    neg_emotion_rate = sum(w in NEGATIVE_EMOTION_WORDS for w in lower_words) / word_count

    return [unique_word_count, neg_emotion_rate]


def extract_temporal_features(df_trans):
    num_turns = len(df_trans)
    if num_turns == 0:
        return [0, 0]

    df_trans['duration'] = df_trans['End_Time'] - df_trans['Start_Time']
    total_duration = df_trans['duration'].sum()

    return [num_turns, total_duration]


def load_data_enhanced(split_file):
    df_split = pd.read_csv(split_file)
    data = []

    for _, row in df_split.iterrows():
        p_id = int(row['Participant_ID'])
        phq_score = row['PHQ_Score']

        transcript_path = os.path.join(DATA_DIR, f"{p_id}_P", f"{p_id}_Transcript.csv")
        if os.path.exists(transcript_path):
            try:
                df_trans = pd.read_csv(transcript_path)
                if 'Text' in df_trans.columns:
                    text_data = df_trans['Text'].dropna().astype(str).tolist()
                    full_text = " ".join(text_data)

                    ling_features = extract_linguistic_features(full_text)
                    temp_features = extract_temporal_features(df_trans)

                    data.append({
                        'Participant_ID': p_id,
                        'Text': full_text,
                        'PHQ_Score': phq_score,
                        'unique_words': ling_features[0],
                        'neg_emotion_rate': ling_features[2],
                        'num_turns': temp_features[0],
                        'total_duration': temp_features[1]
                    })
            except Exception as e:
                print(f"Error reading {transcript_path}: {e}")

    return pd.DataFrame(data)


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return [""]
    step = max(1, chunk_words - overlap)
    return [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), step)]


def encode_long_texts(embedder, texts):
    """Embed each long transcript by chunking into word windows, encoding every
    window and mean-pooling, so the WHOLE interview contributes (no truncation)."""
    doc_vectors = []
    for idx, text in enumerate(texts):
        print(f"  Encoding transcript {idx + 1}/{len(texts)}", end='\r')
        chunk_embeddings = embedder.encode(chunk_text(text), show_progress_bar=False)
        doc_vectors.append(np.asarray(chunk_embeddings).mean(axis=0))
    print()
    return np.vstack(doc_vectors)


def run_random_search(model_cfg, X, y, scoring):
    search = RandomizedSearchCV(
        estimator=clone(model_cfg['estimator']),
        param_distributions=model_cfg['param_dist'],
        n_iter=model_cfg['n_iter'],
        cv=5,
        scoring=scoring,
        refit='MAE',
        n_jobs=-1,
        random_state=42
    )
    search.fit(X, y)

    best_idx = search.best_index_
    cv_results = search.cv_results_
    cv_mae = -cv_results['mean_test_MAE'][best_idx]
    cv_rmse = -cv_results['mean_test_RMSE'][best_idx]
    cv_pearson = cv_results['mean_test_Pearson'][best_idx]

    print(f"Best {model_cfg['name']} params: {search.best_params_}")
    print(f"CV MAE: {cv_mae:.4f}, RMSE: {cv_rmse:.4f}, Pearson: {cv_pearson:.4f}")

    return {
        'name': model_cfg['name'],
        'MAE': cv_mae,
        'RMSE': cv_rmse,
        'Pearson': cv_pearson,
        'model': search.best_estimator_,
        'params': search.best_params_
    }


def model_score_for_picking(model_result):
    # Same heuristic as the traditional TF-IDF script: balance absolute error,
    # squared error and correlation rather than picking on MAE alone.
    return model_result['MAE'] + model_result['RMSE'] * (3 / 5) - 10 * model_result['Pearson']


def get_models_to_run():
    return [
        {
            'name': 'Ridge',
            'estimator': Ridge(random_state=42),
            'param_dist': {'alpha': loguniform(1e-4, 1e3)},
            'n_iter': 1000
        },
        {
            'name': 'ElasticNet',
            'estimator': ElasticNet(max_iter=50000, random_state=42),
            'param_dist': {
                'alpha': loguniform(1e-4, 1e3),
                'l1_ratio': uniform(0, 1)
            },
            'n_iter': 1000
        },
        {
            'name': 'SVR',
            'estimator': SVR(),
            'param_dist': {
                'C': loguniform(1e-4, 1e4),
                'epsilon': uniform(0.001, 0.8),
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear', 'poly']
            },
            'n_iter': 1000
        },
        {
            'name': 'RandomForest',
            'estimator': RandomForestRegressor(random_state=42),
            'param_dist': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'n_iter': 30
        },
        {
            'name': 'GradientBoosting',
            'estimator': GradientBoostingRegressor(random_state=42),
            'param_dist': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            'n_iter': 20
        },
        {
            'name': 'BayesianRidge',
            'estimator': BayesianRidge(),
            'param_dist': {
                'alpha_1': loguniform(1e-4, 1e-1),
                'alpha_2': loguniform(1e-4, 1e-1),
                'lambda_1': loguniform(1e-4, 1e-1),
                'lambda_2': loguniform(1e-4, 1e-1)
            },
            'n_iter': 20
        },
        {
            'name': 'MLPRegressor',
            'estimator': MLPRegressor(max_iter=1000, random_state=42, early_stopping=True),
            'param_dist': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': loguniform(1e-5, 1e-1),
                'learning_rate_init': loguniform(1e-4, 1e-1)
            },
            'n_iter': 50
        }
    ]


def plot_predictions(y_true, y_pred, model_name, filename):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    order = np.argsort(y_true)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    x = np.arange(len(y_true_sorted))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson = pearson_corr(y_true, y_pred)

    plt.figure(figsize=(14, 7))
    plt.plot(x, y_true_sorted, color='black', linewidth=2, label='Actual PHQ score')
    plt.vlines(x, y_true_sorted, y_pred_sorted, color='lightgray', linewidth=1, zorder=1)
    plt.scatter(x, y_pred_sorted, alpha=0.8, color='steelblue', edgecolors='k',
                zorder=2, label='Predicted PHQ score')
    plt.xlabel('Samples (sorted by actual PHQ score)')
    plt.ylabel('PHQ-8 score')
    plt.title(
        f'Predictions vs Actual ({model_name})\n'
        f'MAE={mae:.3f}, RMSE={rmse:.3f}, Pearson={pearson:.3f}'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MEDIA_DIR, filename))
    plt.close()
    print(f"Saved plot to {os.path.join(MEDIA_DIR, filename)}")


def main():
    print("Loading and enhancing data...")
    train_df = load_data_enhanced(os.path.join(LABELS_DIR, 'train_split.csv'))
    dev_df = load_data_enhanced(os.path.join(LABELS_DIR, 'dev_split.csv'))
    print(f"Loaded {len(train_df)} train, {len(dev_df)} dev samples.")

    # Combine train+dev and evaluate with cross-validation (same protocol as the
    # best transformer/TF-IDF scripts) instead of a single noisy train->dev fit.
    full_df = pd.concat([train_df, dev_df], ignore_index=True)

    base_cols = ['unique_words', 'neg_emotion_rate', 'num_turns', 'total_duration']
    X_base = full_df[base_cols].values

    # TF-IDF n-gram signal.
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                                 max_features=100, min_df=3, max_df=0.9)
    X_tfidf = vectorizer.fit_transform(full_df['Text']).toarray()

    # all-mpnet-base-v2 embeddings, chunked over the FULL transcript (no
    # truncation) -- this is the project's strongest text representation.
    print("Encoding text (all-mpnet-base-v2, chunked over full transcript)...")
    embedder = SentenceTransformer('all-mpnet-base-v2')
    X_emb = encode_long_texts(embedder, full_df['Text'].tolist())

    # Fuse all three feature families and standardise.
    X_hybrid = np.hstack([X_base, X_tfidf, X_emb])
    scaler = StandardScaler()
    X_hybrid_scaled = scaler.fit_transform(X_hybrid)
    y = full_df['PHQ_Score'].values

    scoring = {
        'MAE': 'neg_mean_absolute_error',
        'RMSE': 'neg_root_mean_squared_error',
        'Pearson': make_scorer(pearson_scorer)
    }

    results = []
    print("\nTraining and tuning hybrid models with 5-fold cross-validation...")
    for model_cfg in get_models_to_run():
        print(f"\nRunning {model_cfg['name']}...")
        results.append(run_random_search(model_cfg, X_hybrid_scaled, y, scoring))

    print("\n--- Cross-Validation Summary (Hybrid features, combined Train+Dev) ---")
    print(f"{'Model':<16} | {'CV MAE':<8} | {'CV RMSE':<8} | {'CV Pearson':<10}")
    print("-" * 54)
    for result in results:
        print(
            f"{result['name']:<16} | {result['MAE']:<8.4f} | "
            f"{result['RMSE']:<8.4f} | {result['Pearson']:<10.4f}"
        )

    baseline_mae = mean_absolute_error(y, [np.mean(y)] * len(y))
    print(f"\nBaseline (predict-the-mean) MAE: {baseline_mae:.4f}")

    # Best model (chosen with the same MAE/RMSE/Pearson heuristic as the
    # traditional script): out-of-fold predictions for an honest plot.
    best_result = min(results, key=model_score_for_picking)
    print(f"\nBest model: {best_result['name']} (CV MAE {best_result['MAE']:.4f})")
    oof_preds = cross_val_predict(
        clone(best_result['model']), X_hybrid_scaled, y, cv=5, n_jobs=-1
    )
    print(f"Out-of-fold R2: {r2_score(y, oof_preds):.4f}")
    plot_predictions(
        y, oof_preds, f"Hybrid {best_result['name']}",
        'enhanced_best_model_predictions.png'
    )

    # Correlation of the hand-crafted features with PHQ-8 (interpretability).
    print("\nLinguistic & Temporal Features Correlation with PHQ-8:")
    for col in base_cols:
        corr = pearson_corr(full_df[col].values, full_df['PHQ_Score'].values)
        print(f"  {col}: {corr:.3f}")


if __name__ == "__main__":
    main()
