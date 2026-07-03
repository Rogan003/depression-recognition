import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Ridge, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, loguniform, uniform

# During the hyperparameter search many candidate models are intentionally
# weak (e.g. an ElasticNet with a tiny alpha, or one so strongly regularised it
# just predicts the mean). These produce expected, harmless warnings that flood
# the output and obscure the actual results, so we silence them here:
#  - ConvergenceWarning: coordinate descent for some ElasticNet alphas does not
#    fully converge within max_iter; the result is still usable for ranking.
#  - ConstantInputWarning: when a model predicts a constant value, Pearson is
#    undefined; pearson_scorer already handles this by returning 0.0.
warnings.filterwarnings('ignore', category=ConvergenceWarning)
try:
    from scipy.stats import ConstantInputWarning, NearConstantInputWarning
    warnings.filterwarnings('ignore', category=ConstantInputWarning)
    warnings.filterwarnings('ignore', category=NearConstantInputWarning)
except ImportError:  # older SciPy versions
    pass

# all-mpnet-base-v2 has a hard limit of 384 word-pieces (~250-300 words). A full
# clinical interview is thousands of words, so calling embedder.encode() on the
# whole transcript silently throws away everything past the first ~250 words.
# To actually look at the WHOLE interview we split each transcript into
# overlapping word windows, embed every window, and mean-pool the window
# embeddings into a single document vector. Mean-pooling over chunks is the
# standard, robust way to represent a long document with a short-context model
# and consistently beats truncating to the first window.
CHUNK_WORDS = 150       # words per window (safely under the model's token limit)
CHUNK_OVERLAP = 100      # overlap so context isn't cut mid-thought between windows


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return [""]
    step = max(1, chunk_words - overlap)
    chunks = [
        " ".join(words[i:i + chunk_words])
        for i in range(0, len(words), step)
    ]
    return chunks


def encode_long_texts(embedder, texts, show_progress_bar=True):
    doc_vectors = []
    for idx, text in enumerate(texts):
        if show_progress_bar:
            print(f"  Encoding transcript {idx + 1}/{len(texts)}", end='\r')
        chunks = chunk_text(text)
        chunk_embeddings = embedder.encode(chunks, show_progress_bar=False)
        doc_vectors.append(np.asarray(chunk_embeddings).mean(axis=0))
    if show_progress_bar:
        print()
    return np.vstack(doc_vectors)

def load_data(split_file):
    df_split = pd.read_csv(split_file)
    texts = []
    labels = []
    ids = []

    for _, row in df_split.iterrows():
        p_id = int(row['Participant_ID'])
        phq_score = row['PHQ_Score']

        transcript_path = os.path.join(DATA_DIR, f"{p_id}_P", f"{p_id}_Transcript.csv")
        if os.path.exists(transcript_path):
            try:
                df_trans = pd.read_csv(transcript_path)
                if 'Text' in df_trans.columns:
                    # Filter out NaN strings
                    text_data = df_trans['Text'].dropna().astype(str).tolist()
                    full_text = " ".join(text_data)
                    # We might want to truncate if it's too long, but sentence-transformers handles max_seq_length internally.
                    texts.append(full_text)
                    labels.append(phq_score)
                    ids.append(p_id)
            except Exception as e:
                print(f"Error reading {transcript_path}: {e}")

    return ids, texts, labels

def main():
    print("Loading training data...")
    train_ids, train_texts, train_labels = load_data(os.path.join(LABELS_DIR, 'train_split.csv'))

    print("Loading development data...")
    dev_ids, dev_texts, dev_labels = load_data(os.path.join(LABELS_DIR, 'dev_split.csv'))

    print(f"Loaded {len(train_texts)} train transcripts, {len(dev_texts)} dev transcripts.")

    print("Loading SentenceTransformer model (all-mpnet-base-v2)...")
    embedder = SentenceTransformer('all-mpnet-base-v2') # tested a lot of others too, except for the large token ones, this one worked the best

    print("Encoding train texts (chunked over the full transcript)...")
    train_embeddings = encode_long_texts(embedder, train_texts)

    print("Encoding dev texts (chunked over the full transcript)...")
    dev_embeddings = encode_long_texts(embedder, dev_texts)
    
    # Combine train and dev for cross-validation (same approach as the
    # traditional TF-IDF script): we no longer report separate train/dev
    # numbers but a single cross-validated estimate on the combined set.
    X_train_dev = np.vstack([train_embeddings, dev_embeddings])
    y_train_dev = np.concatenate([train_labels, dev_labels])
    
    # Scale embeddings
    scaler = StandardScaler()
    X_train_dev_scaled = scaler.fit_transform(X_train_dev)
    
    scoring = {
        'MAE': 'neg_mean_absolute_error',
        'RMSE': 'neg_root_mean_squared_error',
        'Pearson': make_scorer(pearson_scorer)
    }

    models_to_run = [
        {
            'name': 'Ridge',
            'estimator': Ridge(random_state=42),
            'param_dist': {
                'alpha': loguniform(1e-4, 1e3)
            },
            'n_iter': 1000
        },
        {
            'name': 'ElasticNet',
            'estimator': ElasticNet(max_iter=5000, random_state=42),
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
            'n_iter': 20
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
            'n_iter': 20,
            'needs_dense': True
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
    
    results = []
    print("\nTraining and tuning models with stratified cross-validation...")
    for model_cfg in models_to_run:
        print(f"\nRunning {model_cfg['name']}...")
        result = run_random_search(model_cfg, X_train_dev_scaled, y_train_dev, scoring)
        results.append(result)
    
    print("\n--- Cross-Validation Summary on Combined Train+Dev ---")
    print(f"{'Model':<14} | {'CV MAE':<8} | {'CV RMSE':<8} | {'CV Pearson':<10}")
    print("-" * 50)
    for result in results:
        print(
            f"{result['name']:<14} | {result['MAE']:<8.4f} | "
            f"{result['RMSE']:<8.4f} | {result['Pearson']:<10.4f}"
        )
    
    baseline_mae = mean_absolute_error(y_train_dev, [np.mean(y_train_dev)] * len(y_train_dev))
    print(f"\nBaseline (predict-the-mean) MAE: {baseline_mae:.4f}")

    # Visualize predictions of the best model (lowest CV MAE). Because we only
    # have a single cross-validated set (no held-out test split), we use
    # out-of-fold predictions via cross_val_predict so every sample gets a
    # prediction from a model that did not see it during training.
    best_result = min(results, key=model_score_for_picking)
    print(f"\nGenerating prediction visualization for best model ({best_result['name']})...")
    oof_preds = cross_val_predict(
        clone(best_result['model']),
        X_train_dev_scaled,
        y_train_dev,
        cv=5,
        n_jobs=-1
    )
    plot_predictions(
        y_train_dev,
        oof_preds,
        best_result['name'],
        '../media/transformer_best_model_predictions.png'
    )
    print("\nNote: Visualizing word importance for dense transformer embeddings is non-trivial compared to TF-IDF.")
    print("Please refer to the TF-IDF script output ('tfidf_feature_importance.png') for the most valuable words.")

if __name__ == "__main__":
    main()
