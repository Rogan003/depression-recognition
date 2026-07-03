import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

warnings.filterwarnings('ignore', category=ConvergenceWarning)
try:
    from scipy.stats import ConstantInputWarning, NearConstantInputWarning
    warnings.filterwarnings('ignore', category=ConstantInputWarning)
    warnings.filterwarnings('ignore', category=NearConstantInputWarning)
except ImportError:  # older SciPy versions
    pass

# A simple, interpretable emotion-based approach: instead of generic semantic
# embeddings we use an emotion classifier as a feature extractor. Each long
# transcript is chunked into overlapping word windows, the emotion model
# produces a probability over its emotion classes per chunk, and we mean-pool
# those probabilities into one small, interpretable emotion vector per
# transcript.
EMOTION_MODEL = 'j-hartmann/emotion-english-distilroberta-base'

# Mirror the transformer script: split each transcript into overlapping word
# windows so context isn't cut mid-thought between windows, then mean-pool the
# per-window emotion distributions into a single document vector.
CHUNK_WORDS = 300       # words per window (safely under the model's token limit)
CHUNK_OVERLAP = 200     # overlap so context isn't cut mid-thought between windows


def pearson_corr(y_true, y_pred):
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    return pearsonr(y_true, y_pred)[0]


def pearson_scorer(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.std(y_pred) < 1e-8 or np.std(y_true) < 1e-8:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        corr, _ = pearsonr(y_true, y_pred)
    return 0.0 if np.isnan(corr) else corr


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


@torch.no_grad()
def emotion_probs(tokenizer, model, device, chunks, batch_size=16):
    probs = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
        probs.append(model(**enc).logits)
    return np.vstack(probs)


def encode_long_texts(tokenizer, model, device, texts, show_progress_bar=True):
    # Mirror the transformer script: instead of a generic embedder we use the
    # emotion classifier as the encoder. Each transcript is chunked into
    # overlapping word windows, the model produces a probability over its
    # emotion classes per window, and we mean-pool those distributions into a
    # single small, interpretable emotion vector per transcript.
    doc_vectors = []
    for idx, text in enumerate(texts):
        if show_progress_bar:
            print(f"  Encoding transcript {idx + 1}/{len(texts)}", end='\r')
        chunks = chunk_text(text)
        chunk_probs = emotion_probs(tokenizer, model, device, chunks)
        doc_vectors.append(chunk_probs.mean(axis=0))
    if show_progress_bar:
        print()
    return np.vstack(doc_vectors)


def plot_predictions(y_true, y_pred, model_name, out_path):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    order = np.argsort(y_true)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    x = np.arange(len(y_true_sorted))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson = pearson_scorer(y_true, y_pred)

    plt.figure(figsize=(14, 7))
    plt.plot(x, y_true_sorted, color='black', linewidth=2, label='Actual PHQ score')
    plt.vlines(x, y_true_sorted, y_pred_sorted, color='lightgray', linewidth=1, zorder=1)
    plt.scatter(x, y_pred_sorted, alpha=0.8, color='darkorange', edgecolors='k',
                zorder=2, label='Predicted PHQ score')
    plt.xlabel('Samples (sorted by actual PHQ score)')
    plt.ylabel('PHQ score')
    plt.title(
        f'Predictions vs Actual ({model_name})\n'
        f'MAE={mae:.3f}, RMSE={rmse:.3f}, Pearson={pearson:.3f}'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved prediction visualization to '{out_path}'")


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


DATA_DIR = '../dataset/wwwedaic/data'
LABELS_DIR = '../dataset/wwwedaic/labels'


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
                    text_data = df_trans['Text'].dropna().astype(str).tolist()
                    full_text = " ".join(text_data)
                    texts.append(full_text)
                    labels.append(phq_score)
                    ids.append(p_id)
            except Exception as e:
                print(f"Error reading {transcript_path}: {e}")

    return ids, texts, labels


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


def main():
    print("Loading training data...")
    train_ids, train_texts, train_labels = load_data(os.path.join(LABELS_DIR, 'train_split.csv'))

    print("Loading development data...")
    dev_ids, dev_texts, dev_labels = load_data(os.path.join(LABELS_DIR, 'dev_split.csv'))

    print(f"Loaded {len(train_texts)} train transcripts, {len(dev_texts)} dev transcripts.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading emotion model '{EMOTION_MODEL}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL).to(device)
    model.eval()
    # Human-readable emotion labels (order matches the model's output).
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(f"Emotion classes: {labels}")

    print("Encoding train texts (chunked over the full transcript)...")
    train_features = encode_long_texts(tokenizer, model, device, train_texts)

    print("Encoding dev texts (chunked over the full transcript)...")
    dev_features = encode_long_texts(tokenizer, model, device, dev_texts)

    X_train_dev = np.vstack([train_features, dev_features])
    y_train_dev = np.concatenate([train_labels, dev_labels])

    scaler = StandardScaler()
    X_train_dev_scaled = scaler.fit_transform(X_train_dev)

    scoring = {
        'MAE': 'neg_mean_absolute_error',
        'RMSE': 'neg_root_mean_squared_error',
        'Pearson': make_scorer(pearson_scorer)
    }

    results = []
    print("\nTraining and tuning models with cross-validation...")
    for model_cfg in get_models_to_run():
        print(f"\nRunning {model_cfg['name']}...")
        result = run_random_search(model_cfg, X_train_dev_scaled, y_train_dev, scoring)
        results.append(result)

    print("\n--- Cross-Validation Summary (Emotion features) ---")
    print(f"{'Model':<14} | {'CV MAE':<8} | {'CV RMSE':<8} | {'CV Pearson':<10}")
    print("-" * 50)
    for result in results:
        print(
            f"{result['name']:<14} | {result['MAE']:<8.4f} | "
            f"{result['RMSE']:<8.4f} | {result['Pearson']:<10.4f}"
        )

    baseline_mae = mean_absolute_error(y_train_dev, [np.mean(y_train_dev)] * len(y_train_dev))
    print(f"\nBaseline (predict-the-mean) MAE: {baseline_mae:.4f}")

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
        f"Emotion + {best_result['name']}",
        '../media/emotion_best_model_predictions.png'
    )


if __name__ == "__main__":
    main()
