import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, Ridge, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, KFold
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


def pearson_corr(y_true, y_pred):
    # Guard against constant predictions (pearsonr is undefined / returns nan).
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    return pearsonr(y_true, y_pred)[0]


def pearson_scorer(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # If either side has (near) no variance, Pearson is undefined; treat as 0.
    # A slightly larger threshold also covers the near-constant predictions of
    # strongly regularised models that would otherwise trigger SciPy warnings.
    if np.std(y_pred) < 1e-8 or np.std(y_true) < 1e-8:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        corr, _ = pearsonr(y_true, y_pred)
    return 0.0 if np.isnan(corr) else corr


def plot_predictions(y_true, y_pred, model_name, out_path):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Sort samples by the actual PHQ score so the "real" values form a smooth
    # ascending reference line and the predictions can be compared against it.
    order = np.argsort(y_true)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    x = np.arange(len(y_true_sorted))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson = pearson_scorer(y_true, y_pred)

    plt.figure(figsize=(14, 7))

    # Actual values as a clear reference line.
    plt.plot(x, y_true_sorted, color='black', linewidth=2, label='Actual PHQ score')

    # Predictions as dots, with thin connectors to their corresponding actual
    # value to make the residual (error) for each sample visible at a glance.
    plt.vlines(x, y_true_sorted, y_pred_sorted, color='lightgray', linewidth=1, zorder=1)
    plt.scatter(x, y_pred_sorted, alpha=0.8, color='steelblue', edgecolors='k',
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

# all-mpnet-base-v2 has a hard limit of 384 word-pieces (~250-300 words). A full
# clinical interview is thousands of words, so calling embedder.encode() on the
# whole transcript silently throws away everything past the first ~250 words.
# To actually look at the WHOLE interview we split each transcript into
# overlapping word windows and embed every window. The base transformer script
# then mean-pools those window embeddings into one document vector.
#
# Here the DEEP-LEARNING model does NOT use any fixed pooling (mean/max/std).
# Instead it keeps the ragged per-window embeddings and learns, with an
# attention network, how much each window contributes to the final document
# representation (see AttentionRegressor). To make this LEAK-FREE, the attention
# network is re-trained from scratch inside every cross-validation fold on that
# fold's training rows only, so no sample's prediction is ever influenced by its
# own label (there is no supervised transform applied to the whole set up front,
# which is what caused the earlier leakage). The classic sklearn models are kept
# for comparison but they are ALSO fed vectors produced by a learned attention
# pooling layer (window weights, no regression head) rather than a plain mean -
# so no model uses mean pooling anymore. That pooling is done out-of-fold so it
# stays leak-free.
CHUNK_WORDS = 150       # words per window (safely under the model's token limit)
CHUNK_OVERLAP = 100      # overlap so context isn't cut mid-thought between windows

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    doc_chunk_embeddings = []
    for idx, text in enumerate(texts):
        if show_progress_bar:
            print(f"  Encoding transcript {idx + 1}/{len(texts)}", end='\r')
        chunks = chunk_text(text)
        chunk_embeddings = embedder.encode(chunks, show_progress_bar=False)
        doc_chunk_embeddings.append(np.asarray(chunk_embeddings, dtype=np.float32))
    if show_progress_bar:
        print()
    return doc_chunk_embeddings


# ---------------------------------------------------------------------------
# Deep-learning model: attention pooling over windows + regression head,
# trained END-TO-END against PHQ.
#
# This is the requested "neural net instead of mean/std/max": rather than fixing
# how windows are combined, the network learns an attention weight for each
# window (softmax over windows), forms a weighted-sum document vector, and
# regresses PHQ from it. Because it is trained fresh inside each CV fold on that
# fold's training rows only (see cross_validate_attention), it never sees the
# label of a sample it predicts -> NO leakage, unlike the earlier design that
# fit one aggregator on all of train and transformed those same rows.
#
# Regularisation choices that matter on this small (~190 transcript) dataset:
#  - per-feature standardisation fit on the fold's training windows only,
#  - dropout + weight decay,
#  - an internal validation split with EARLY STOPPING (restore best weights),
# which together keep the network from overfitting the way the old supervised
# aggregator did.
# ---------------------------------------------------------------------------
class _AttentionPoolNet(nn.Module):
    def __init__(self, embed_dim, attn_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()
        # Scores each window; softmax over windows turns scores into weights.
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1),
        )
        # Regression head on top of the attention-pooled document vector.
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def pool(self, chunk_embs, mask):
        # chunk_embs: (B, T, D); mask: (B, T) with 1 for real windows, 0 for pad.
        # Learns an attention weight per window and returns the weighted-sum
        # document vector (B, D). This is the learned replacement for mean/max/
        # std pooling: the "deep-learning layer with window weights" that ALL
        # models are fed from.
        scores = self.attn(chunk_embs).squeeze(-1)              # (B, T)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=1)                 # (B, T)
        pooled = torch.bmm(weights.unsqueeze(1), chunk_embs).squeeze(1)  # (B, D)
        return pooled

    def forward(self, chunk_embs, mask):
        pooled = self.pool(chunk_embs, mask)                   # (B, D)
        pred = self.head(pooled).squeeze(-1)                   # (B,)
        return pred


class AttentionRegressor(BaseEstimator, RegressorMixin):
    """End-to-end attention-pooling regressor. `fit`/`predict` take a LIST of
    per-window embedding matrices (one ragged array per transcript)."""

    def __init__(self, attn_dim=128, hidden_dim=128, dropout=0.3, lr=1e-5,
                 epochs=400, batch_size=16, weight_decay=1e-5,
                 val_frac=0.2, patience=20, random_state=42):
        self.attn_dim = attn_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.val_frac = val_frac
        self.patience = patience
        self.random_state = random_state

    def _pad_batch(self, batch_lists):
        max_t = max(len(c) for c in batch_lists)
        X = np.zeros((len(batch_lists), max_t, self.embed_dim_), dtype=np.float32)
        mask = np.zeros((len(batch_lists), max_t), dtype=np.float32)
        for i, emb in enumerate(batch_lists):
            t = len(emb)
            X[i, :t] = (emb - self.mean_) / self.std_
            mask[i, :t] = 1.0
        return torch.tensor(X, device=DEVICE), torch.tensor(mask, device=DEVICE)

    def fit(self, chunk_lists, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        y = np.asarray(y, dtype=np.float32)
        self.embed_dim_ = chunk_lists[0].shape[1]

        # Per-feature standardisation fit on THIS fold's training windows only.
        all_windows = np.vstack(chunk_lists)
        self.mean_ = all_windows.mean(axis=0)
        self.std_ = all_windows.std(axis=0) + 1e-6

        # Internal train/validation split for early stopping.
        n = len(chunk_lists)
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        n_val = max(1, int(round(self.val_frac * n)))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        self.net_ = _AttentionPoolNet(self.embed_dim_, self.attn_dim,
                                      self.hidden_dim, self.dropout).to(DEVICE)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        best_val = float('inf')
        best_state = None
        bad_epochs = 0
        for _ in range(self.epochs):
            self.net_.train()
            batch_perm = tr_idx[rng.permutation(len(tr_idx))]
            for i in range(0, len(batch_perm), self.batch_size):
                idx = batch_perm[i:i + self.batch_size]
                X, mask = self._pad_batch([chunk_lists[j] for j in idx])
                yt = torch.tensor(y[idx], device=DEVICE)
                opt.zero_grad()
                pred = self.net_(X, mask)
                loss = loss_fn(pred, yt)
                loss.backward()
                opt.step()

            # Early-stopping check on the held-out internal validation split.
            self.net_.eval()
            with torch.no_grad():
                Xv, mv = self._pad_batch([chunk_lists[j] for j in val_idx])
                yv = torch.tensor(y[val_idx], device=DEVICE)
                val_loss = loss_fn(self.net_(Xv, mv), yv).item()
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().clone()
                              for k, v in self.net_.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def predict(self, chunk_lists):
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(chunk_lists), self.batch_size):
                batch_lists = chunk_lists[i:i + self.batch_size]
                X, mask = self._pad_batch(batch_lists)
                out.append(self.net_(X, mask).cpu().numpy())
        return np.concatenate(out)

    def transform(self, chunk_lists):
        """Return the learned attention-pooled document vector per transcript
        (the weighted-sum of window embeddings), WITHOUT the regression head.
        This is what feeds the classic sklearn models: instead of mean-pooling
        the windows, we pool them with the learned attention weights. The head
        is only used during `fit` to give the attention layer a training
        signal; it is dropped here."""
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(chunk_lists), self.batch_size):
                batch_lists = chunk_lists[i:i + self.batch_size]
                X, mask = self._pad_batch(batch_lists)
                out.append(self.net_.pool(X, mask).cpu().numpy())
        return np.vstack(out)


def cross_validate_attention(chunk_lists, y, n_splits=5, random_state=42):
    """Leak-free cross-validation for the attention regressor: a fresh network
    is trained on each fold's training rows and used to predict the held-out
    rows, so no sample is ever predicted by a model that saw its own label.
    Returns the out-of-fold predictions (one per sample)."""
    y = np.asarray(y, dtype=np.float32)
    n = len(chunk_lists)
    oof = np.zeros(n, dtype=np.float32)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(np.arange(n))):
        print(f"  [attention] fold {fold + 1}/{n_splits}: "
              f"train {len(tr_idx)}, held-out {len(val_idx)}")
        model = AttentionRegressor()
        model.fit([chunk_lists[j] for j in tr_idx], y[tr_idx])
        oof[val_idx] = model.predict([chunk_lists[j] for j in val_idx])
    return oof


def attention_pool_oof(chunk_lists, y, n_splits=5, random_state=42):
    """Produce ONE document vector per transcript using the LEARNED attention
    pooling (window weights), in a leak-free out-of-fold manner. This is the
    replacement for `mean_pool`: every model is now fed vectors that were pooled
    with a deep-learning layer of window weights instead of a plain mean.

    To learn the window weights the attention layer needs a training signal, so
    a small regression head is attached during `fit` (the same architecture as
    the AttentionNet model). The head itself is DROPPED for pooling: `transform`
    returns only the attention-weighted document vector. Because the pooled
    vector is a convex combination of the window embeddings, vectors produced by
    different per-fold poolers all live in the same embedding space and are
    directly comparable.

    Leak-freedom: each transcript's vector is produced by a pooler trained ONLY
    on the other folds' rows/labels, so no sample's features ever encode its own
    label (this is what avoids the leakage of the earlier fit-on-all design)."""
    y = np.asarray(y, dtype=np.float32)
    n = len(chunk_lists)
    embed_dim = chunk_lists[0].shape[1]
    pooled = np.zeros((n, embed_dim), dtype=np.float32)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(np.arange(n))):
        print(f"  [attn-pool] fold {fold + 1}/{n_splits}: "
              f"train {len(tr_idx)}, pooled {len(val_idx)}")
        pooler = AttentionRegressor()
        pooler.fit([chunk_lists[j] for j in tr_idx], y[tr_idx])
        pooled[val_idx] = pooler.transform([chunk_lists[j] for j in val_idx])
    return pooled


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

    print("Encoding train texts (per-window embeddings over the full transcript)...")
    train_chunk_embeddings = encode_long_texts(embedder, train_texts)

    print("Encoding dev texts (per-window embeddings over the full transcript)...")
    dev_chunk_embeddings = encode_long_texts(embedder, dev_texts)

    # Combine train and dev for cross-validation (same approach as the
    # traditional TF-IDF script): we no longer report separate train/dev
    # numbers but a single cross-validated estimate on the combined set.
    all_chunk_embeddings = train_chunk_embeddings + dev_chunk_embeddings
    y_train_dev = np.concatenate([train_labels, dev_labels])

    # Classic sklearn models need one fixed vector per transcript. Instead of a
    # plain mean-pooling, we now pool the windows with a LEARNED deep-learning
    # layer of window weights (attention), exactly like the AttentionNet model
    # but WITHOUT its regression head - the head is only used to train the
    # attention weights and is dropped for pooling. This is done out-of-fold so
    # each transcript's vector is produced by a pooler that never saw its own
    # label (leak-free). Every model below is therefore fed attention-pooled
    # document vectors rather than mean-pooled ones.
    print("\nAttention-pooling windows into document vectors "
          "(leak-free out-of-fold) to feed the classic models...")
    X_train_dev = attention_pool_oof(all_chunk_embeddings, y_train_dev)

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
    # print("\nTraining and tuning models with cross-validation...")
    # for model_cfg in models_to_run:
    #     print(f"\nRunning {model_cfg['name']}...")
    #     result = run_random_search(model_cfg, X_train_dev_scaled, y_train_dev, scoring)
    #     results.append(result)

    # Deep-learning model requested by the issue: the end-to-end attention
    # network that LEARNS how to pool the windows (instead of mean/max/std). It
    # consumes the ragged per-window embeddings and is evaluated with leak-free
    # cross-validation (a fresh network trained inside every fold). We keep its
    # out-of-fold predictions for both the summary metrics and the final plot.
    print("\nRunning AttentionNet (leak-free out-of-fold cross-validation)...")
    attn_oof = cross_validate_attention(all_chunk_embeddings, y_train_dev)
    attn_result = {
        'name': 'AttentionNet',
        'MAE': mean_absolute_error(y_train_dev, attn_oof),
        'RMSE': np.sqrt(mean_squared_error(y_train_dev, attn_oof)),
        'Pearson': pearson_scorer(y_train_dev, attn_oof),
        'model': None,
        'oof_preds': attn_oof
    }
    print(f"CV MAE: {attn_result['MAE']:.4f}, RMSE: {attn_result['RMSE']:.4f}, "
          f"Pearson: {attn_result['Pearson']:.4f}")
    results.append(attn_result)

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
    if best_result.get('oof_preds') is not None:
        # The attention network already produced leak-free out-of-fold
        # predictions during its own cross-validation; reuse them directly.
        oof_preds = best_result['oof_preds']
    else:
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
        '../media/full_deep_learning_best_model_predictions.png'
    )
    print("\nNote: Visualizing word importance for dense transformer embeddings is non-trivial compared to TF-IDF.")
    print("Please refer to the TF-IDF script output ('tfidf_feature_importance.png') for the most valuable words.")

if __name__ == "__main__":
    main()
