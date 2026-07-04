import os

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import common

common.suppress_expected_warnings()

CHUNK_WORDS = 150
CHUNK_OVERLAP = 100

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def encode_long_texts(embedder, texts, show_progress_bar=True):
    doc_chunk_embeddings = []
    for idx, text in enumerate(texts):
        if show_progress_bar:
            print(f"  Encoding transcript {idx + 1}/{len(texts)}", end='\r')
        chunks = common.chunk_text(text, CHUNK_WORDS, CHUNK_OVERLAP)
        chunk_embeddings = embedder.encode(chunks, show_progress_bar=False)
        doc_chunk_embeddings.append(np.asarray(chunk_embeddings, dtype=np.float32))
    if show_progress_bar:
        print()
    return doc_chunk_embeddings


class _AttentionPoolNet(nn.Module):
    def __init__(self, embed_dim, attn_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(embed_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1),
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def pool(self, chunk_embs, mask):
        scores = self.attn(chunk_embs).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), chunk_embs).squeeze(1)
        return pooled

    def forward(self, chunk_embs, mask):
        pooled = self.pool(chunk_embs, mask)
        pred = self.head(pooled).squeeze(-1)
        return pred


class AttentionRegressor(BaseEstimator, RegressorMixin):
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

        all_windows = np.vstack(chunk_lists)
        self.mean_ = all_windows.mean(axis=0)
        self.std_ = all_windows.std(axis=0) + 1e-6

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
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(chunk_lists), self.batch_size):
                batch_lists = chunk_lists[i:i + self.batch_size]
                X, mask = self._pad_batch(batch_lists)
                out.append(self.net_.pool(X, mask).cpu().numpy())
        return np.vstack(out)


def cross_validate_attention(chunk_lists, y, n_splits=5, random_state=42):
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


def main():
    print("Loading training data...")
    train_ids, train_texts, train_labels = common.load_data(
        os.path.join(common.LABELS_DIR, 'train_split.csv'))

    print("Loading development data...")
    dev_ids, dev_texts, dev_labels = common.load_data(
        os.path.join(common.LABELS_DIR, 'dev_split.csv'))

    print("Loading test data...")
    test_ids, test_texts, test_labels = common.load_data(
        os.path.join(common.LABELS_DIR, 'test_split.csv'))

    print(f"Loaded {len(train_texts)} train transcripts, {len(dev_texts)} dev transcripts, "
          f"{len(test_texts)} test transcripts.")

    print("Loading SentenceTransformer model (all-mpnet-base-v2)...")
    embedder = SentenceTransformer('all-mpnet-base-v2')

    print("Encoding train texts (per-window embeddings over the full transcript)...")
    train_chunk_embeddings = encode_long_texts(embedder, train_texts)

    print("Encoding dev texts (per-window embeddings over the full transcript)...")
    dev_chunk_embeddings = encode_long_texts(embedder, dev_texts)

    print("Encoding test texts (per-window embeddings over the full transcript)...")
    test_chunk_embeddings = encode_long_texts(embedder, test_texts)

    all_chunk_embeddings = train_chunk_embeddings + dev_chunk_embeddings
    y_train_dev = np.concatenate([train_labels, dev_labels])
    y_test = np.asarray(test_labels, dtype=float)

    X = attention_pool_oof(all_chunk_embeddings, y_train_dev)
    # Scaling is done leak-free inside the cross-validation pipeline.
    results = common.cross_validate_models(
            common.default_models(bayesian_needs_dense=True),
            X, y_train_dev, scaler=StandardScaler()
    )

    # Pool the held-out test transcripts with an attention pooler fit on the full
    # train+dev set (fit once, no leakage from the test set).
    print("\nPooling test transcripts for zoo evaluation...")
    test_pooler = AttentionRegressor()
    test_pooler.fit(all_chunk_embeddings, y_train_dev)
    X_test_pooled = test_pooler.transform(test_chunk_embeddings)

    print("\nRunning AttentionNet (leak-free out-of-fold cross-validation)...")
    attn_oof = cross_validate_attention(all_chunk_embeddings, y_train_dev)
    attn_result = {
        'name': 'AttentionNet',
        'MAE': mean_absolute_error(y_train_dev, attn_oof),
        'RMSE': np.sqrt(mean_squared_error(y_train_dev, attn_oof)),
        'Pearson': common.pearson_scorer(y_train_dev, attn_oof),
        'model': None,
        'oof_preds': attn_oof
    }
    print(f"CV MAE: {attn_result['MAE']:.4f}, RMSE: {attn_result['RMSE']:.4f}, "
          f"Pearson: {attn_result['Pearson']:.4f}")
    results.append(attn_result)

    common.print_cv_summary(results, 'Cross-Validation Summary on Combined Train+Dev')
    common.print_baseline(y_train_dev)

    best_result = min(results, key=common.model_score_for_picking)

    test_metrics = common.evaluate_on_test(results, X_test_pooled, y_test)

    attn_test_preds = test_pooler.predict(test_chunk_embeddings)
    attn_test_metric = {
        'name': 'AttentionNet',
        'MAE': mean_absolute_error(y_test, attn_test_preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, attn_test_preds)),
        'Pearson': common.pearson_corr(y_test, attn_test_preds),
        'preds': attn_test_preds,
    }
    print(f"{attn_test_metric['name']:<14} | {attn_test_metric['MAE']:<8.4f} | "
          f"{attn_test_metric['RMSE']:<8.4f} | {attn_test_metric['Pearson']:<10.4f}")
    test_metrics.append(attn_test_metric)

    best_test = min(test_metrics, key=common.model_score_for_picking)
    print(f"\nGenerating test-set prediction visualization for best model "
          f"({best_test['name']})...")
    common.plot_predictions(
        y_test,
        best_test['preds'],
        best_test['name'],
        common.media_path('full_deep_learning_best_model_test_predictions.png')
    )

    print(f"\nGenerating out-of-fold prediction visualization for best model "
          f"({best_result['name']})...")
    if best_result.get('oof_preds') is not None:
        oof_preds = best_result['oof_preds']
    else:
        X_pooled = attention_pool_oof(all_chunk_embeddings, y_train_dev)
        oof_preds = common.out_of_fold_predictions(best_result['model'], X_pooled, y_train_dev)
    common.plot_predictions(
        y_train_dev,
        oof_preds,
        best_result['name'],
        common.media_path('full_deep_learning_best_model_predictions.png')
    )
    print("\nNote: Visualizing word importance for dense transformer embeddings is non-trivial compared to TF-IDF.")
    print("Please refer to the TF-IDF script output ('tfidf_feature_importance.png') for the most valuable words.")


if __name__ == "__main__":
    main()
