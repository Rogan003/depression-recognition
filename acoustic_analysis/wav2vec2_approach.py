import importlib.util
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _load_text_common():
    common_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "text_analysis", "common.py")
    spec = importlib.util.spec_from_file_location("text_common", common_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load_text_common()
common.suppress_expected_warnings()

WIN_LENGTH_S = 30.0
HOP_LENGTH_S = 20.0

DEVICE = 'cpu'


def extract_wav2vec2_features(file_path, feature_extractor, model):
    sr = 16000
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    if len(audio) == 0:
        return None

    win_length = int(WIN_LENGTH_S * sr)
    hop_length = int(HOP_LENGTH_S * sr)

    windows = [audio[s:s + win_length]
               for s in range(0, len(audio) - win_length + 1, hop_length)]
    if len(windows) == 0:
        windows = [audio]

    model.eval()
    with torch.inference_mode():
        inputs = feature_extractor(
            windows,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        outputs = model(**inputs)
        return outputs.last_hidden_state


def load_data(csv_path, feature_extractor, model, model_name, max_samples=None):
    df = pd.read_csv(csv_path)

    X = []
    y = []

    safe_model_name = model_name.replace("/", "_")
    cache_dir = os.path.join("../features", safe_model_name)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    count = 0
    for index, row in df.iterrows():
        if max_samples and count >= max_samples:
            break

        participant_id = int(row['Participant_ID'])
        score = row['PHQ_Score']
        file_path = f"../dataset/wwwedaic/data/{participant_id}_P/{participant_id}_AUDIO.wav"

        cache_file = os.path.join(cache_dir, f"{participant_id}_{WIN_LENGTH_S}_{HOP_LENGTH_S}.npy")

        if os.path.exists(cache_file):
            print(f"Loading cached features for {participant_id}...")
            features = np.load(cache_file)
            X.append(features)
            y.append(score)
            count += 1
        elif os.path.exists(file_path):
            print(f"Processing {file_path}...")
            features = extract_wav2vec2_features(file_path, feature_extractor, model)
            if features is not None:
                np.save(cache_file, features)
                X.append(np.asarray(features))
                y.append(score)
                count += 1
            else:
                print(f"Warning: no features extracted from {file_path}.")
        else:
            print(f"Warning: {file_path} not found.")

    return X, np.array(y, dtype=np.float32)


def frames_to_window_embeddings(feature_list):
    window_lists = []
    for features in feature_list:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=1)
        elif arr.ndim == 1:
            arr = arr[None, :]
        window_lists.append(arr.astype(np.float32))
    return window_lists


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

    def pool(self, window_embs, mask):
        scores = self.attn(window_embs).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), window_embs).squeeze(1)
        return pooled

    def forward(self, window_embs, mask):
        pooled = self.pool(window_embs, mask)
        pred = self.head(pooled).squeeze(-1)
        return pred


class AttentionRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, attn_dim=128, hidden_dim=128, dropout=0.3, lr=1e-4,
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

    def fit(self, window_lists, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        y = np.asarray(y, dtype=np.float32)
        self.embed_dim_ = window_lists[0].shape[1]

        all_windows = np.vstack(window_lists)
        self.mean_ = all_windows.mean(axis=0)
        self.std_ = all_windows.std(axis=0) + 1e-6

        n = len(window_lists)
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
                X, mask = self._pad_batch([window_lists[j] for j in idx])
                yt = torch.tensor(y[idx], device=DEVICE)
                opt.zero_grad()
                pred = self.net_(X, mask)
                loss = loss_fn(pred, yt)
                loss.backward()
                opt.step()

            self.net_.eval()
            with torch.no_grad():
                Xv, mv = self._pad_batch([window_lists[j] for j in val_idx])
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

    def predict(self, window_lists):
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(window_lists), self.batch_size):
                batch_lists = window_lists[i:i + self.batch_size]
                X, mask = self._pad_batch(batch_lists)
                out.append(self.net_(X, mask).cpu().numpy())
        return np.concatenate(out)

    def transform(self, window_lists):
        self.net_.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(window_lists), self.batch_size):
                batch_lists = window_lists[i:i + self.batch_size]
                X, mask = self._pad_batch(batch_lists)
                out.append(self.net_.pool(X, mask).cpu().numpy())
        return np.vstack(out)


def cross_validate_attention(window_lists, y, n_splits=5, random_state=42):
    y = np.asarray(y, dtype=np.float32)
    n = len(window_lists)
    oof = np.zeros(n, dtype=np.float32)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(np.arange(n))):
        print(f"  [attention] fold {fold + 1}/{n_splits}: "
              f"train {len(tr_idx)}, held-out {len(val_idx)}")
        model = AttentionRegressor()
        model.fit([window_lists[j] for j in tr_idx], y[tr_idx])
        oof[val_idx] = model.predict([window_lists[j] for j in val_idx])
    return oof


def attention_pool_oof(window_lists, y, n_splits=5, random_state=42):
    """Leak-free out-of-fold attention pooling -> fixed-size feature matrix."""
    y = np.asarray(y, dtype=np.float32)
    n = len(window_lists)
    embed_dim = window_lists[0].shape[1]
    pooled = np.zeros((n, embed_dim), dtype=np.float32)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(np.arange(n))):
        print(f"  [attn-pool] fold {fold + 1}/{n_splits}: "
              f"train {len(tr_idx)}, pooled {len(val_idx)}")
        pooler = AttentionRegressor()
        pooler.fit([window_lists[j] for j in tr_idx], y[tr_idx])
        pooled[val_idx] = pooler.transform([window_lists[j] for j in val_idx])
    return pooled


def main():
    print(f"Using device: {DEVICE}")

    model_name = "facebook/wav2vec2-base"
    print(f"Loading {model_name}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(DEVICE)

    print("Loading training data...")
    X_train, y_train = load_data("../dataset/wwwedaic/labels/train_split.csv",
                                 feature_extractor, model, model_name)

    print("Loading validation data...")
    X_val, y_val = load_data("../dataset/wwwedaic/labels/dev_split.csv",
                             feature_extractor, model, model_name)

    print("Loading test data...")
    X_test, y_test = load_data("../dataset/wwwedaic/labels/test_split.csv",
                               feature_extractor, model, model_name)

    if len(X_train) == 0 or len(X_val) == 0:
        print("Not enough data to train. Exiting.")
        return

    # Collapse the per-frame time axis -> variable-length window-embedding lists.
    print("\nReducing frame-level features to per-window embeddings...")
    train_windows = frames_to_window_embeddings(X_train)
    dev_windows = frames_to_window_embeddings(X_val)
    test_windows = frames_to_window_embeddings(X_test)

    all_windows = train_windows + dev_windows
    y_train_dev = np.concatenate([y_train, y_val])
    y_test = np.asarray(y_test, dtype=float)

    print("\n=== Approach 1: attention-pooled features + model zoo ===")
    X = attention_pool_oof(all_windows, y_train_dev)
    results = common.cross_validate_models(
        common.default_models(bayesian_needs_dense=True),
        X, y_train_dev, scaler=StandardScaler()
    )

    print("\nPooling test samples for zoo evaluation...")
    test_pooler = AttentionRegressor()
    test_pooler.fit(all_windows, y_train_dev)
    X_test_pooled = test_pooler.transform(test_windows)

    print("\n=== Approach 2: end-to-end AttentionNet regressor ===")
    print("Running AttentionNet (leak-free out-of-fold cross-validation)...")
    attn_oof = cross_validate_attention(all_windows, y_train_dev)
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
    print(f"\n>>> Best model picked: {best_result['name']} <<<")

    if len(X_test) > 0:
        test_metrics = common.evaluate_on_test(results, X_test_pooled, y_test)

        attn_test_preds = test_pooler.predict(test_windows)
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
        common.ensure_media_dir()
        common.plot_predictions(
            y_test,
            best_test['preds'],
            best_test['name'],
            common.media_path('wav2vec2_best_model_test_predictions.png')
        )

    print(f"\nGenerating out-of-fold prediction visualization for best model "
          f"({best_result['name']})...")
    common.ensure_media_dir()
    if best_result.get('oof_preds') is not None:
        oof_preds = best_result['oof_preds']
    else:
        X_pooled = attention_pool_oof(all_windows, y_train_dev)
        oof_preds = common.out_of_fold_predictions(best_result['model'], X_pooled, y_train_dev)
    common.plot_predictions(
        y_train_dev,
        oof_preds,
        best_result['name'],
        common.media_path('wav2vec2_best_model_predictions.png')
    )


if __name__ == "__main__":
    main()
