import os

import numpy as np
import torch
import torch.nn as nn

import common

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
LABEL = "Transformer Pool"

WIN_LENGTH_S = 5.0
HOP_LENGTH_S = 3.0

SEED = 42
ENSEMBLE_SIZE = 5
BATCH_SIZE = 8


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


class WindowTransformerRegressor(nn.Module):
    def __init__(self, in_dim, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True)

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.attn = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, mask):
        h = self.proj(x)
        pad_mask = mask == 0
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        scores = self.attn(h).squeeze(-1)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (h * weights).sum(dim=1)
        return self.head(pooled).squeeze(-1)


def _pad_batch(seqs, device):
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    dim = seqs[0].shape[1]
    x = np.zeros((len(seqs), max_len, dim), dtype=np.float32)
    mask = np.zeros((len(seqs), max_len), dtype=np.float32)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
        mask[i, :len(s)] = 1.0
    return (torch.from_numpy(x).to(device),
            torch.from_numpy(mask).to(device))


def _standardize(seqs, mean, std):
    return [((s - mean) / std).astype(np.float32) for s in seqs]


def _feature_stats(windows_per_file):
    dim = windows_per_file[0].shape[1]
    count = 0
    s1 = np.zeros(dim, dtype=np.float64)
    s2 = np.zeros(dim, dtype=np.float64)
    for w in windows_per_file:
        s1 += w.sum(axis=0)
        s2 += (w.astype(np.float64) ** 2).sum(axis=0)
        count += len(w)
    mean = s1 / count
    var = np.maximum(s2 / count - mean ** 2, 0.0)
    std = np.sqrt(var) + 1e-8
    return mean.astype(np.float32)[None, :], std.astype(np.float32)[None, :]


def _batched_loss(model, seqs, y, mean, std, device, criterion, batch_size,
                  optimizer=None):
    total, n = 0.0, len(seqs)
    for s in range(0, n, batch_size):
        batch = _standardize(seqs[s:s + batch_size], mean, std)
        x, mask = _pad_batch(batch, device)
        yb = torch.tensor(y[s:s + batch_size], dtype=torch.float32, device=device)
        if optimizer is not None:
            model.train()
            optimizer.zero_grad()
            loss = criterion(model(x, mask), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        else:
            with torch.no_grad():
                loss = criterion(model(x, mask), yb)
        total += loss.item() * len(batch)
        del x, mask, yb
    return total / max(1, n)


def train_one(train_seqs, y_train, device, mean, std, seed=SEED,
              epochs=300, lr=5e-4, val_fraction=0.2, patience=30,
              batch_size=BATCH_SIZE):
    set_seed(seed)
    idx = np.random.permutation(len(train_seqs))
    n_val = max(1, int(len(train_seqs) * val_fraction))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    tr_seqs = [train_seqs[i] for i in tr_idx]
    val_seqs = [train_seqs[i] for i in val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    model = WindowTransformerRegressor(train_seqs[0].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.SmoothL1Loss()

    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        perm = np.random.permutation(len(tr_seqs))
        _batched_loss(model, [tr_seqs[i] for i in perm], y_tr[perm],
                      mean, std, device, criterion, batch_size, optimizer)

        model.eval()
        val_loss = _batched_loss(model, val_seqs, y_val, mean, std, device,
                                 criterion, batch_size)
        if val_loss < best_val - 1e-4:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, seqs, mean, std, device, batch_size=BATCH_SIZE):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(seqs), batch_size):
            batch = _standardize(seqs[s:s + batch_size], mean, std)
            x, mask = _pad_batch(batch, device)
            preds.append(model(x, mask).cpu().numpy())
    return np.concatenate(preds, axis=0)


def predict_ensemble(models, seqs, mean, std, device):
    return np.mean([predict(m, seqs, mean, std, device) for m in models], axis=0)


def main():
    common.suppress_expected_warnings()
    set_seed()

    device = common.select_device()
    print(f"Using device: {device}")

    print(f"Loading {MODEL_NAME}...")
    feature_extractor, model, device = common.load_model(MODEL_NAME, device)

    def _load(split):
        return common.load_audio_window_data(
            os.path.join(common.LABELS_DIR, split), feature_extractor, model,
            device, MODEL_NAME, win_length_s=WIN_LENGTH_S, hop_length_s=HOP_LENGTH_S)

    print("Loading training data...")
    train_windows, y_train = _load("train_split.csv")
    print("Loading validation data...")
    val_windows, y_val = _load("dev_split.csv")
    print("Loading test data...")
    test_windows, y_test = _load("test_split.csv")

    if len(train_windows) == 0 or len(val_windows) == 0:
        print("Not enough data to train. Exiting.")
        return None

    fit_windows = train_windows + val_windows
    y_fit = np.concatenate([y_train, y_val])
    print(f"\n[{LABEL}] Total train+dev files: {len(fit_windows)}")

    mean, std = _feature_stats(fit_windows)

    print("\nTraining Transformer-encoder regressor ensemble...")
    models = []
    for i in range(ENSEMBLE_SIZE):
        print(f"  training Transformer regressor {i + 1}/{ENSEMBLE_SIZE}...")
        models.append(train_one(fit_windows, y_fit, device, mean, std, seed=SEED + i))

    if len(test_windows) > 0:
        preds = predict_ensemble(models, test_windows, mean, std, device)
        print(f"\n--- {LABEL}: Transformer Regressor (ensemble) — Test Set ---")
        common.print_point_metrics(y_test, preds)
        common.plot_predictions(
            y_test, preds, f"{LABEL} Regressor",
            common.media_path("transformer_pool_predictions.png"))
    else:
        print("No test split available; skipping evaluation.")

    return models


if __name__ == "__main__":
    main()
