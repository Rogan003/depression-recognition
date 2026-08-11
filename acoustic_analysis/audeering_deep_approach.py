import os

import numpy as np
import torch
import torch.nn as nn

import common

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
LABEL = "Audeering Deep"

WIN_LENGTH_S = 5.0
HOP_LENGTH_S = 3.0

SEED = 42

AE_CONFIG = { # best config found through an extensive search
    "n_features": 200,
    "hidden": (512,),
    "dropout": 0.1,
    "activation": "relu",
    "batchnorm": False,
    "epochs": 120,
    "batch_size": 64,
    "lr": 3e-3,
    "weight_decay": 1e-4,
    "tie_weights": False,
}


AE_INPUT_NOISE_STD = 0.1


ATTENTION_ENSEMBLE_SIZE = 5


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


class WindowAutoencoder(nn.Module):
    def __init__(self, in_dim, n_features=200, hidden=(512,),
                 dropout=0.1, activation="relu", batchnorm=False):
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation, nn.ReLU)

        def block(in_features, out_features):
            layers = [nn.Linear(in_features, out_features)]
            if batchnorm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(act_cls())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            return layers

        enc_layers = []
        prev = in_dim
        for h in hidden:
            enc_layers += block(prev, h)
            prev = h
        enc_layers.append(nn.Linear(prev, n_features))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = [act_cls()]
        prev = n_features
        for h in reversed(hidden):
            dec_layers += block(prev, h)
            prev = h
        dec_layers.append(nn.Linear(prev, in_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(train_windows, device, verbose=True):
    stacked = np.concatenate(train_windows, axis=0).astype(np.float32)

    mean = stacked.mean(axis=0, keepdims=True)
    std = stacked.std(axis=0, keepdims=True) + 1e-8
    stacked_norm = (stacked - mean) / std

    x = torch.from_numpy(stacked_norm)
    dataset = torch.utils.data.TensorDataset(x)
    batch_size = min(AE_CONFIG["batch_size"], max(1, len(dataset)))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=len(dataset) > batch_size)

    model = WindowAutoencoder(
        stacked.shape[1], n_features=AE_CONFIG["n_features"], hidden=AE_CONFIG["hidden"],
        dropout=AE_CONFIG["dropout"], activation=AE_CONFIG["activation"],
        batchnorm=AE_CONFIG["batchnorm"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=AE_CONFIG["lr"], weight_decay=AE_CONFIG["weight_decay"])
    criterion = nn.MSELoss()

    epochs = AE_CONFIG["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            noisy = batch + AE_INPUT_NOISE_STD * torch.randn_like(batch) \
                if AE_INPUT_NOISE_STD > 0 else batch
            optimizer.zero_grad()
            recon, _ = model(noisy)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.size(0)
        scheduler.step()
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"  [AE] epoch {epoch + 1:>3}/{epochs}  recon MSE = "
                  f"{total / len(dataset):.4f}")

    return model, (mean, std)


def encode_windows(model, norm_stats, windows_per_file, device, batch_size=256):
    mean, std = norm_stats
    model.eval()
    encoded = []
    with torch.no_grad():
        for win in windows_per_file:
            win_norm = ((win - mean) / std).astype(np.float32)
            z_chunks = []
            for s in range(0, len(win_norm), batch_size):
                batch = torch.from_numpy(win_norm[s:s + batch_size]).to(device)
                z_chunks.append(model.encode(batch).cpu().numpy())
            encoded.append(np.concatenate(z_chunks, axis=0))
    return encoded


def aggregate_files(encoded_windows):
    feats = []
    for enc in encoded_windows:
        mean_emb = enc.mean(axis=0)
        std_emb = enc.std(axis=0)
        feats.append(np.concatenate([mean_emb, std_emb], axis=-1))
    return np.array(feats, dtype=np.float32)


class AttentionPoolRegressor(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, mask):
        h = self.proj(x)
        scores = self.attn(h).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
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


def train_attention_regressor(train_seqs, y_train, device, feat_mean, feat_std,
                              epochs=200, lr=1e-3, val_fraction=0.2, patience=25,
                              seed=SEED):
    set_seed(seed)
    n = len(train_seqs)
    idx = np.random.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def norm(seq):
        return ((seq - feat_mean) / feat_std).astype(np.float32)

    tr_seqs = [norm(train_seqs[i]) for i in tr_idx]
    val_seqs = [norm(train_seqs[i]) for i in val_idx]
    y_tr = torch.tensor(y_train[tr_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(y_train[val_idx], dtype=torch.float32, device=device)

    model = AttentionPoolRegressor(train_seqs[0].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()

    x_tr, mask_tr = _pad_batch(tr_seqs, device)
    x_val, mask_val = _pad_batch(val_seqs, device)

    best_val = float("inf")
    best_state = None
    bad = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_tr, mask_tr)
        loss = criterion(pred, y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val, mask_val)
            val_loss = criterion(val_pred, y_val).item()

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_attention_ensemble(train_seqs, y_train, device, feat_mean, feat_std,
                             n_models=ATTENTION_ENSEMBLE_SIZE):
    models = []
    for i in range(n_models):
        print(f"  training attention regressor {i + 1}/{n_models}...")
        models.append(train_attention_regressor(
            train_seqs, y_train, device, feat_mean, feat_std, seed=SEED + i))
    return models


def predict_attention(model, seqs, device, feat_mean, feat_std, batch_size=16):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(seqs), batch_size):
            chunk = [((seq - feat_mean) / feat_std).astype(np.float32)
                     for seq in seqs[s:s + batch_size]]
            x, mask = _pad_batch(chunk, device)
            preds.append(model(x, mask).cpu().numpy())
    return np.concatenate(preds, axis=0)


def predict_attention_ensemble(models, seqs, device, feat_mean, feat_std):
    preds = [predict_attention(m, seqs, device, feat_mean, feat_std)
             for m in models]
    return np.mean(preds, axis=0)


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

    print(f"\nTraining deep denoising autoencoder to keep "
          f"{AE_CONFIG['n_features']} features per window with config: {AE_CONFIG}")
    set_seed()
    ae_model, norm_stats = train_autoencoder(fit_windows, device)

    enc_fit = encode_windows(ae_model, norm_stats, fit_windows, device)
    enc_test = encode_windows(ae_model, norm_stats, test_windows, device) \
        if len(test_windows) > 0 else []

    print("\n" + "=" * 70)
    print(f"APPROACH 1 — model zoo on {AE_CONFIG['n_features']}-dim deep window features")
    print("=" * 70)
    X_fit = aggregate_files(enc_fit)
    X_test = aggregate_files(enc_test) if len(enc_test) > 0 else np.array([])

    common.run_regression_pipeline(
        X_fit, y_fit,
        X_test=X_test if len(X_test) > 0 else None,
        y_test=y_test if len(enc_test) > 0 else None,
        summary_title=f"{LABEL} (zoo): Cross-Validation Summary on Combined Train+Dev",
        plot_out_path=common.media_path("audeering_deep_zoo_predictions.png"),
        cv_plot_out_path=common.media_path("audeering_deep_zoo_cv_predictions.png"),
        plot_label=f"{LABEL} (zoo)")

    print("\n" + "=" * 70)
    print("APPROACH 2 — deep attention-pooling regressor (ensemble)")
    print("=" * 70)

    stacked = np.concatenate(enc_fit, axis=0)
    feat_mean = stacked.mean(axis=0, keepdims=True)
    feat_std = stacked.std(axis=0, keepdims=True) + 1e-8

    print("Training attention-pooling regressor ensemble...")
    attn_models = train_attention_ensemble(
        enc_fit, y_fit, device, feat_mean, feat_std)

    if len(enc_test) > 0:
        preds = predict_attention_ensemble(
            attn_models, enc_test, device, feat_mean, feat_std)
        print(f"\n--- {LABEL}: Attention Regressor (ensemble) — Test Set ---")
        common.print_point_metrics(y_test, preds)
        common.plot_predictions(
            y_test, preds, f"{LABEL} Attention Regressor",
            common.media_path("audeering_deep_attention_predictions.png"))
    else:
        print("No test split available; skipping attention-regressor evaluation.")

    return attn_models


if __name__ == "__main__":
    main()
