import pandas as pd
import numpy as np
import os
import contextlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings

# Ignore warning messages from huggingface/transformers
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Extraction hyper-parameters (kept in one place so cache keys stay consistent)
WIN_LENGTH_S = 15.0
HOP_LENGTH_S = 7.5
# How many windows are pushed through the transformer in a single forward pass.
# MPS (Apple GPU) has a tight memory budget, so we pick a smaller default there
# and shrink further on the fly if an out-of-memory error hits.
GPU_BATCH_WINDOWS = 8
MPS_BATCH_WINDOWS = 4

def _default_batch_windows(device):
    """Pick a safe starting batch size per device (MPS has little headroom)."""
    if device.type == "mps":
        return MPS_BATCH_WINDOWS
    if device.type == "cuda":
        return GPU_BATCH_WINDOWS
    return GPU_BATCH_WINDOWS

def _empty_cache(device):
    """Release cached allocator memory so it does not accumulate across windows."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()

def _is_oom_error(err):
    msg = str(err).lower()
    return "out of memory" in msg or "mps backend out of memory" in msg

def _autocast_ctx(device):
    """Enable mixed precision on GPU/MPS to speed up the transformer forward pass."""
    if device.type in ("cuda", "mps"):
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return contextlib.nullcontext()

def extract_wavlm_features(file_path, feature_extractor, model, device,
                           batch_windows=None):
    """
    Loads audio, normalizes it and slices it into overlapping windows, then runs
    WavLM over *batches* of windows at once (instead of one window at a time)
    to extract per-window embeddings.

    Returns a 2D array of shape (n_windows, 2 * hidden_size): for every window we
    keep the mean and std of the frame-level hidden states. The second, crude
    mean/std pooling across windows is intentionally NOT done here anymore -
    that step is replaced by a learnable attention-pooling network downstream.
    """
    sr = 16000
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    if len(audio) == 0:
        return None

    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    win_length = int(WIN_LENGTH_S * sr)
    hop_length = int(HOP_LENGTH_S * sr)

    # Slice the audio into equally sized windows.
    windows = [audio[s:s + win_length]
               for s in range(0, len(audio) - win_length + 1, hop_length)]
    if len(windows) == 0:
        # Audio shorter than a single window: use the whole clip as one window.
        windows = [audio]

    if batch_windows is None:
        batch_windows = _default_batch_windows(device)

    model.eval()
    embs = []
    # Process several windows per forward pass. This is the main speed-up:
    # the transformer is called ceil(n_windows / batch_windows) times instead
    # of once per window, which better saturates the CPU/GPU.
    #
    # If the device runs out of memory, we halve the batch size and retry that
    # same chunk (down to a single window). This keeps large windows working on
    # MPS/CUDA without a hard crash.
    with torch.inference_mode():
        i = 0
        while i < len(windows):
            chunk = windows[i:i + batch_windows]
            try:
                inputs = feature_extractor(
                    chunk, sampling_rate=sr, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with _autocast_ctx(device):
                    outputs = model(**inputs)

                # (batch, sequence_length, hidden_size)
                last_hidden = outputs.last_hidden_state.float()

                # Frame -> window pooling (mean + std over the time axis).
                mean_emb = last_hidden.mean(dim=1)
                std_emb = last_hidden.std(dim=1)
                window_emb = torch.cat([mean_emb, std_emb], dim=-1)
                embs.append(window_emb.cpu())

                # Free per-chunk tensors before moving on (important on MPS).
                del inputs, outputs, last_hidden, mean_emb, std_emb, window_emb
                _empty_cache(device)
                i += batch_windows
            except RuntimeError as e:
                if _is_oom_error(e) and batch_windows > 1:
                    _empty_cache(device)
                    new_batch = max(1, batch_windows // 2)
                    print(f"  OOM at batch_windows={batch_windows}; retrying with "
                          f"batch_windows={new_batch}...")
                    batch_windows = new_batch
                    continue
                raise

    embs = torch.cat(embs, dim=0)  # (n_windows, 2 * hidden_size)
    return embs.numpy().astype(np.float32)

def load_data(csv_path, feature_extractor, model, device, model_name, max_samples=None):
    df = pd.read_csv(csv_path)

    # X is a list of 2D arrays (n_windows, feat_dim); files differ in length.
    X = []
    y = []

    # Hyperparams used in extraction (from extract_wavlm_features)
    win_length_s = WIN_LENGTH_S
    hop_length_s = HOP_LENGTH_S

    # Create cache directory based on model name
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

        # "seq" marks the new per-window sequence cache (vs. the old pre-pooled vector).
        cache_file = os.path.join(cache_dir, f"{participant_id}_{win_length_s}_{hop_length_s}_seq.npy")

        if os.path.exists(cache_file):
            print(f"Loading cached features for {participant_id}...")
            features = np.load(cache_file)
            X.append(features)
            y.append(score)
            count += 1
        elif os.path.exists(file_path):
            print(f"Processing {file_path}...")
            features = extract_wavlm_features(file_path, feature_extractor, model, device)
            if features is not None:
                np.save(cache_file, features)
                X.append(features)
                y.append(score)
                count += 1
            else:
                print(f"Warning: no features extracted from {file_path}.")
        else:
            print(f"Warning: {file_path} not found.")

    return X, np.array(y, dtype=np.float32)


def aggregate_sequences(X):
    """Collapse each per-window sequence into a single fixed-size vector via
    mean + std pooling across windows. Used only to feed the classical
    (SVR / ElasticNet) baselines, which require fixed-length inputs."""
    out = []
    for seq in X:
        seq = np.asarray(seq, dtype=np.float32)
        if seq.ndim == 1:
            seq = seq[None, :]
        mean = seq.mean(axis=0)
        std = seq.std(axis=0)
        out.append(np.concatenate([mean, std], axis=-1))
    return np.asarray(out, dtype=np.float32)


class SequenceDataset(Dataset):
    """Yields a variable-length window sequence and its target score."""
    def __init__(self, X, y):
        self.X = [np.asarray(s, dtype=np.float32) for s in X]
        self.y = np.asarray(y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        seq = self.X[idx]
        if seq.ndim == 1:
            seq = seq[None, :]
        return torch.from_numpy(seq), torch.FloatTensor([self.y[idx]])


class AttentivePoolingRegressor(nn.Module):
    """Learnable replacement for the crude window-level mean/std pooling.

    Instead of averaging every window equally, the network projects each window
    embedding, learns an attention weight per window, and forms a weighted
    context vector. Depression-relevant cues are not spread uniformly across a
    long interview, so letting the model decide which windows matter tends to
    beat a plain mean.
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # x: (batch, n_windows, input_dim)
        h = self.proj(x)
        weights = torch.softmax(self.attn(h), dim=1)  # (batch, n_windows, 1)
        pooled = (weights * h).sum(dim=1)             # (batch, hidden_dim)
        return self.head(pooled)


def combined_loss(y_pred, y_true, alpha=0.7):
    """MAE + (1 - Pearson) loss, matching the CNN approach in this project."""
    mae = torch.mean(torch.abs(y_pred - y_true))
    if y_true.numel() < 2:
        return mae
    yt = y_true - torch.mean(y_true)
    yp = y_pred - torch.mean(y_pred)
    eps = 1e-8
    denom = torch.sqrt(torch.sum(yt ** 2) + eps) * torch.sqrt(torch.sum(yp ** 2) + eps)
    pearson = torch.sum(yt * yp) / denom
    return alpha * mae + (1 - alpha) * (1 - pearson)


def train_neural_regressor(X_train, y_train, X_val, y_val, device,
                           epochs=120, lr=1e-3, patience=15, batch_size=8):
    """Train the attention-pooling regressor with early stopping on val MAE."""
    torch.manual_seed(42)
    np.random.seed(42)

    input_dim = np.asarray(X_train[0]).shape[-1]
    model = AttentivePoolingRegressor(input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    train_ds = SequenceDataset(X_train, y_train)
    # batch_size=1 keeps things simple and robust to variable window counts.
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    best_val = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for seq, target in train_loader:
            seq = seq.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(seq)
            loss = combined_loss(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)

        val_pred = predict_neural(model, X_val, device)
        val_mae = mean_absolute_error(y_val, val_pred)
        scheduler.step(val_mae)

        if val_mae < best_val - 1e-4:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        print(f"Epoch {epoch+1}/{epochs} - train_loss: {avg_loss:.4f}, val_MAE: {val_mae:.4f}")
        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1} (best val MAE: {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_neural(model, X, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for seq in X:
            seq = np.asarray(seq, dtype=np.float32)
            if seq.ndim == 1:
                seq = seq[None, :]
            t = torch.from_numpy(seq).unsqueeze(0).to(device)
            preds.append(model(t).item())
    return np.array(preds, dtype=np.float32)

def main():
    # Use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "microsoft/wavlm-base-plus"
    print(f"Loading {model_name}...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    print("Loading training data...")
    X_train, y_train = load_data("../dataset/wwwedaic/labels/train_split.csv", feature_extractor, model, device, model_name)
    
    print("Loading validation data...")
    X_val, y_val = load_data("../dataset/wwwedaic/labels/dev_split.csv", feature_extractor, model, device, model_name)
    
    if len(X_train) == 0 or len(X_val) == 0:
        print("Not enough data to train. Exiting.")
        return

    # ------------------------------------------------------------------
    # Primary model: learnable attention-pooling neural regressor.
    # This replaces the old "second" crude mean/std pooling over windows.
    # ------------------------------------------------------------------
    print("\nStandardizing window features for the neural model...")
    seq_scaler = StandardScaler()
    seq_scaler.fit(np.vstack([np.asarray(s, dtype=np.float32) for s in X_train]))
    X_train_seq = [seq_scaler.transform(np.asarray(s, dtype=np.float32)) for s in X_train]
    X_val_seq = [seq_scaler.transform(np.asarray(s, dtype=np.float32)) for s in X_val]

    print(f"Training AttentivePoolingRegressor with {len(X_train)} samples...")
    nn_model = train_neural_regressor(X_train_seq, y_train, X_val_seq, y_val, device)
    y_pred_nn = predict_neural(nn_model, X_val_seq, device)
    mae_nn = mean_absolute_error(y_val, y_pred_nn)
    rmse_nn = np.sqrt(mean_squared_error(y_val, y_pred_nn))
    pearson_nn = pearsonr(y_val, y_pred_nn)[0] if len(np.unique(y_pred_nn)) > 1 else 0.0
    print(f"\nWavLM + AttentivePooling (neural) Evaluation Results:")
    print(f"MAE: {mae_nn:.4f}")
    print(f"RMSE: {rmse_nn:.4f}")
    print(f"Pearson correlation: {pearson_nn:.4f}")

    # ------------------------------------------------------------------
    # Classical baselines (for comparison). They need fixed-length inputs,
    # so window sequences are collapsed with mean/std pooling here.
    # ------------------------------------------------------------------
    print("\nAggregating sequences for classical baselines...")
    X_train_agg = aggregate_sequences(X_train)
    X_val_agg = aggregate_sequences(X_val)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_agg)
    X_val_scaled = scaler.transform(X_val_agg)

    print(f"Training SVR model with {len(X_train)} samples using RandomizedSearchCV...")
    param_distributions = {
        'C': [0.01, 0.1, 1, 10, 100],
        'epsilon': [0.001, 0.01, 0.1, 0.5],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }
    
    cv_folds = min(5, len(X_train)) if len(X_train) > 1 else 2
    cv_folds = max(2, cv_folds)
    if len(X_train) < 2:
        print("Not enough samples for CV, falling back to default SVR.")
        svr = SVR(C=1, epsilon=0.01, gamma='scale', kernel='rbf')
        svr.fit(X_train_scaled, y_train)
    else:
        base_svr = SVR()
        svr = RandomizedSearchCV(
            estimator=base_svr,
            param_distributions=param_distributions,
            n_iter=10,
            cv=cv_folds,
            scoring='neg_mean_absolute_error',
            random_state=42,
            n_jobs=-1
        )
        svr.fit(X_train_scaled, y_train)
        print(f"Best parameters found: {svr.best_params_}")
    
    print("Evaluating model...")
    y_pred = svr.predict(X_val_scaled)
    
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    if len(np.unique(y_pred)) > 1:
        pearson_corr, _ = pearsonr(y_val, y_pred)
    else:
        pearson_corr = 0.0
    
    print(f"\nWavLM + SVR Evaluation Results:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Pearson correlation: {pearson_corr:.4f}")

    print("\nTraining ElasticNet model using ElasticNetCV...")
    enet_alphas = [0.01, 0.1, 1, 10, 100]
    enet_l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    if len(X_train) < 2:
        print("Not enough samples for CV, falling back to default ElasticNet.")
        enet = ElasticNet(alpha=1.0, l1_ratio=0.5)
        enet.fit(X_train_scaled, y_train)
    else:
        enet = ElasticNetCV(
            alphas=enet_alphas,
            l1_ratio=enet_l1_ratios,
            cv=cv_folds,
            n_jobs=-1
        )
        enet.fit(X_train_scaled, y_train)
        print(f"Best parameters found for ElasticNet: alpha={enet.alpha_}, l1_ratio={enet.l1_ratio_}")
    
    y_pred_enet = enet.predict(X_val_scaled)
    
    mae_enet = mean_absolute_error(y_val, y_pred_enet)
    rmse_enet = np.sqrt(mean_squared_error(y_val, y_pred_enet))
    
    if len(np.unique(y_pred_enet)) > 1:
        pearson_corr_enet, _ = pearsonr(y_val, y_pred_enet)
    else:
        pearson_corr_enet = 0.0
        
    print(f"\nWavLM + ElasticNet Evaluation Results:")
    print(f"MAE: {mae_enet:.4f}")
    print(f"RMSE: {rmse_enet:.4f}")
    print(f"Pearson correlation: {pearson_corr_enet:.4f}")

if __name__ == "__main__":
    main()
