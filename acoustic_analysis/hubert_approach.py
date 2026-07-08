import pandas as pd
import numpy as np
import os
import torch
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

def extract_hubert_features(file_path, feature_extractor, model, device):
    """
    Loads audio, normalizes it, and processes it in windows through HuBERT
    to extract average hidden states.
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
    
    # 15s windows, 7.5s hop length
    win_length = int(15 * sr)
    hop_length = int(7.5 * sr)
    
    embs = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(audio) - win_length + 1, hop_length):
            window = audio[s:s+win_length]
            
            inputs = feature_extractor(window, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            # last_hidden_state is (batch, sequence_length, hidden_size)
            last_hidden = outputs.last_hidden_state.squeeze(0)
            
            # Aggregate over the sequence length for this window
            mean_emb = last_hidden.mean(dim=0)
            std_emb = last_hidden.std(dim=0)
            window_emb = torch.cat([mean_emb, std_emb], dim=-1)
            embs.append(window_emb.cpu())
            
    if len(embs) == 0:
        # Fallback if audio is shorter than window length
        inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state.squeeze(0)
            mean_emb = last_hidden.mean(dim=0)
            std_emb = last_hidden.std(dim=0)
            window_emb = torch.cat([mean_emb, std_emb], dim=-1)
            embs.append(window_emb.cpu())

    embs = torch.stack(embs)
    
    # Aggregate across all windows for the entire audio file (mean and std pooling)
    final_emb = torch.cat([embs.mean(dim=0), embs.std(dim=0)], dim=-1)
    
    return final_emb.numpy()

def load_data(csv_path, feature_extractor, model, device, model_name, max_samples=None):
    df = pd.read_csv(csv_path)

    X = []
    y = []

    # Hyperparams used in extraction (from extract_hubert_features)
    win_length_s = 15.0
    hop_length_s = 7.5
    
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
        
        # Cache file path includes participant ID and window/hop params
        cache_file = os.path.join(cache_dir, f"{participant_id}_{win_length_s}_{hop_length_s}.npy")

        if os.path.exists(cache_file):
            print(f"Loading cached features for {participant_id}...")
            features = np.load(cache_file)
            X.append(features)
            y.append(score)
            count += 1
        elif os.path.exists(file_path):
            print(f"Processing {file_path}...")
            features = extract_hubert_features(file_path, feature_extractor, model, device)
            if features is not None:
                np.save(cache_file, features)
                X.append(features)
                y.append(score)
                count += 1
            else:
                print(f"Warning: no features extracted from {file_path}.")
        else:
            print(f"Warning: {file_path} not found.")

    return np.array(X), np.array(y)

def main():
    # Use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "facebook/hubert-base-ls960"
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
        
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print(f"Training SVR model with {len(X_train)} samples using RandomizedSearchCV...")
    param_distributions = {
        'C': [0.01, 0.1, 1, 10, 100],
        'epsilon': [0.001, 0.01, 0.1, 0.5],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1]
    }
    
    cv_folds = min(5, len(X_train)) if len(X_train) > 1 else 2
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
    
    print(f"\nHuBERT + SVR Evaluation Results:")
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
        
    print(f"\nHuBERT + ElasticNet Evaluation Results:")
    print(f"MAE: {mae_enet:.4f}")
    print(f"RMSE: {rmse_enet:.4f}")
    print(f"Pearson correlation: {pearson_corr_enet:.4f}")

if __name__ == "__main__":
    main()
