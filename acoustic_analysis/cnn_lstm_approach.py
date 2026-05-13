import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import pandas as pd
import numpy as np
import os
from common import get_mfcc_windows
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

class AudioDataset(Dataset):
    """One sample per audio: a sequence of MFCC windows + one PHQ score."""
    def __init__(self, csv_path, window_size=10, hop_length=5):
        self.df = pd.read_csv(csv_path)
        self.data = []

        print(f"Loading data from {csv_path}...")
        for index, row in self.df.iterrows():
            participant_id = int(row['Participant_ID'])
            score = float(row['PHQ_Score'])
            file_path = f"dataset/wwwedaic/data/{participant_id}_P/{participant_id}_AUDIO.wav"

            if os.path.exists(file_path):
                print(f"Processing {file_path}...")
                windows = get_mfcc_windows(file_path, window_size_s=window_size, hop_length_s=hop_length)
                if len(windows) > 0:
                    # shape: (num_windows, n_mfcc, time_frames)
                    windows_arr = np.asarray(windows, dtype=np.float32)
                    self.data.append((windows_arr, score))
                else:
                    print(f"Warning: no windows extracted from {file_path}.")
            else:
                print(f"Warning: {file_path} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        windows, y = self.data[idx]
        # add channel dim -> (num_windows, 1, n_mfcc, time_frames)
        x = torch.from_numpy(windows).unsqueeze(1)
        return x, torch.FloatTensor([y])

class CNNLSTM(nn.Module):
    """
    Hierarchical encoder:
      - CNN encodes each MFCC window into a fixed-size vector.
      - LSTM consumes the sequence of window vectors for one audio.
      - FC head produces a single PHQ score per audio.
    """
    def __init__(self, cnn_feat_dim=32, lstm_hidden=64):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, cnn_feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_feat_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
        )
        # collapse remaining spatial dims so each window becomes a vector
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.lstm = nn.LSTM(
            input_size=cnn_feat_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            num_layers=2,
            dropout=0.2,
        )
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # x: (batch, num_windows, 1, n_mfcc, time)
        b, n, c, h, w = x.size()
        # treat all windows in the batch as a flat batch for the CNN
        x = x.view(b * n, c, h, w)
        feat = self.cnn(x)               # (b*n, C, h', w')
        feat = self.pool(feat)           # (b*n, C, 1, 1)
        feat = feat.view(b, n, -1)       # (b, n, C)

        lstm_out, _ = self.lstm(feat)    # (b, n, hidden)
        return self.fc(lstm_out[:, -1, :])  # (b, 1) -- one score per audio

def combined_loss(y_pred, y_true, alpha=0.5):
    mae = torch.mean(torch.abs(y_pred - y_true))
    mse = torch.mean((y_pred - y_true) ** 2)

    if y_true.numel() < 2:
        return alpha * mae + (1 - alpha) * mse

    y_true_centered = y_true - torch.mean(y_true)
    y_pred_centered = y_pred - torch.mean(y_pred)

    numerator = torch.sum(y_true_centered * y_pred_centered)
    # add eps *inside* each sqrt so gradients stay finite when variance is ~0
    eps = 1e-8
    denominator = torch.sqrt(torch.sum(y_true_centered ** 2) + eps) * \
                  torch.sqrt(torch.sum(y_pred_centered ** 2) + eps)

    pearson = numerator / denominator
    return alpha * mae + (1 - alpha) * (1 - pearson)

def main(window_size=10, hop_length=5):
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cpu") # using Mac GPU got me worse results

    train_dataset = AudioDataset("../dataset/wwwedaic/labels/train_split.csv", window_size, hop_length)
    val_dataset = AudioDataset("../dataset/wwwedaic/labels/dev_split.csv", window_size, hop_length)
    test_dataset = AudioDataset("../dataset/wwwedaic/labels/test_split.csv", window_size, hop_length)

    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_maes = []
    cv_rmses = []
    cv_pearsons = []

    print(f"Starting 5-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(combined_dataset)))):
        print(f"\n--- Fold {fold + 1} ---")
        
        train_sub = Subset(combined_dataset, train_idx)
        val_sub = Subset(combined_dataset, val_idx)

        # batch_size=1 because each audio has a different number of windows.
        # This keeps the implementation simple and avoids padding/masking.
        train_loader = DataLoader(train_sub, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=1, shuffle=False)

        model = CNNLSTM().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        epochs = 20
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = combined_loss(outputs, batch_y, alpha=0.7)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item() * batch_x.size(0)
            avg_loss = total_loss / len(train_loader.dataset)
            scheduler.step(avg_loss)
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        print("Evaluating fold...")
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                preds = outputs.cpu().numpy().flatten()
                targets = batch_y.numpy().flatten()
                all_preds.extend(preds)
                all_targets.extend(targets)

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        if len(np.unique(all_preds)) > 1:
            pearson_corr, _ = pearsonr(all_targets, all_preds)
        else:
            pearson_corr = 0.0
            
        cv_maes.append(mae)
        cv_rmses.append(rmse)
        cv_pearsons.append(pearson_corr)
        
        print(f"Fold {fold+1} Results - MAE: {mae:.4f}, RMSE: {rmse:.4f}, Pearson: {pearson_corr:.4f}")

    print(f"\nCNN+LSTM CV Results (Average):")
    print(f"MAE: {np.mean(cv_maes):.4f} (+/- {np.std(cv_maes):.4f})")
    print(f"RMSE: {np.mean(cv_rmses):.4f} (+/- {np.std(cv_rmses):.4f})")
    print(f"Pearson correlation: {np.mean(cv_pearsons):.4f} (+/- {np.std(cv_pearsons):.4f})")

    # print("\nFinal Test Evaluation...")
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # model.eval()
    # test_preds = []
    # test_targets = []
    # with torch.no_grad():
    #     for batch_x, batch_y in test_loader:
    #         batch_x = batch_x.to(device)
    #         outputs = model(batch_x)
    #         test_preds.extend(outputs.cpu().numpy().flatten())
    #         test_targets.extend(batch_y.numpy().flatten())
    # test_mae = mean_absolute_error(test_targets, test_preds)
    # test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    # test_pearson, _ = pearsonr(test_targets, test_preds)
    # print(f"Test Results - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, Pearson: {test_pearson:.4f}")

if __name__ == "__main__":
    main(42, 18)
