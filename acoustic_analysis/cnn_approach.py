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
    def __init__(self, csv_path, window_size=10, hop_length=5):
        self.df = pd.read_csv(csv_path)
        self.data = []

        print(f"Loading data from {csv_path}...")
        for index, row in self.df.iterrows():
            participant_id = int(row['Participant_ID'])
            score = float(row['PHQ_Score'])
            file_path = f"../dataset/wwwedaic/data/{participant_id}_P/{participant_id}_AUDIO.wav"

            if os.path.exists(file_path):
                print(f"Processing {file_path}...")
                windows = get_mfcc_windows(file_path, 40, window_size_s=window_size, hop_length_s=hop_length)
                if len(windows) > 0:
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
        return torch.from_numpy(windows), torch.FloatTensor([y])


class CNNRegressor(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.GroupNorm(8, 48),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(96, 1)

    def forward(self, x):
        b, n, c, h, w = x.size()
        x = x.view(b * n, c, h, w)
        feat = self.cnn(x)
        feat = self.pool(feat).flatten(1)
        feat = feat.view(b, n, -1)
        mean = feat.mean(dim=1)
        std = feat.std(dim=1)
        pooled = torch.cat([mean, std], dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)


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

    device = torch.device("mps")

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

        model = CNNRegressor().to(device)
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

    print(f"\nCNN CV Results (Average):")
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
    main(30, 30)
