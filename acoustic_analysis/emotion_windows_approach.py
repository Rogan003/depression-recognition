import os

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

import common

MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition" # EMOTIONS: angry, calm, disgust, fearful, happy, neutral, sad, surprised
LABEL = "Emotion Windows"

WIN_LENGTH_S = 5.0
HOP_LENGTH_S = 3.0


def load_ser_model(device):
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    emotions = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return feature_extractor, model, emotions


@torch.no_grad()
def extract_window_emotions(file_path, feature_extractor, model, device):
    sr = common.SAMPLE_RATE
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

    def window_probs(chunk):
        inputs = feature_extractor(chunk, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits.squeeze(0)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    probs = []
    for s in range(0, len(audio) - win_length + 1, hop_length):
        probs.append(window_probs(audio[s:s + win_length]))
    if len(probs) == 0:
        probs.append(window_probs(audio))
    return np.array(probs, dtype=np.float32)


def load_emotion_tracks(split_file, feature_extractor, model, device):
    if not os.path.exists(split_file):
        return [], np.array([])

    df = pd.read_csv(split_file)
    tracks, y = [], []

    safe_model_name = MODEL_NAME.replace("/", "_") + "_emotions"
    cache_dir = os.path.join(common.FEATURES_DIR, safe_model_name)
    os.makedirs(cache_dir, exist_ok=True)

    for _, row in df.iterrows():
        participant_id = int(row["Participant_ID"])
        score = row["PHQ_Score"]
        file_path = os.path.join(
            common.DATA_DIR, f"{participant_id}_P", f"{participant_id}_AUDIO.wav")
        cache_file = os.path.join(
            cache_dir, f"{participant_id}_{WIN_LENGTH_S}_{HOP_LENGTH_S}.npy")

        if os.path.exists(cache_file):
            print(f"Loading cached emotion track for {participant_id}...")
            tracks.append(np.load(cache_file))
            y.append(score)
        elif os.path.exists(file_path):
            print(f"Processing {file_path}...")
            track = extract_window_emotions(file_path, feature_extractor, model, device)
            if track is not None:
                np.save(cache_file, track)
                tracks.append(track)
                y.append(score)
            else:
                print(f"Warning: no emotion track extracted from {file_path}.")
        else:
            print(f"Warning: {file_path} not found.")

    return tracks, np.array(y)


def summarize_tracks(tracks, emotions):
    feats = []
    for track in tracks:
        dominant = track.argmax(axis=1)
        dominant_frac = np.array(
            [np.mean(dominant == e) for e in range(len(emotions))], dtype=np.float32)
        feats.append(np.concatenate([
            track.mean(axis=0),
            track.std(axis=0),
            track.max(axis=0),
            dominant_frac,
        ]))
    return np.array(feats, dtype=np.float32)


def feature_names(emotions):
    names = []
    for stat in ["mean", "std", "max", "dominant_frac"]:
        names += [f"{stat}_{emo}" for emo in emotions]
    return names


def main():
    common.suppress_expected_warnings()

    device = common.select_device()
    print(f"Using device: {device}")

    print(f"Loading emotion model '{MODEL_NAME}'...")
    feature_extractor, model, emotions = load_ser_model(device)
    print(f"Emotion classes: {emotions}")

    def _load(split):
        return load_emotion_tracks(
            os.path.join(common.LABELS_DIR, split), feature_extractor, model, device)

    print("Loading training data...")
    train_tracks, y_train = _load("train_split.csv")
    print("Loading validation data...")
    val_tracks, y_val = _load("dev_split.csv")
    print("Loading test data...")
    test_tracks, y_test = _load("test_split.csv")

    if len(train_tracks) == 0 or len(val_tracks) == 0:
        print("Not enough data to train. Exiting.")
        return None

    X_train = summarize_tracks(train_tracks + val_tracks, emotions)
    y_fit = np.concatenate([y_train, y_val])
    X_test = summarize_tracks(test_tracks, emotions) if len(test_tracks) > 0 else np.array([])

    print(f"\n[{LABEL}] Total train+dev files: {len(X_train)}, "
          f"feature dim: {X_train.shape[1]} ({feature_names(emotions)})")

    common.run_regression_pipeline(
        X_train, y_fit,
        X_test=X_test if len(X_test) > 0 else None,
        y_test=y_test if len(test_tracks) > 0 else None,
        scaler=StandardScaler(),
        summary_title=f"{LABEL}: Cross-Validation Summary on Combined Train+Dev",
        plot_out_path=common.media_path("emotion_windows_predictions.png"),
        cv_plot_out_path=common.media_path("emotion_windows_cv_predictions.png"),
        plot_label=LABEL,
        plot_color="darkorange")


if __name__ == "__main__":
    main()
