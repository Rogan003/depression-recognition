import os
import warnings

import librosa
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, loguniform, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, RidgeCV, ElasticNetCV
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    RandomizedSearchCV, cross_val_predict, cross_validate,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from transformers import AutoFeatureExtractor, AutoModel

DATA_DIR = "../dataset/wwwedaic/data"
LABELS_DIR = "../dataset/wwwedaic/labels"
FEATURES_DIR = "../features"
MEDIA_DIR = "../media"
SAMPLE_RATE = 16000


def media_path(filename):
    os.makedirs(MEDIA_DIR, exist_ok=True)
    return os.path.join(MEDIA_DIR, filename)


def get_mfcc_windows(file_path, n_mfcc=40, window_size_s=10, hop_length_s=5):
    sr = 16000
    audio = preprocess(file_path, sr)
    hop_length = 512

    features = extract_mfcc(
        audio,
        sr,
        n_mfcc=n_mfcc,
        hop_length=hop_length
    )

    frames_per_sec = sr / hop_length
    window_frames = int(window_size_s * frames_per_sec)
    hop_frames = int(hop_length_s * frames_per_sec)
    
    windows = []

    total_frames = features.shape[2]

    for start in range(
        0,
        total_frames - window_frames + 1,
        hop_frames
    ):
        window = features[:, :, start:start + window_frames]

        windows.append(window)

    return np.array(windows, dtype=np.float32) # (n_windows, 3, n_mfcc, window_frames)


def preprocess(file_path, sr):
    audio, sr = librosa.load(file_path, sr=sr, mono=True)

    print("Original length:", len(audio)/sr, "seconds")

    # Remove interviewer - WORSE METRICS WITH THIS!
    # audio_without_interviewer = remove_interviewer_from_audio(audio, file_path[8:11], sr)

    audio_normalized = audio / np.max(np.abs(audio))

    return audio_normalized


# def remove_interviewer_from_audio(audio, file_id, sr):
#     transcript = pd.read_csv(f"dataset/{file_id}_TRANSCRIPT.csv", sep="\t")
#
#     participant_segments = transcript[transcript["speaker"] == "Participant"]
#
#     audio_segments = []
#     for _, row in participant_segments.iterrows():
#         start_sample = int(row["start_time"] * sr)
#         stop_sample = int(row["stop_time"] * sr)
#         audio_segments.append(audio[start_sample:stop_sample])
#
#     if len(audio_segments) > 0:
#         audio = np.concatenate(audio_segments)
#     else:
#         audio = np.array([])
#
#     return audio


def extract_mfcc(audio, sr, n_mfcc=13, hop_length=512):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.stack([mfcc, delta, delta2], axis=0)
    return features


def suppress_expected_warnings():
    # TODO: Is this even necessary? Or should I add more things here?
    warnings.filterwarnings("ignore")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    try:
        from scipy.stats import ConstantInputWarning, NearConstantInputWarning
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        warnings.filterwarnings("ignore", category=NearConstantInputWarning)
    except ImportError:
        pass


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name, device=None):
    device = select_device() if device is None else device
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return feature_extractor, model, device


def extract_transformer_features(file_path, feature_extractor, model, device,
                                 win_length_s=5.0, hop_length_s=3.0):
    sr = SAMPLE_RATE
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    if len(audio) == 0:
        return None

    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    win_length = int(win_length_s * sr)
    hop_length = int(hop_length_s * sr)

    def window_embedding(chunk):
        inputs = feature_extractor(chunk, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state.squeeze(0)
        mean_emb = last_hidden.mean(dim=0)
        std_emb = last_hidden.std(dim=0)
        return torch.cat([mean_emb, std_emb], dim=-1).cpu()

    embs = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(audio) - win_length + 1, hop_length):
            embs.append(window_embedding(audio[s:s + win_length]))

        if len(embs) == 0:
            # Fallback if audio is shorter than the window length
            embs.append(window_embedding(audio))

    embs = torch.stack(embs)

    # Aggregate across all windows for the entire audio file (mean and std)
    final_emb = torch.cat([embs.mean(dim=0), embs.std(dim=0)], dim=-1)
    return final_emb.numpy()


def load_audio_data(split_file, feature_extractor, model, device, model_name,
                    win_length_s=5.0, hop_length_s=3.0, max_samples=None, verbose=True):
    if not os.path.exists(split_file):
        return np.array([]), np.array([])

    df = pd.read_csv(split_file)
    X, y = [], []

    safe_model_name = model_name.replace("/", "_")
    cache_dir = os.path.join(FEATURES_DIR, safe_model_name)
    os.makedirs(cache_dir, exist_ok=True)

    count = 0
    for _, row in df.iterrows():
        if max_samples and count >= max_samples:
            break

        participant_id = int(row["Participant_ID"])
        score = row["PHQ_Score"]
        file_path = os.path.join(DATA_DIR, f"{participant_id}_P", f"{participant_id}_AUDIO.wav")
        cache_file = os.path.join(cache_dir, f"{participant_id}_{win_length_s}_{hop_length_s}.npy")

        if os.path.exists(cache_file):
            if verbose:
                print(f"Loading cached features for {participant_id}...")
            X.append(np.load(cache_file))
            y.append(score)
            count += 1
        elif os.path.exists(file_path):
            if verbose:
                print(f"Processing {file_path}...")
            features = extract_transformer_features(
                file_path, feature_extractor, model, device,
                win_length_s=win_length_s, hop_length_s=hop_length_s)
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


def extract_transformer_window_features(file_path, feature_extractor, model, device,
                                        win_length_s=5.0, hop_length_s=3.0):
    sr = SAMPLE_RATE
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    if len(audio) == 0:
        return None

    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    win_length = int(win_length_s * sr)
    hop_length = int(hop_length_s * sr)

    def window_descriptor(chunk):
        inputs = feature_extractor(chunk, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state.squeeze(0)  # (T, H)
        mean_emb = last_hidden.mean(dim=0)
        std_emb = last_hidden.std(dim=0)
        max_emb = last_hidden.max(dim=0).values
        min_emb = last_hidden.min(dim=0).values
        return torch.cat([mean_emb, std_emb, max_emb, min_emb], dim=-1).cpu()

    embs = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(audio) - win_length + 1, hop_length):
            embs.append(window_descriptor(audio[s:s + win_length]))
        if len(embs) == 0:
            embs.append(window_descriptor(audio))

    return torch.stack(embs).numpy().astype(np.float32)


def load_audio_window_data(split_file, feature_extractor, model, device, model_name,
                           win_length_s=5.0, hop_length_s=3.0, max_samples=None,
                           verbose=True):
    if not os.path.exists(split_file):
        return [], np.array([])

    df = pd.read_csv(split_file)
    windows_per_file, y = [], []

    safe_model_name = model_name.replace("/", "_") + "_rich"
    cache_dir = os.path.join(FEATURES_DIR, safe_model_name)
    os.makedirs(cache_dir, exist_ok=True)

    count = 0
    for _, row in df.iterrows():
        if max_samples and count >= max_samples:
            break

        participant_id = int(row["Participant_ID"])
        score = row["PHQ_Score"]
        file_path = os.path.join(DATA_DIR, f"{participant_id}_P", f"{participant_id}_AUDIO.wav")
        cache_file = os.path.join(cache_dir, f"{participant_id}_{win_length_s}_{hop_length_s}.npy")

        if os.path.exists(cache_file):
            if verbose:
                print(f"Loading cached window features for {participant_id}...")
            windows_per_file.append(np.load(cache_file))
            y.append(score)
            count += 1
        elif os.path.exists(file_path):
            if verbose:
                print(f"Processing {file_path}...")
            features = extract_transformer_window_features(
                file_path, feature_extractor, model, device,
                win_length_s=win_length_s, hop_length_s=hop_length_s)
            if features is not None:
                np.save(cache_file, features)
                windows_per_file.append(features)
                y.append(score)
                count += 1
            else:
                print(f"Warning: no features extracted from {file_path}.")
        else:
            print(f"Warning: {file_path} not found.")

    return windows_per_file, np.array(y)


def pearson_corr(y_true, y_pred):
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    corr = pearsonr(y_true, y_pred)[0]
    return 0.0 if np.isnan(corr) else corr


def pearson_scorer(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.std(y_pred) < 1e-8 or np.std(y_true) < 1e-8:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr, _ = pearsonr(y_true, y_pred)
    return 0.0 if np.isnan(corr) else corr


def make_scoring():
    return {
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "Pearson": make_scorer(pearson_scorer),
    }


def model_score_for_picking(model_result):
    return model_result["MAE"] + model_result["RMSE"] * (3 / 5) - 10 * model_result["Pearson"]


def default_models(bayesian_needs_dense=False):
    bayesian_cfg = {
        "name": "BayesianRidge",
        "estimator": BayesianRidge(),
        "param_dist": {
            "alpha_1": loguniform(1e-4, 1e-1),
            "alpha_2": loguniform(1e-4, 1e-1),
            "lambda_1": loguniform(1e-4, 1e-1),
            "lambda_2": loguniform(1e-4, 1e-1),
        },
        "n_iter": 20,
        "n_jobs": 1,
    }
    if bayesian_needs_dense:
        bayesian_cfg["needs_dense"] = True

    return [
        {
            "name": "Ridge",
            "estimator": RidgeCV(),
            "n_iter": 1000,
        },
        {
            "name": "ElasticNet",
            "estimator": ElasticNetCV(max_iter=100, random_state=42),
            "param_dist": {
                "l1_ratio": uniform(0, 1),
            },
            "n_iter": 100,
        },
        {
            "name": "SVR",
            "estimator": SVR(),
            "param_dist": {
                "C": loguniform(1e-4, 1e4),
                "epsilon": uniform(0.001, 0.8),
                "gamma": ["scale", "auto", 0.1, 0.01],
                "kernel": ["rbf", "linear", "poly"],
            },
            "n_iter": 100,
        },
        {
            "name": "RandomForest",
            "estimator": RandomForestRegressor(random_state=42),
            "param_dist": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
            "n_iter": 20,
        },
        {
            "name": "GradientBoosting",
            "estimator": GradientBoostingRegressor(random_state=42),
            "param_dist": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
            },
            "n_iter": 20,
        },
        bayesian_cfg,
        {
            "name": "MLPRegressor",
            "estimator": MLPRegressor(max_iter=200, random_state=42, early_stopping=True),
            "param_dist": {
                "hidden_layer_sizes": [(50,), (100,)],
                "activation": ["relu", "tanh"],
                "alpha": loguniform(1e-5, 1e-2),
                "learning_rate_init": loguniform(1e-5, 1e-2),
            },
            "n_iter": 50,
        },
    ]


def run_random_search(model_cfg, X, y, scoring, scaler=None):
    if model_cfg.get("needs_dense", False) and hasattr(X, "toarray"):
        X = X.toarray()

    estimator = clone(model_cfg["estimator"])
    param_dist = model_cfg.get("param_dist")
    n_jobs = model_cfg.get("n_jobs", -1)

    if not param_dist:
        if scaler is not None:
            estimator = Pipeline([("scaler", clone(scaler)), ("model", estimator)])

        cv_results = cross_validate(
            estimator, X, y, cv=5, scoring=scoring, n_jobs=n_jobs)
        cv_mae = -np.mean(cv_results["test_MAE"])
        cv_rmse = -np.mean(cv_results["test_RMSE"])
        cv_pearson = np.mean(cv_results["test_Pearson"])

        estimator.fit(X, y)

        print(f"CV MAE: {cv_mae:.4f}, RMSE: {cv_rmse:.4f}, Pearson: {cv_pearson:.4f}")

        return {
            "name": model_cfg["name"],
            "MAE": cv_mae,
            "RMSE": cv_rmse,
            "Pearson": cv_pearson,
            "model": estimator,
            "params": {},
            "needs_dense": model_cfg.get("needs_dense", False),
        }

    if scaler is not None:
        estimator = Pipeline([("scaler", clone(scaler)), ("model", estimator)])
        param_dist = {f"model__{key}": value for key, value in param_dist.items()}

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=model_cfg["n_iter"],
        cv=5,
        scoring=scoring,
        refit="MAE",
        n_jobs=n_jobs,
        random_state=42,
    )
    search.fit(X, y)

    best_idx = search.best_index_
    cv_results = search.cv_results_
    cv_mae = -cv_results["mean_test_MAE"][best_idx]
    cv_rmse = -cv_results["mean_test_RMSE"][best_idx]
    cv_pearson = cv_results["mean_test_Pearson"][best_idx]

    best_params = {key.replace("model__", "", 1): value
                   for key, value in search.best_params_.items()}

    print(f"Best {model_cfg['name']} params: {best_params}")
    print(f"CV MAE: {cv_mae:.4f}, RMSE: {cv_rmse:.4f}, Pearson: {cv_pearson:.4f}")

    return {
        "name": model_cfg["name"],
        "MAE": cv_mae,
        "RMSE": cv_rmse,
        "Pearson": cv_pearson,
        "model": search.best_estimator_,
        "params": best_params,
        "needs_dense": model_cfg.get("needs_dense", False),
    }


def cross_validate_models(models, X, y, scoring=None, scaler=None):
    scoring = make_scoring() if scoring is None else scoring
    results = []
    for model_cfg in models:
        print(f"\nRunning {model_cfg['name']}...")
        results.append(run_random_search(model_cfg, X, y, scoring, scaler=scaler))
    return results


def print_cv_summary(results, title, name_width=16):
    print(f"\n--- {title} ---")
    print(f"{'Model':<{name_width}} | {'CV MAE':<8} | {'CV RMSE':<8} | {'CV Pearson':<10}")
    print("-" * (name_width + 36))
    for result in results:
        print(
            f"{result['name']:<{name_width}} | {result['MAE']:<8.4f} | "
            f"{result['RMSE']:<8.4f} | {result['Pearson']:<10.4f}"
        )


def print_baseline(y, mean_value=None):
    y = np.asarray(y, dtype=float)
    reference = np.mean(y) if mean_value is None else mean_value
    baseline_mae = mean_absolute_error(y, [reference] * len(y))
    print(f"\nBaseline (predict-the-mean) MAE: {baseline_mae:.4f}")
    return baseline_mae


def out_of_fold_predictions(model, X, y, cv=5):
    return cross_val_predict(clone(model), X, y, cv=cv, n_jobs=-1)


def print_point_metrics(y_true, y_pred):
    print(f"MAE:     {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE:    {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"Pearson: {pearson_corr(y_true, y_pred):.4f}")


def plot_predictions(y_true, y_pred, model_name, out_path, color="steelblue", ylabel="PHQ score"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    order = np.argsort(y_true)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    x = np.arange(len(y_true_sorted))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson = pearson_scorer(y_true, y_pred)

    plt.figure(figsize=(14, 7))
    plt.plot(x, y_true_sorted, color="black", linewidth=2, label="Actual PHQ score")
    plt.vlines(x, y_true_sorted, y_pred_sorted, color="lightgray", linewidth=1, zorder=1)
    plt.scatter(x, y_pred_sorted, alpha=0.8, color=color, edgecolors="k",
                zorder=2, label="Predicted PHQ score")
    plt.xlabel("Samples (sorted by actual PHQ score)")
    plt.ylabel(ylabel)
    plt.title(
        f"Predictions vs Actual ({model_name})\n"
        f"MAE={mae:.3f}, RMSE={rmse:.3f}, Pearson={pearson:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved prediction visualization to '{out_path}'")


def evaluate_on_test(results, X_test, y_test,
                     title="Test Set Evaluation (all models)", name_width=16,
                     plot_out_path=None, best_name=None,
                     plot_color="steelblue", plot_ylabel="PHQ score", label_fn=None):
    y_test = np.asarray(y_test, dtype=float)
    print(f"\n--- {title} ---")
    print(f"{'Model':<{name_width}} | {'MAE':<8} | {'RMSE':<8} | {'Pearson':<10}")
    print("-" * (name_width + 36))

    metrics = []
    for result in results:
        model = result.get("model")
        if model is None:
            continue
        X_eval = X_test
        if result.get("needs_dense", False) and hasattr(X_test, "toarray"):
            X_eval = X_test.toarray()
        preds = model.predict(X_eval)
        test_mae = mean_absolute_error(y_test, preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, preds))
        test_pearson = pearson_corr(y_test, preds)
        print(
            f"{result['name']:<{name_width}} | {test_mae:<8.4f} | "
            f"{test_rmse:<8.4f} | {test_pearson:<10.4f}"
        )
        metrics.append({
            "name": result["name"],
            "MAE": test_mae,
            "RMSE": test_rmse,
            "Pearson": test_pearson,
            "preds": preds,
        })

    if plot_out_path and metrics:
        best_metric = None
        if best_name is not None:
            best_metric = next((m for m in metrics if m["name"] == best_name), None)
        if best_metric is None:
            best_metric = min(metrics, key=model_score_for_picking)
        print(f"\nGenerating test-set prediction visualization for best model "
              f"({best_metric['name']})...")
        label = label_fn(best_metric["name"]) if label_fn else best_metric["name"]
        plot_predictions(y_test, best_metric["preds"], label, plot_out_path,
                         color=plot_color, ylabel=plot_ylabel)

    return metrics


def run_regression_pipeline(X, y, *, X_test=None, y_test=None, models=None,
                            scoring=None, scaler=StandardScaler(),
                            summary_title="Cross-Validation Summary on Combined Train+Dev",
                            name_width=16, plot_out_path=None, cv_plot_out_path=None,
                            plot_label=None, plot_color="steelblue", plot_ylabel="PHQ score"):
    models = default_models() if models is None else models
    scoring = make_scoring() if scoring is None else scoring

    print("\nTraining and tuning models with cross-validation...")
    results = cross_validate_models(models, X, y, scoring, scaler=scaler)

    print_cv_summary(results, summary_title, name_width)
    print_baseline(y)

    best_result = min(results, key=model_score_for_picking)
    print(f"\n>>> Best model picked: {best_result['name']} <<<")

    if cv_plot_out_path:
        print(f"\nGenerating out-of-fold (CV) prediction visualization for best "
              f"model ({best_result['name']})...")
        oof_preds = out_of_fold_predictions(best_result["model"], X, y)
        cv_label = f"{plot_label} - CV" if plot_label else f"{best_result['name']} - CV"
        plot_predictions(y, oof_preds, cv_label, cv_plot_out_path,
                         color=plot_color, ylabel=plot_ylabel)

    test_metrics = None
    if X_test is not None and y_test is not None and len(X_test) > 0:
        test_metrics = evaluate_on_test(
            results, X_test, y_test, name_width=name_width,
            plot_out_path=plot_out_path, best_name=best_result["name"],
            plot_color=plot_color, plot_ylabel=plot_ylabel)

    return results, best_result, test_metrics


def run_acoustic_pipeline(model_name, label, *, win_length_s=5.0, hop_length_s=3.0,
                          models=None, scaler=StandardScaler(), device=None,
                          max_samples=None, visualize=True):
    suppress_expected_warnings()
    device = select_device() if device is None else device
    print(f"Using device: {device}")

    print(f"Loading {model_name}...")
    feature_extractor, model, device = load_model(model_name, device)

    def _load(split):
        return load_audio_data(
            os.path.join(LABELS_DIR, split), feature_extractor, model, device,
            model_name, win_length_s=win_length_s, hop_length_s=hop_length_s,
            max_samples=max_samples)

    print("Loading training data...")
    X_train, y_train = _load("train_split.csv")
    print("Loading validation data...")
    X_val, y_val = _load("dev_split.csv")
    print("Loading test data...")
    X_test, y_test = _load("test_split.csv")

    if len(X_train) == 0 or len(X_val) == 0:
        print("Not enough data to train. Exiting.")
        return None

    X = np.concatenate([X_train, X_val])
    y = np.concatenate([y_train, y_val])
    print(f"\n[{label}] Total train+dev samples: {len(X)}")

    plot_out_path = cv_plot_out_path = None
    if visualize:
        safe_label = label.lower().replace(" ", "_")
        plot_out_path = media_path(f"{safe_label}_best_model_predictions.png")
        cv_plot_out_path = media_path(f"{safe_label}_best_model_cv_predictions.png")

    return run_regression_pipeline(
        X, y, X_test=X_test, y_test=y_test, models=models, scaler=scaler,
        summary_title=f"{label}: Cross-Validation Summary on Combined Train+Dev",
        plot_out_path=plot_out_path, cv_plot_out_path=cv_plot_out_path,
        plot_label=label)