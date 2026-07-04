import os
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, loguniform, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge
from sklearn.metrics import (
    make_scorer, mean_absolute_error, mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVR

DATA_DIR = '../dataset/wwwedaic/data'
LABELS_DIR = '../dataset/wwwedaic/labels'
MEDIA_DIR = '../media'


def ensure_media_dir():
    os.makedirs(MEDIA_DIR, exist_ok=True)


def media_path(filename):
    return os.path.join(MEDIA_DIR, filename)


def suppress_expected_warnings():
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    try:
        from scipy.stats import ConstantInputWarning, NearConstantInputWarning
        warnings.filterwarnings('ignore', category=ConstantInputWarning)
        warnings.filterwarnings('ignore', category=NearConstantInputWarning)
    except ImportError:
        pass


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
        warnings.simplefilter('ignore')
        corr, _ = pearsonr(y_true, y_pred)
    return 0.0 if np.isnan(corr) else corr


def make_scoring():
    return {
        'MAE': 'neg_mean_absolute_error',
        'RMSE': 'neg_root_mean_squared_error',
        'Pearson': make_scorer(pearson_scorer),
    }


def model_score_for_picking(model_result):
    return model_result['MAE'] + model_result['RMSE'] * (3 / 5) - 10 * model_result['Pearson']


def load_data(split_file, lowercase=False, verbose=False):
    df_split = pd.read_csv(split_file)
    ids, texts, labels = [], [], []

    for _, row in df_split.iterrows():
        p_id = int(row['Participant_ID'])
        phq_score = row['PHQ_Score']

        transcript_path = os.path.join(DATA_DIR, f"{p_id}_P", f"{p_id}_Transcript.csv")
        if not os.path.exists(transcript_path):
            if verbose:
                print(f"Warning: Transcript not found for participant {p_id}")
            continue

        try:
            df_trans = pd.read_csv(transcript_path)
        except Exception as e:
            print(f"Error reading {transcript_path}: {e}")
            continue

        if 'Text' not in df_trans.columns:
            if verbose:
                print(f"Warning: 'Text' column not found in {transcript_path}")
            continue

        text_data = df_trans['Text'].dropna().astype(str).tolist()
        if lowercase:
            text_data = [text.strip().lower() for text in text_data]
        texts.append(" ".join(text_data))
        labels.append(phq_score)
        ids.append(p_id)

    return ids, texts, labels


def chunk_text(text, chunk_words, overlap):
    words = text.split()
    if not words:
        return [""]
    step = max(1, chunk_words - overlap)
    return [
        " ".join(words[i:i + chunk_words])
        for i in range(0, len(words), step)
    ]


def encode_long_texts(embedder, texts, chunk_words, overlap, show_progress_bar=True):
    doc_vectors = []
    for idx, text in enumerate(texts):
        if show_progress_bar:
            print(f"  Encoding transcript {idx + 1}/{len(texts)}", end='\r')
        chunks = chunk_text(text, chunk_words, overlap)
        chunk_embeddings = embedder.encode(chunks, show_progress_bar=False)
        doc_vectors.append(np.asarray(chunk_embeddings).mean(axis=0))
    if show_progress_bar:
        print()
    return np.vstack(doc_vectors)


def default_models(bayesian_needs_dense=False):
    bayesian_cfg = {
        'name': 'BayesianRidge',
        'estimator': BayesianRidge(),
        'param_dist': {
            'alpha_1': loguniform(1e-4, 1e-1),
            'alpha_2': loguniform(1e-4, 1e-1),
            'lambda_1': loguniform(1e-4, 1e-1),
            'lambda_2': loguniform(1e-4, 1e-1),
        },
        'n_iter': 20,
    }
    if bayesian_needs_dense:
        bayesian_cfg['needs_dense'] = True

    return [
        {
            'name': 'Ridge',
            'estimator': Ridge(random_state=42),
            'param_dist': {'alpha': loguniform(1e-4, 1e3)},
            'n_iter': 1000,
        },
        {
            'name': 'ElasticNet',
            'estimator': ElasticNet(max_iter=5000, random_state=42),
            'param_dist': {
                'alpha': loguniform(1e-4, 1e3),
                'l1_ratio': uniform(0, 1),
            },
            'n_iter': 1000,
        },
        {
            'name': 'SVR',
            'estimator': SVR(),
            'param_dist': {
                'C': loguniform(1e-4, 1e4),
                'epsilon': uniform(0.001, 0.8),
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear', 'poly'],
            },
            'n_iter': 1000,
        },
        {
            'name': 'RandomForest',
            'estimator': RandomForestRegressor(random_state=42),
            'param_dist': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            },
            'n_iter': 20,
        },
        {
            'name': 'GradientBoosting',
            'estimator': GradientBoostingRegressor(random_state=42),
            'param_dist': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
            },
            'n_iter': 20,
        },
        bayesian_cfg,
        {
            'name': 'MLPRegressor',
            'estimator': MLPRegressor(max_iter=1000, random_state=42, early_stopping=True),
            'param_dist': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': loguniform(1e-5, 1e-1),
                'learning_rate_init': loguniform(1e-4, 1e-1),
            },
            'n_iter': 50,
        },
    ]


def run_random_search(model_cfg, X, y, scoring, scaler=None):
    if model_cfg.get('needs_dense', False) and hasattr(X, 'toarray'):
        X = X.toarray()

    estimator = clone(model_cfg['estimator'])
    param_dist = model_cfg['param_dist']
    if scaler is not None:
        estimator = Pipeline([('scaler', clone(scaler)), ('model', estimator)])
        param_dist = {f'model__{key}': value for key, value in param_dist.items()}

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=model_cfg['n_iter'],
        cv=5,
        scoring=scoring,
        refit='MAE',
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X, y)

    best_idx = search.best_index_
    cv_results = search.cv_results_
    cv_mae = -cv_results['mean_test_MAE'][best_idx]
    cv_rmse = -cv_results['mean_test_RMSE'][best_idx]
    cv_pearson = cv_results['mean_test_Pearson'][best_idx]

    best_params = {key.replace('model__', '', 1): value
                   for key, value in search.best_params_.items()}

    print(f"Best {model_cfg['name']} params: {best_params}")
    print(f"CV MAE: {cv_mae:.4f}, RMSE: {cv_rmse:.4f}, Pearson: {cv_pearson:.4f}")

    return {
        'name': model_cfg['name'],
        'MAE': cv_mae,
        'RMSE': cv_rmse,
        'Pearson': cv_pearson,
        'model': search.best_estimator_,
        'params': best_params,
        'needs_dense': model_cfg.get('needs_dense', False),
    }


def cross_validate_models(models, X, y, scoring=None, scaler=None):
    scoring = make_scoring() if scoring is None else scoring
    results = []
    for model_cfg in models:
        print(f"\nRunning {model_cfg['name']}...")
        results.append(run_random_search(model_cfg, X, y, scoring, scaler=scaler))
    return results


def evaluate_on_test(results, X_test, y_test,
                     title='Test Set Evaluation (all models)', name_width=14,
                     plot_out_path=None, best_name=None,
                     plot_color='steelblue', plot_ylabel='PHQ score', label_fn=None):
    y_test = np.asarray(y_test, dtype=float)
    print(f"\n--- {title} ---")
    print(f"{'Model':<{name_width}} | {'MAE':<8} | {'RMSE':<8} | {'Pearson':<10}")
    print("-" * (name_width + 36))

    metrics = []
    for result in results:
        model = result.get('model')
        if model is None:
            continue
        X_eval = X_test
        if result.get('needs_dense', False) and hasattr(X_test, 'toarray'):
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
            'name': result['name'],
            'MAE': test_mae,
            'RMSE': test_rmse,
            'Pearson': test_pearson,
            'preds': preds,
        })

    if plot_out_path and metrics:
        best_metric = None
        if best_name is not None:
            best_metric = next((m for m in metrics if m['name'] == best_name), None)
        if best_metric is None:
            best_metric = min(metrics, key=model_score_for_picking)
        print(f"\nGenerating test-set prediction visualization for best model "
              f"({best_metric['name']})...")
        label = label_fn(best_metric['name']) if label_fn else best_metric['name']
        plot_predictions(y_test, best_metric['preds'], label, plot_out_path,
                         color=plot_color, ylabel=plot_ylabel)

    return metrics


def out_of_fold_predictions(model, X, y, cv=5):
    return cross_val_predict(clone(model), X, y, cv=cv, n_jobs=-1)


def print_cv_summary(results, title, name_width=14):
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


def print_point_metrics(y_true, y_pred):
    print(f"MAE:     {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE:    {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"Pearson: {pearson_corr(y_true, y_pred):.4f}")


def plot_predictions(y_true, y_pred, model_name, out_path, color='steelblue', ylabel='PHQ score'):
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
    plt.plot(x, y_true_sorted, color='black', linewidth=2, label='Actual PHQ score')
    plt.vlines(x, y_true_sorted, y_pred_sorted, color='lightgray', linewidth=1, zorder=1)
    plt.scatter(x, y_pred_sorted, alpha=0.8, color=color, edgecolors='k',
                zorder=2, label='Predicted PHQ score')
    plt.xlabel('Samples (sorted by actual PHQ score)')
    plt.ylabel(ylabel)
    plt.title(
        f'Predictions vs Actual ({model_name})\n'
        f'MAE={mae:.3f}, RMSE={rmse:.3f}, Pearson={pearson:.3f}'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved prediction visualization to '{out_path}'")


def plot_signed_feature_importance(values, feature_names, model_name, out_path, top_n=20):
    top_positive_idx = np.argsort(values)[-top_n:]
    top_negative_idx = np.argsort(values)[:top_n]
    top_idx = np.concatenate([top_negative_idx, top_positive_idx])

    features = [feature_names[i] for i in top_idx]
    importances = values[top_idx]

    plt.figure(figsize=(12, 10))
    colors = ['red' if c < 0 else 'blue' for c in importances]
    plt.barh(range(len(features)), importances, color=colors)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Signed coefficient value')
    plt.title(f'Top influential words/phrases ({model_name}, positive vs negative)')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved signed feature-importance visualization to '{out_path}'")


def plot_unsigned_feature_importance(values, feature_names, model_name, out_path, top_n=30):
    top_idx = np.argsort(values)[-top_n:]
    features = [feature_names[i] for i in top_idx]
    importances = values[top_idx]

    plt.figure(figsize=(12, 10))
    plt.barh(range(len(features)), importances, color='darkgreen')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title(f'Top influential words/phrases ({model_name})')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved feature-importance visualization to '{out_path}'")


def run_regression_pipeline(
    X, y, plot_out_path, *,
    X_test=None, y_test=None,
    models=None,
    scoring=None,
    scaler=StandardScaler(),
    summary_title='Cross-Validation Summary on Combined Train+Dev',
    name_width=14,
    label_fn=None,
    plot_color='steelblue',
    plot_ylabel='PHQ score',
    plot_cv_predictions=False,
    predict_on_test=True,
):
    models = default_models() if models is None else models
    scoring = make_scoring() if scoring is None else scoring

    print("\nTraining and tuning models with cross-validation...")
    results = cross_validate_models(models, X, y, scoring, scaler=scaler)

    print_cv_summary(results, summary_title, name_width)
    print_baseline(y)

    best_result = min(results, key=model_score_for_picking)
    print(f"\n>>> Best model picked: {best_result['name']} <<<")

    test_metrics = None
    if predict_on_test and X_test is not None and y_test is not None:
        test_metrics = evaluate_on_test(
            results, X_test, y_test, name_width=name_width,
            plot_out_path=plot_out_path, best_name=best_result['name'],
            plot_color=plot_color, plot_ylabel=plot_ylabel, label_fn=label_fn)

    if plot_cv_predictions:
        print(f"\nGenerating out-of-fold prediction visualization for best model "
              f"({best_result['name']})...")
        oof_preds = out_of_fold_predictions(best_result['model'], X, y)
        label = label_fn(best_result['name']) if label_fn else best_result['name']
        plot_predictions(y, oof_preds, label, plot_out_path,
                         color=plot_color, ylabel=plot_ylabel)

    return results, best_result, test_metrics
