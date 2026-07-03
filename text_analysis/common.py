import warnings

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


def pearson_corr(y_true, y_pred):
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    return pearsonr(y_true, y_pred)[0]


def pearson_scorer(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if np.std(y_pred) < 1e-8 or np.std(y_true) < 1e-8:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        corr, _ = pearsonr(y_true, y_pred)
    return 0.0 if np.isnan(corr) else corr

def plot_predictions(y_true, y_pred, model_name, out_path):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Sort samples by the actual PHQ score so the "real" values form a smooth
    # ascending reference line and the predictions can be compared against it.
    order = np.argsort(y_true)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    x = np.arange(len(y_true_sorted))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson = pearson_scorer(y_true, y_pred)

    plt.figure(figsize=(14, 7))

    # Actual values as a clear reference line.
    plt.plot(x, y_true_sorted, color='black', linewidth=2, label='Actual PHQ score')

    # Predictions as dots, with thin connectors to their corresponding actual
    # value to make the residual (error) for each sample visible at a glance.
    plt.vlines(x, y_true_sorted, y_pred_sorted, color='lightgray', linewidth=1, zorder=1)
    plt.scatter(x, y_pred_sorted, alpha=0.8, color='steelblue', edgecolors='k',
                zorder=2, label='Predicted PHQ score')

    plt.xlabel('Samples (sorted by actual PHQ score)')
    plt.ylabel('PHQ score')
    plt.title(
        f'Predictions vs Actual ({model_name})\n'
        f'MAE={mae:.3f}, RMSE={rmse:.3f}, Pearson={pearson:.3f}'
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved prediction visualization to '{out_path}'")


def run_random_search(model_cfg, X, y, scoring):
    search = RandomizedSearchCV(
        estimator=clone(model_cfg['estimator']),
        param_distributions=model_cfg['param_dist'],
        n_iter=model_cfg['n_iter'],
        cv=5,
        scoring=scoring,
        refit='MAE',
        n_jobs=-1,
        random_state=42
    )
    search.fit(X, y)

    best_idx = search.best_index_
    cv_results = search.cv_results_
    cv_mae = -cv_results['mean_test_MAE'][best_idx]
    cv_rmse = -cv_results['mean_test_RMSE'][best_idx]
    cv_pearson = cv_results['mean_test_Pearson'][best_idx]

    print(f"Best {model_cfg['name']} params: {search.best_params_}")
    print(f"CV MAE: {cv_mae:.4f}, RMSE: {cv_rmse:.4f}, Pearson: {cv_pearson:.4f}")

    return {
        'name': model_cfg['name'],
        'MAE': cv_mae,
        'RMSE': cv_rmse,
        'Pearson': cv_pearson,
        'model': search.best_estimator_,
        'params': search.best_params_
    }


def model_score_for_picking(model_result):
    return model_result['MAE'] + model_result['RMSE'] * (3 / 5) - 10 * model_result['Pearson']

DATA_DIR = '../dataset/wwwedaic/data'
LABELS_DIR = '../dataset/wwwedaic/labels'