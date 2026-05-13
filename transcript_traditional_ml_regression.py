import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, loguniform, uniform

DATA_DIR = 'dataset/wwwedaic/data'
LABELS_DIR = 'dataset/wwwedaic/labels'

def load_data(split_file):
    df_split = pd.read_csv(split_file)
    texts = []
    labels = []
    ids = []
    
    for _, row in df_split.iterrows():
        p_id = int(row['Participant_ID'])
        phq_score = row['PHQ_Score']
        
        transcript_path = os.path.join(DATA_DIR, f"{p_id}_P", f"{p_id}_Transcript.csv")
        if os.path.exists(transcript_path):
            try:
                df_trans = pd.read_csv(transcript_path)
                if 'Text' in df_trans.columns:
                    text_data = df_trans['Text'].dropna().astype(str).tolist()
                    text_data = [text.strip().lower() for text in text_data]
                    full_text = " ".join(text_data)
                    texts.append(full_text)
                    labels.append(phq_score)
                    ids.append(p_id)
                else:
                    print(f"Warning: 'Text' column not found in {transcript_path}")
            except Exception as e:
                print(f"Error reading {transcript_path}: {e}")
        else:
            print(f"Warning: Transcript not found for participant {p_id}")
            
    return ids, texts, labels

def ensure_media_dir():
    os.makedirs('media', exist_ok=True)


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

def pearson_scorer(y_true, y_pred):
    if np.std(y_pred) < 1e-9 or np.std(y_true) < 1e-9:
        return 0.0
    corr, _ = pearsonr(y_true, y_pred)
    return corr


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


def create_interpretability_plots(best_model_info, all_results, X_test_tfidf, y_test, feature_names):
    ensure_media_dir()
    best_model = best_model_info['model']

    # Generic model-agnostic importance for whichever model wins.
    # Convert sparse X_test_tfidf to dense array because permutation_importance 
    # might raise InvalidParameterError with some sklearn versions/models if sparse.
    X_test_dense = X_test_tfidf.toarray() if hasattr(X_test_tfidf, "toarray") else X_test_tfidf

    perm = permutation_importance(
        best_model,
        X_test_dense,
        y_test,
        scoring='neg_mean_absolute_error',
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )
    permutation_values = perm.importances_mean
    plot_unsigned_feature_importance(
        permutation_values,
        feature_names,
        f"{best_model_info['name']} permutation",
        'media/tfidf_best_model_permutation_importance.png'
    )

    # Extra signed view when a linear model is available.
    linear_candidates = [r for r in all_results if hasattr(r['model'], 'coef_')]
    if linear_candidates:
        best_linear = min(linear_candidates, key=model_score_for_picking)
        coef = np.asarray(best_linear['model'].coef_).ravel()
        plot_signed_feature_importance(
            coef,
            feature_names,
            best_linear['name'],
            'media/tfidf_best_linear_coefficients.png'
        )

    # Optional tree-specific view.
    tree_candidates = [r for r in all_results if hasattr(r['model'], 'feature_importances_')]
    if tree_candidates:
        best_tree = min(tree_candidates, key=model_score_for_picking)
        tree_values = np.asarray(best_tree['model'].feature_importances_)
        plot_unsigned_feature_importance(
            tree_values,
            feature_names,
            best_tree['name'],
            'media/tfidf_best_tree_feature_importance.png'
        )

def main():
    print("Loading training data...")
    train_ids, train_texts, train_labels = load_data(os.path.join(LABELS_DIR, 'train_split.csv'))
    
    print("Loading development data...")
    dev_ids, dev_texts, dev_labels = load_data(os.path.join(LABELS_DIR, 'dev_split.csv'))

    print("Loading test data...")
    test_ids, test_texts, test_labels = load_data(os.path.join(LABELS_DIR, 'test_split.csv'))
    
    print(f"Loaded {len(train_texts)} train transcripts, {len(dev_texts)} dev transcripts, {len(test_texts)} test transcripts.")

    # Combine train and dev for cross-validation
    X_train_dev_texts = train_texts + dev_texts
    y_train_dev = np.concatenate([train_labels, dev_labels])

    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1,
        max_df=0.93
    )

    print("Vectorizing text data...")
    X_train_dev_tfidf = vectorizer.fit_transform(X_train_dev_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    scoring = {
        'MAE': 'neg_mean_absolute_error',
        'RMSE': 'neg_root_mean_squared_error',
        'Pearson': make_scorer(pearson_scorer)
    }

    models_to_run = [
        {
            'name': 'Ridge',
            'estimator': Ridge(random_state=42),
            'param_dist': {
                'alpha': loguniform(1e-4, 1e3)
            },
            'n_iter': 1000
        },
        {
            'name': 'ElasticNet',
            'estimator': ElasticNet(max_iter=10000, random_state=42),
            'param_dist': {
                'alpha': loguniform(1e-4, 1e3),
                'l1_ratio': uniform(0, 1)
            },
            'n_iter': 1000
        },
        {
            'name': 'SVR',
            'estimator': SVR(),
            'param_dist': {
                'C': loguniform(1e-4, 1e4),
                'epsilon': uniform(0.001, 0.8),
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear', 'polynomial']
            },
            'n_iter': 1000
        },
        {
            'name': 'RandomForest',
            'estimator': RandomForestRegressor(random_state=42),
            'param_dist': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'n_iter': 50
        },
        {
            'name': 'GradientBoosting',
            'estimator': GradientBoostingRegressor(random_state=42),
            'param_dist': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            'n_iter': 20
        }
    ]

    results_for_picking = []
    print("\nTraining and tuning models...")
    for model_cfg in models_to_run:
        print(f"Running {model_cfg['name']}...")
        result = run_random_search(model_cfg, X_train_dev_tfidf, y_train_dev, scoring)
        results_for_picking.append(result)

    print("\n--- Cross-Validation Summary on Combined Train+Dev ---")
    print(f"{'Model':<18} | {'CV MAE':<8} | {'CV RMSE':<8} | {'CV Pearson':<10}")
    print("-" * 60)
    for result in results_for_picking:
        print(
            f"{result['name']:<18} | {result['MAE']:<8.4f} | "
            f"{result['RMSE']:<8.4f} | {result['Pearson']:<10.4f}"
        )

    # Picking the best model using the same heuristic as in traditional_ml_sound.py
    best_model_info = min(results_for_picking, key=model_score_for_picking)
    print(f"\n>>> Best Model Picked: {best_model_info['name']} <<<")


    # print("\n--- FINAL SUMMARY OF THE BEST MODEL ON TEST ---")
    # print("=" * 50)
    # best_name = best_model_info['name']
    # best_model = best_model_info['model']
    # test_preds = best_model.predict(X_test_tfidf)
    # t_mae = mean_absolute_error(test_labels, test_preds)
    # t_rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
    # t_pearson = pearson_scorer(test_labels, test_preds)
    # print(f"{best_name} | {best_model_info['params']}")
    # print(f"MAE: {t_mae:.4f} | RMSE: {t_rmse:.4f} | Pearson: {t_pearson:.4f}")
    # print("=" * 50)

    # TODO: Add multiple words vectorization, it was already tried and it was just a tiny bit worse than this. Also try combination of the 2 approaches.

    print("\n--- Interpretability Visualizations ---")
    feature_names = vectorizer.get_feature_names_out()
    create_interpretability_plots(
        best_model_info,
        results_for_picking,
        X_test_tfidf,
        np.array(test_labels),
        feature_names
    )

if __name__ == "__main__":
    main()
