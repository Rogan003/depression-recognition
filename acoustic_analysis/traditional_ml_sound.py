import os
import librosa

import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import loguniform, uniform, pearsonr

from common import get_mfcc_windows


def pearson_scorer(y_true, y_pred):
    # Check for constant or nearly constant input arrays.
    # pearsonr is undefined if an input array is constant (standard deviation is 0).
    # NearConstantInputWarning occurs when standard deviation is extremely low.
    if np.std(y_pred) < 1e-9 or np.std(y_true) < 1e-9:
        return 0.0
    corr, _ = pearsonr(y_true, y_pred)
    return corr

def load_audio_features(csv_path):
    df = pd.read_csv(csv_path)

    X = []
    y = []

    for index, row in df.iterrows():
        participant_id = int(row['Participant_ID'])
        score = row['PHQ_Score']
        file_path = f"../dataset/wwwedaic/data/{participant_id}_P/{participant_id}_AUDIO.wav"

        if os.path.exists(file_path):
            print(f"Processing {file_path}...")
            windows = get_mfcc_windows(file_path, 40, 30, 30)
            if len(windows) == 0:
                print(f"Warning: no windows extracted from {file_path}.")
                continue

            windows = np.array(windows, dtype=np.float32)
            mfcc = windows[:, 0, :, :]
            delta = windows[:, 1, :, :]
            delta2 = windows[:, 2, :, :]
            features = np.concatenate([
                mfcc.mean(axis=(0, 2)),
                mfcc.std(axis=(0, 2)),
                mfcc.min(axis=(0, 2)),
                mfcc.max(axis=(0, 2)),
                delta.mean(axis=(0, 2)),
                delta.std(axis=(0, 2)),
                delta2.mean(axis=(0, 2)),
                delta2.std(axis=(0, 2)),
            ])

            X.append(features)
            y.append(score)
        else:
            print(f"Warning: {file_path} not found.")

    return np.array(X), np.array(y)

def run_search(name, estimator, param_dist, X, y, scoring, n_iter=100):
    print(f"\nHyperparameter tuning for {name} model...")
    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        refit='MAE',
        cv=5,
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X, y)
    
    print(f"Best Parameters for {name}:")
    print(random_search.best_params_)
    
    best_index = random_search.best_index_
    results = random_search.cv_results_
    
    print(f"Best Cross-Validation Results (refit by MAE):")
    print(f"MAE: {-results['mean_test_MAE'][best_index]:.4f}")
    print(f"RMSE: {-results['mean_test_RMSE'][best_index]:.4f}")
    print(f"Pearson correlation: {results['mean_test_Pearson'][best_index]:.4f}")
    
    return {
        'name': name,
        'best_params': random_search.best_params_,
        'MAE': -results['mean_test_MAE'][best_index],
        'RMSE': -results['mean_test_RMSE'][best_index],
        'Pearson': results['mean_test_Pearson'][best_index],
        'estimator': random_search.best_estimator_
    }

def main():
    print("Loading training data...")
    X_train, y_train = load_audio_features("../dataset/wwwedaic/labels/train_split.csv")
    
    print("Loading validation data...")
    X_val, y_val = load_audio_features("../dataset/wwwedaic/labels/dev_split.csv")

    X = np.concatenate([X_train, X_val])
    y = np.concatenate([y_train, y_val])
    
    print(f"Total samples: {len(X)}")
    
    scoring = {
        'MAE': 'neg_mean_absolute_error',
        'RMSE': 'neg_root_mean_squared_error',
        'Pearson': make_scorer(pearson_scorer)
    }
    
    models_to_run = [
        {
            'name': 'SVR',
            'estimator': Pipeline([
                ('scaler', MinMaxScaler()),
                ('model', SVR())
            ]),
            'param_dist': {
                'model__C': loguniform(1e-3, 1e3),
                'model__epsilon': uniform(0.001, 0.5),
                'model__gamma': loguniform(1e-4, 1e-1),
                'model__kernel': ['rbf', 'linear', 'poly']
            },
            'n_iter': 1000
        },
        {
            'name': 'ElasticNet',
            'estimator': Pipeline([
                ('scaler', MinMaxScaler()),
                ('model', ElasticNet(max_iter=10000))
            ]),
            'param_dist': {
                'model__alpha': uniform(0.0001, 10),
                'model__l1_ratio': uniform(0, 1)
            },
            'n_iter': 1000
        },
        {
            'name': 'KNN',
            'estimator': Pipeline([
                ('scaler', MinMaxScaler()),
                ('model', KNeighborsRegressor())
            ]),
            'param_dist': {
                'model__n_neighbors': range(1, 51),
                'model__weights': ['uniform', 'distance'],
                'model__p': [1, 2, 3],
                'model__metric': ['minkowski', 'euclidean', 'manhattan']
            },
            'n_iter': 1000
        },
        {
            'name': 'RandomForest',
            'estimator': Pipeline([
                ('scaler', MinMaxScaler()),
                ('model', RandomForestRegressor(random_state=42))
            ]),
            'param_dist': {
                'model__n_estimators': [100, 200, 500],
                'model__max_features': ['sqrt', 'log2', None],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__bootstrap': [True, False]
            },
            'n_iter': 40
        }
    ]
    
    all_results = []
    for model_info in models_to_run:
        res = run_search(
            model_info['name'],
            model_info['estimator'],
            model_info['param_dist'],
            X,
            y,
            scoring,
            n_iter=model_info['n_iter']
        )
        all_results.append(res)
    
    print("\n" + "="*50)
    print("FINAL SUMMARY OF ALL MODELS")
    print("="*50)
    print(f"{'Model':<15} | {'MAE':<8} | {'RMSE':<8} | {'Pearson':<8}")
    print("-" * 50)

    for res in all_results:
        print(f"{res['name']:<15} | {res['MAE']:<8.4f} | {res['RMSE']:<8.4f} | {res['Pearson']:<8.4f}")


    # X_test, y_test = load_audio_features("../dataset/wwwedaic/labels/test_split.csv")
    #
    # print("FINAL SUMMARY OF THE BEST MODEL ON TEST")
    # print("=" * 50)
    #
    # best_model = min(all_results, key=lambda x: x['MAE'] + x['RMSE'] * (3/5) - 10 * x['Pearson'])
    # y_pred = best_model['estimator'].predict(X_test)
    # print(f"{best_model['name']} | {best_model['best_params']}")
    # print(f"{mean_absolute_error(y_test, y_pred):<8.4f} | {np.sqrt(mean_squared_error(y_test, y_pred)):<8.4f} | {pearson_scorer(y_test, y_pred):<8.4f}")

if __name__ == "__main__":
    main()
