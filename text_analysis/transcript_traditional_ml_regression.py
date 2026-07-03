import os

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.inspection import permutation_importance

import common

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

lemmatizer = WordNetLemmatizer()


def custom_lemma_tokenizer(text):
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(token) for token in tokens]


def create_interpretability_plots(best_model_info, all_results, X_test_tfidf, y_test, feature_names, svd=None):
    common.ensure_media_dir()
    best_model = best_model_info['model']

    if svd is not None:
        from sklearn.pipeline import Pipeline
        model_for_perm = Pipeline([
            ('svd', svd),
            ('model', best_model)
        ])
    else:
        model_for_perm = best_model

    X_test_dense = X_test_tfidf.toarray() if hasattr(X_test_tfidf, "toarray") else X_test_tfidf

    perm = permutation_importance(
        model_for_perm,
        X_test_dense,
        y_test,
        scoring='neg_mean_absolute_error',
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )
    permutation_values = perm.importances_mean
    common.plot_unsigned_feature_importance(
        permutation_values,
        feature_names,
        f"{best_model_info['name']} permutation",
        common.media_path('tfidf_best_model_permutation_importance.png')
    )

    linear_candidates = [r for r in all_results if hasattr(r['model'], 'coef_')]
    if linear_candidates:
        best_linear = min(linear_candidates, key=common.model_score_for_picking)
        coef = np.asarray(best_linear['model'].coef_).ravel()
        if svd is not None:
            coef = coef @ svd.components_
        common.plot_signed_feature_importance(
            coef,
            feature_names,
            best_linear['name'],
            common.media_path('tfidf_best_linear_coefficients.png')
        )

    tree_candidates = [r for r in all_results if hasattr(r['model'], 'feature_importances_')]
    if tree_candidates:
        if svd is not None:
            print("Skipping tree-specific feature importance because TruncatedSVD mapping is not straightforward.")
        else:
            best_tree = min(tree_candidates, key=common.model_score_for_picking)
            tree_values = np.asarray(best_tree['model'].feature_importances_)
            common.plot_unsigned_feature_importance(
                tree_values,
                feature_names,
                best_tree['name'],
                common.media_path('tfidf_best_tree_feature_importance.png')
            )


def main():
    print("Loading training data...")
    train_ids, train_texts, train_labels = common.load_data(
        os.path.join(common.LABELS_DIR, 'train_split.csv'), lowercase=True, verbose=True)

    print("Loading development data...")
    dev_ids, dev_texts, dev_labels = common.load_data(
        os.path.join(common.LABELS_DIR, 'dev_split.csv'), lowercase=True, verbose=True)

    print("Loading test data...")
    test_ids, test_texts, test_labels = common.load_data(
        os.path.join(common.LABELS_DIR, 'test_split.csv'), lowercase=True, verbose=True)

    print(f"Loaded {len(train_texts)} train transcripts, {len(dev_texts)} dev transcripts, "
          f"{len(test_texts)} test transcripts.")

    # Combine train and dev for cross-validation
    X_train_dev_texts = train_texts + dev_texts
    y_train_dev = np.concatenate([train_labels, dev_labels])

    vectorizer = TfidfVectorizer(
        tokenizer=custom_lemma_tokenizer,
        token_pattern=None,
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1,
        max_df=0.93
    )

    print("Vectorizing text data...")
    X_train_dev_tfidf = vectorizer.fit_transform(X_train_dev_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    print("Applying TruncatedSVD...")
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_train_dev_features = svd.fit_transform(X_train_dev_tfidf)
    X_test_features = svd.transform(X_test_tfidf)

    models_to_run = common.default_models(bayesian_needs_dense=True)

    print("\nTraining and tuning models...")
    results_for_picking = common.cross_validate_models(
        models_to_run, X_train_dev_features, y_train_dev)

    common.print_cv_summary(results_for_picking, 'Cross-Validation Summary on Combined Train+Dev',
                            name_width=18)

    best_model_info = min(results_for_picking, key=common.model_score_for_picking)
    print(f"\n>>> Best Model Picked: {best_model_info['name']} <<<")

    # print("\n--- FINAL SUMMARY OF THE BEST MODEL ON TEST ---")
    # print("=" * 50)
    # best_name = best_model_info['name']
    # best_model = best_model_info['model']
    # X_test_for_pred = X_test_features.toarray() if best_model_info.get('needs_dense', False) and hasattr(X_test_features, 'toarray') else X_test_features
    # test_preds = best_model.predict(X_test_for_pred)
    # t_mae = mean_absolute_error(test_labels, test_preds)
    # t_rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
    # t_pearson = common.pearson_scorer(test_labels, test_preds)
    # print(f"{best_name} | {best_model_info['params']}")
    # print(f"MAE: {t_mae:.4f} | RMSE: {t_rmse:.4f} | Pearson: {t_pearson:.4f}")
    # print("=" * 50)
    #
    # # Plot predictions of the best model (on the original PHQ scale by adding back the mean).
    # common.plot_predictions(
    #     test_labels,
    #     test_preds,
    #     best_name,
    #     common.media_path('tfidf_best_model_predictions.png')
    # )

    print("\n--- Interpretability Visualizations ---")
    feature_names = vectorizer.get_feature_names_out()
    create_interpretability_plots(
        best_model_info,
        results_for_picking,
        X_test_tfidf,
        np.array(test_labels),
        feature_names,
        svd=svd
    )


if __name__ == "__main__":
    main()
