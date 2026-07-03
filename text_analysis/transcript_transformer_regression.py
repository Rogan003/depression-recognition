import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

import common

common.suppress_expected_warnings()

CHUNK_WORDS = 150
CHUNK_OVERLAP = 100


def main():
    print("Loading training data...")
    train_ids, train_texts, train_labels = common.load_data(
        os.path.join(common.LABELS_DIR, 'train_split.csv'))

    print("Loading development data...")
    dev_ids, dev_texts, dev_labels = common.load_data(
        os.path.join(common.LABELS_DIR, 'dev_split.csv'))

    print(f"Loaded {len(train_texts)} train transcripts, {len(dev_texts)} dev transcripts.")

    print("Loading SentenceTransformer model (all-mpnet-base-v2)...")
    embedder = SentenceTransformer('all-mpnet-base-v2')

    print("Encoding train texts (chunked over the full transcript)...")
    train_embeddings = common.encode_long_texts(embedder, train_texts, CHUNK_WORDS, CHUNK_OVERLAP)

    print("Encoding dev texts (chunked over the full transcript)...")
    dev_embeddings = common.encode_long_texts(embedder, dev_texts, CHUNK_WORDS, CHUNK_OVERLAP)

    X_train_dev = np.vstack([train_embeddings, dev_embeddings])
    y_train_dev = np.concatenate([train_labels, dev_labels])

    scaler = StandardScaler()
    X_train_dev_scaled = scaler.fit_transform(X_train_dev)

    common.run_regression_pipeline(
        X_train_dev_scaled,
        y_train_dev,
        common.media_path('transformer_best_model_predictions.png'),
        models=common.default_models(bayesian_needs_dense=True),
    )

    print("\nNote: Visualizing word importance for dense transformer embeddings is non-trivial compared to TF-IDF.")
    print("Please refer to the TF-IDF script output ('tfidf_feature_importance.png') for the most valuable words.")


if __name__ == "__main__":
    main()
