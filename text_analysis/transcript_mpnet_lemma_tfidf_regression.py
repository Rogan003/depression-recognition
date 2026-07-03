import os

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

import common

common.suppress_expected_warnings()

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

_lemmatizer = WordNetLemmatizer()

TOP_K_WORDS = 200


def lemma_tokenizer(text):
    tokens = word_tokenize(text.lower())
    return [_lemmatizer.lemmatize(tok) for tok in tokens if tok.isalpha()]


def build_reduced_texts(texts, top_k=TOP_K_WORDS):
    vectorizer = TfidfVectorizer(
        tokenizer=lemma_tokenizer,
        token_pattern=None,
        stop_words='english',
        max_features=15000
    )
    tfidf = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())

    reduced = []
    for i in range(tfidf.shape[0]):
        row = tfidf.getrow(i)
        if row.nnz == 0:
            reduced.append("")
            continue

        data = row.data
        cols = row.indices
        order = np.argsort(data)[::-1][:top_k]
        top_terms = vocab[cols[order]]
        reduced.append(" ".join(top_terms))
    return reduced


def main():
    print("Loading training data...")
    train_ids, train_texts, train_labels = common.load_data(
        os.path.join(common.LABELS_DIR, 'train_split.csv'))

    print("Loading development data...")
    dev_ids, dev_texts, dev_labels = common.load_data(
        os.path.join(common.LABELS_DIR, 'dev_split.csv'))

    print(f"Loaded {len(train_texts)} train transcripts, {len(dev_texts)} dev transcripts.")

    all_texts = train_texts + dev_texts
    print(f"Lemmatizing and selecting top-{TOP_K_WORDS} TF-IDF words per transcript...")
    reduced_texts = build_reduced_texts(all_texts, top_k=TOP_K_WORDS)

    print("Loading SentenceTransformer model (all-mpnet-base-v2)...")
    embedder = SentenceTransformer('all-mpnet-base-v2')

    print("Encoding reduced texts...")
    embeddings = embedder.encode(reduced_texts, show_progress_bar=True)

    X_train_dev = np.asarray(embeddings)
    y_train_dev = np.concatenate([train_labels, dev_labels])

    scaler = StandardScaler()
    X_train_dev_scaled = scaler.fit_transform(X_train_dev)

    common.run_regression_pipeline(
        X_train_dev_scaled,
        y_train_dev,
        common.media_path('mpnet_lemma_tfidf_best_model_predictions.png'),
        summary_title='Cross-Validation Summary (mpnet + lemma + TF-IDF top words)',
        label_fn=lambda name: f"mpnet+lemma+TFIDF + {name}",
        plot_color='seagreen',
    )


if __name__ == "__main__":
    main()
