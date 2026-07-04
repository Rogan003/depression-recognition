import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import StandardScaler

import common

common.suppress_expected_warnings()

EMOTION_MODEL = 'j-hartmann/emotion-english-distilroberta-base'

CHUNK_WORDS = 300
CHUNK_OVERLAP = 200


@torch.no_grad()
def emotion_probs(tokenizer, model, device, chunks, batch_size=16):
    probs = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
        probs.append(model(**enc).logits)
    return np.vstack(probs)


def encode_long_texts(tokenizer, model, device, texts, show_progress_bar=True):
    doc_vectors = []
    for idx, text in enumerate(texts):
        if show_progress_bar:
            print(f"  Encoding transcript {idx + 1}/{len(texts)}", end='\r')
        chunks = common.chunk_text(text, CHUNK_WORDS, CHUNK_OVERLAP)
        chunk_probs = emotion_probs(tokenizer, model, device, chunks)
        doc_vectors.append(chunk_probs.mean(axis=0))
    if show_progress_bar:
        print()
    return np.vstack(doc_vectors)


def main():
    print("Loading training data...")
    train_ids, train_texts, train_labels = common.load_data(
        os.path.join(common.LABELS_DIR, 'train_split.csv'))

    print("Loading development data...")
    dev_ids, dev_texts, dev_labels = common.load_data(
        os.path.join(common.LABELS_DIR, 'dev_split.csv'))

    print("Loading test data...")
    test_ids, test_texts, test_labels = common.load_data(
        os.path.join(common.LABELS_DIR, 'test_split.csv'))

    print(f"Loaded {len(train_texts)} train transcripts, {len(dev_texts)} dev transcripts, "
          f"{len(test_texts)} test transcripts.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading emotion model '{EMOTION_MODEL}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL).to(device)
    model.eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(f"Emotion classes: {labels}")

    print("Encoding train texts (chunked over the full transcript)...")
    train_features = encode_long_texts(tokenizer, model, device, train_texts)

    print("Encoding dev texts (chunked over the full transcript)...")
    dev_features = encode_long_texts(tokenizer, model, device, dev_texts)

    print("Encoding test texts (chunked over the full transcript)...")
    test_features = encode_long_texts(tokenizer, model, device, test_texts)

    X_train_dev = np.vstack([train_features, dev_features])
    y_train_dev = np.concatenate([train_labels, dev_labels])

    common.run_regression_pipeline(
        X_train_dev,
        y_train_dev,
        common.media_path('emotion_best_model_predictions.png'),
        X_test=test_features,
        y_test=test_labels,
        scaler=StandardScaler(),
        summary_title='Cross-Validation Summary (Emotion features)',
        label_fn=lambda name: f"Emotion + {name}",
        plot_color='darkorange',
    )


if __name__ == "__main__":
    main()
