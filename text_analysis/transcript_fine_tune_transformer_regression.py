import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error

import common

warnings.filterwarnings('ignore')
set_seed(42)

MODEL_NAME = 'distilroberta-base'
CHUNK_WORDS = 150
CHUNK_OVERLAP = 100
MAX_TOKENS = 256
NUM_EPOCHS = 4
BATCH_SIZE = 8
LEARNING_RATE = 2e-5

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_chunks(texts, labels):
    chunk_texts, chunk_labels, chunk_doc_idx = [], [], []
    for doc_idx, (text, label) in enumerate(zip(texts, labels)):
        for chunk in common.chunk_text(text, CHUNK_WORDS, CHUNK_OVERLAP):
            chunk_texts.append(chunk)
            chunk_labels.append(label)
            chunk_doc_idx.append(doc_idx)
    return chunk_texts, np.array(chunk_labels, dtype=float), np.array(chunk_doc_idx)


class ChunkDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds).ravel()
    labels = np.asarray(labels).ravel()
    return {
        'mae': mean_absolute_error(labels, preds),
        'rmse': float(np.sqrt(mean_squared_error(labels, preds))),
    }


def aggregate_by_doc(chunk_preds, chunk_doc_idx, n_docs):
    doc_preds = np.zeros(n_docs, dtype=float)
    for d in range(n_docs):
        mask = chunk_doc_idx == d
        doc_preds[d] = chunk_preds[mask].mean() if mask.any() else 0.0
    return doc_preds


def main():
    print(f"Using device: {DEVICE}")

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

    train_chunk_texts, train_chunk_labels, _ = build_chunks(train_texts, train_labels)
    dev_chunk_texts, dev_chunk_labels, dev_chunk_doc_idx = build_chunks(dev_texts, dev_labels)
    test_chunk_texts, test_chunk_labels, test_chunk_doc_idx = build_chunks(test_texts, test_labels)
    print(f"Train chunks: {len(train_chunk_texts)}, Dev chunks: {len(dev_chunk_texts)}, "
          f"Test chunks: {len(test_chunk_texts)}")

    y_mean = float(np.mean(train_chunk_labels))
    y_std = float(np.std(train_chunk_labels)) or 1.0
    train_chunk_labels_norm = (train_chunk_labels - y_mean) / y_std

    print(f"Loading tokenizer and model '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1            # num_labels=1 -> regression (MSE loss)
    ).to(DEVICE)

    train_enc = tokenizer(
        train_chunk_texts, truncation=True, padding=True, max_length=MAX_TOKENS
    )
    dev_enc = tokenizer(
        dev_chunk_texts, truncation=True, padding=True, max_length=MAX_TOKENS
    )
    test_enc = tokenizer(
        test_chunk_texts, truncation=True, padding=True, max_length=MAX_TOKENS
    )

    train_dataset = ChunkDataset(train_enc, train_chunk_labels_norm)
    dev_dataset = ChunkDataset(dev_enc)  # labels handled manually after aggregation
    test_dataset = ChunkDataset(test_enc)  # labels handled manually after aggregation

    training_args = TrainingArguments(
        output_dir='./ft_checkpoints',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=32,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy='no',
        report_to=[],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    print("\nFine-tuning the regression head + transformer on train chunks...")
    trainer.train()

    print("\nPredicting on dev chunks...")
    raw_preds = trainer.predict(dev_dataset).predictions.ravel()
    chunk_preds = raw_preds * y_std + y_mean

    dev_doc_preds = aggregate_by_doc(chunk_preds, dev_chunk_doc_idx, len(dev_texts))
    dev_doc_preds = np.clip(dev_doc_preds, 0, 24)
    dev_true = np.array(dev_labels, dtype=float)

    print(f"\n--- Fine-tuned {MODEL_NAME} PHQ-8 Regression (held-out dev) ---")
    common.print_point_metrics(dev_true, dev_doc_preds)
    common.print_baseline(dev_true, mean_value=y_mean)

    common.plot_predictions(
        dev_true,
        dev_doc_preds,
        f"Fine-tuned {MODEL_NAME}",
        common.media_path('finetuned_best_model_predictions.png'),
        color='indianred'
    )

    print("\nPredicting on test chunks...")
    test_raw_preds = trainer.predict(test_dataset).predictions.ravel()
    test_chunk_preds = test_raw_preds * y_std + y_mean

    test_doc_preds = aggregate_by_doc(test_chunk_preds, test_chunk_doc_idx, len(test_texts))
    test_doc_preds = np.clip(test_doc_preds, 0, 24)
    test_true = np.array(test_labels, dtype=float)

    print(f"\n--- Fine-tuned {MODEL_NAME} PHQ-8 Regression (held-out test) ---")
    common.print_point_metrics(test_true, test_doc_preds)
    common.print_baseline(test_true, mean_value=y_mean)

    common.plot_predictions(
        test_true,
        test_doc_preds,
        f"Fine-tuned {MODEL_NAME}",
        common.media_path('finetuned_best_model_test_predictions.png'),
        color='indianred'
    )


if __name__ == "__main__":
    main()
