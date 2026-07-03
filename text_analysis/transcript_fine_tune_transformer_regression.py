import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Why this script changed from "prompting" to "fine-tuning"
# ---------------------------------------------------------------------------
# The previous version asked a frozen instruction model (FLAN-T5 / an OpenAI
# chat model) to *read a transcript and guess a PHQ-8 score* with no training.
# That has three problems you ran into:
#   1. It is never trained on YOUR data, so it can't learn the dataset's scale
#      or how these particular patients talk -> weak, mean-collapsed predictions.
#   2. Generating a number as text is slow (one full generation per transcript)
#      and brittle (you have to parse the output).
#   3. The good hosted models need an OpenAI token you don't have.
#
# The depression-detection literature (e.g. Weber et al. 2025 fine-tuning a BERT
# regression head on clinical-interview text, and the DAIC/E-DAIC transcript
# regression benchmarks reaching MAE ~3.5-3.9) shows that *fine-tuning* a
# compact transformer with a regression head is the strongest and most practical
# text-only approach. So this script now:
#   - uses `distilroberta-base`, a small, fast, fully open model that fine-tunes
#     well on CPU and is a solid "promptable/general" encoder you can fine-tune;
#   - fixes the long-transcript problem by SPLITTING each interview into word
#     windows (chunks), so the model sees the WHOLE transcript, not just the
#     first 512 tokens;
#   - trains a single regression head (num_labels=1, MSE loss) on the chunks;
#   - aggregates the per-chunk predictions back to one score per participant by
#     averaging (a simple multi-instance scheme), then evaluates with MAE / RMSE
#     / Pearson and the same prediction plot as the other scripts.
#
# To swap in a different fine-tunable encoder just change MODEL_NAME below
# (e.g. 'roberta-base', 'bert-base-uncased', or a mental-health model such as
# 'mental/mental-roberta-base').

warnings.filterwarnings('ignore')
set_seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = 'distilroberta-base'   # open, fast, fine-tunable; swap freely
CHUNK_WORDS = 150                   # words per window (safely under the 512-token limit)
CHUNK_OVERLAP = 100                  # overlap so context isn't cut mid-thought
MAX_TOKENS = 256                    # tokenizer max length per chunk
NUM_EPOCHS = 4
BATCH_SIZE = 8
LEARNING_RATE = 2e-5

DATA_DIR = '../dataset/wwwedaic/data'
LABELS_DIR = '../dataset/wwwedaic/labels'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def pearson_corr(y_true, y_pred):
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    corr = pearsonr(y_true, y_pred)[0]
    return 0.0 if np.isnan(corr) else corr


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
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
                    full_text = " ".join(text_data)
                    texts.append(full_text)
                    labels.append(float(phq_score))
                    ids.append(p_id)
            except Exception as e:
                print(f"Error reading {transcript_path}: {e}")

    return ids, texts, labels


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return [""]
    step = max(1, chunk_words - overlap)
    return [
        " ".join(words[i:i + chunk_words])
        for i in range(0, len(words), step)
    ]


def build_chunks(texts, labels):
    """Explode each transcript into chunks. Every chunk inherits its parent
    transcript's PHQ score (label) and remembers its parent index so we can
    average the chunk predictions back into one score per participant."""
    chunk_texts, chunk_labels, chunk_doc_idx = [], [], []
    for doc_idx, (text, label) in enumerate(zip(texts, labels)):
        for chunk in chunk_text(text):
            chunk_texts.append(chunk)
            chunk_labels.append(label)
            chunk_doc_idx.append(doc_idx)
    return chunk_texts, np.array(chunk_labels, dtype=float), np.array(chunk_doc_idx)


# ---------------------------------------------------------------------------
# Torch dataset
# ---------------------------------------------------------------------------
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
    """Average the per-chunk predictions for every parent transcript."""
    doc_preds = np.zeros(n_docs, dtype=float)
    for d in range(n_docs):
        mask = chunk_doc_idx == d
        doc_preds[d] = chunk_preds[mask].mean() if mask.any() else 0.0
    return doc_preds


def plot_predictions(y_true, y_pred, model_name, out_path):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    order = np.argsort(y_true)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    x = np.arange(len(y_true_sorted))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson = pearson_corr(y_true, y_pred)

    plt.figure(figsize=(14, 7))
    plt.plot(x, y_true_sorted, color='black', linewidth=2, label='Actual PHQ score')
    plt.vlines(x, y_true_sorted, y_pred_sorted, color='lightgray', linewidth=1, zorder=1)
    plt.scatter(x, y_pred_sorted, alpha=0.8, color='indianred', edgecolors='k',
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


def main():
    print(f"Using device: {DEVICE}")

    print("Loading training data...")
    train_ids, train_texts, train_labels = load_data(os.path.join(LABELS_DIR, 'train_split.csv'))

    print("Loading development data...")
    dev_ids, dev_texts, dev_labels = load_data(os.path.join(LABELS_DIR, 'dev_split.csv'))

    print(f"Loaded {len(train_texts)} train transcripts, {len(dev_texts)} dev transcripts.")

    # Fine-tune on TRAIN, evaluate on DEV (a real held-out split, unlike the
    # zero-shot version which had nothing to train on).
    train_chunk_texts, train_chunk_labels, _ = build_chunks(train_texts, train_labels)
    dev_chunk_texts, dev_chunk_labels, dev_chunk_doc_idx = build_chunks(dev_texts, dev_labels)
    print(f"Train chunks: {len(train_chunk_texts)}, Dev chunks: {len(dev_chunk_texts)}")

    # Standardise the regression target for stable training; we invert the
    # scaling on the predictions before computing metrics on the real PHQ scale.
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

    train_dataset = ChunkDataset(train_enc, train_chunk_labels_norm)
    dev_dataset = ChunkDataset(dev_enc)  # labels handled manually after aggregation

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
    # Invert the target standardisation to get predictions on the PHQ scale.
    chunk_preds = raw_preds * y_std + y_mean

    # Aggregate chunk-level predictions into one score per dev participant.
    dev_doc_preds = aggregate_by_doc(chunk_preds, dev_chunk_doc_idx, len(dev_texts))
    dev_doc_preds = np.clip(dev_doc_preds, 0, 24)
    dev_true = np.array(dev_labels, dtype=float)

    print(f"\n--- Fine-tuned {MODEL_NAME} PHQ-8 Regression (held-out dev) ---")
    print(f"MAE:     {mean_absolute_error(dev_true, dev_doc_preds):.4f}")
    print(f"RMSE:    {np.sqrt(mean_squared_error(dev_true, dev_doc_preds)):.4f}")
    print(f"Pearson: {pearson_corr(dev_true, dev_doc_preds):.4f}")

    baseline_mae = mean_absolute_error(dev_true, [y_mean] * len(dev_true))
    print(f"\nBaseline (predict train-mean) MAE: {baseline_mae:.4f}")

    plot_predictions(
        dev_true,
        dev_doc_preds,
        f"Fine-tuned {MODEL_NAME}",
        '../media/finetuned_best_model_predictions.png'
    )


if __name__ == "__main__":
    main()
