import os
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Zero-shot PROMPT-based PHQ-8 estimation (no API keys, fast local model)
# ---------------------------------------------------------------------------
# This is the "prompting" approach you asked for, kept separate from the
# fine-tuned script. It needs NO training and NO API key: it simply *asks* a
# small, fast, locally-runnable instruction model to read the interview and
# return depression information.
#
# What changed vs. the old prompt attempt (which was slow and weak):
#   1. Model: google/flan-t5-base instead of flan-t5-large -> ~3x smaller and
#      much faster on CPU, while still being instruction-tuned enough to follow
#      a scoring prompt. (Swap MODEL_NAME to flan-t5-small for even more speed
#      or flan-t5-large for a bit more accuracy.)
#   2. Prompt strategy: inspired by the interpretable-LLM-prompting literature
#      (Lee et al., PLOS Digital Health 2025, MAE ~2.9 on DAIC-WOZ; and the
#      Reutlingen work on symptom-wise PHQ estimation), instead of asking for a
#      single 0-24 number in one shot (which mean-collapses badly), we ask the
#      model to rate EACH of the 8 PHQ-8 symptom items on its native 0-3 scale
#      and then SUM them. This mirrors how the PHQ-8 questionnaire is actually
#      constructed and gives far more grounded, less collapsed predictions.
#
# It is inference-only: there is no .fit()/CV step, we just prompt + evaluate.

warnings.filterwarnings('ignore')

MODEL_NAME = 'google/flan-t5-base'   # fast, open, no API key; swap freely
# flan-t5 has a 512-token context, so a full (very long) transcript cannot be
# fed in one shot. Like the embedding-based scripts, we therefore split each
# transcript into word windows, score every window, and aggregate (mean) the
# per-window item scores into one PHQ-8 estimate -- so we now look at the WHOLE
# interview instead of just the opening minutes.
#
# We use NON-overlapping windows here (overlap = 0). Overlap helps the embedding
# scripts because it avoids cutting a sentence mid-thought between two windows
# that get pooled together; but here every extra window costs 8 separate model
# generations (one per PHQ item), so overlap multiplies an already heavy
# inference cost for little benefit. Set CHUNK_OVERLAP > 0 if you want windows
# to overlap exactly like the transformer/emotion scripts.
CHUNK_WORDS = 300       # words per window
CHUNK_OVERLAP = 0       # 0 = non-overlapping; raise to overlap like other scripts

DATA_DIR = '../dataset/wwwedaic/data'
LABELS_DIR = '../dataset/wwwedaic/labels'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# The 8 PHQ-8 symptom items. We score each on 0-3 and sum to a 0-24 total,
# exactly how the real questionnaire works.
PHQ8_ITEMS = [
    "little interest or pleasure in doing things",
    "feeling down, depressed, or hopeless",
    "trouble falling or staying asleep, or sleeping too much",
    "feeling tired or having little energy",
    "poor appetite or overeating",
    "feeling bad about yourself, or that you are a failure",
    "trouble concentrating on things",
    "moving or speaking slowly, or being restless and fidgety",
]

ITEM_PROMPT_TEMPLATE = (
    "You are a clinical psychologist assistant. Read the following excerpt of a "
    "clinical interview with a patient. \n\n"
    "Transcript:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
    "Over the last two weeks, how severely was the patient been bothered by: "
    "{symptom}?\n"
    "Answer only with a single integer on this scale:\n"
    "0 = not at all, 1 = several days, 2 = more than half the days, "
    "3 = nearly every day.\n"
    "Be very hesitant and sure to give high grades.\n"
    "Examples for this symptom:\n{examples}\n"
    "Answer:"
)

# A few-shot example for every score (0/1/2/3) of every PHQ-8 item. The example
# is matched to whichever symptom the current prompt is about, so the model sees
# what each rating level looks like for THAT specific symptom before answering.
# Keys match the strings in PHQ8_ITEMS exactly.
ITEM_EXAMPLES = {
    "little interest or pleasure in doing things": (
        '- "I still enjoy my hobbies and look forward to seeing my friends." -> 0\n'
        '- "A few days I didn\'t really feel like doing much." -> 1\n'
        '- "More than half the days I had no interest in things I used to like." -> 2\n'
        '- "I don\'t enjoy anything at all anymore, every single day." -> 3'
    ),
    "feeling down, depressed, or hopeless": (
        '- "My mood has been good and I feel hopeful about the future." -> 0\n'
        '- "I felt a bit down on a couple of days." -> 1\n'
        '- "I have been feeling sad and hopeless most days." -> 2\n'
        '- "I feel completely hopeless and depressed every day." -> 3'
    ),
    "trouble falling or staying asleep, or sleeping too much": (
        '- "I sleep well and wake up feeling rested." -> 0\n'
        '- "I had trouble sleeping on a night or two." -> 1\n'
        '- "My sleep has been disturbed more than half the nights." -> 2\n'
        '- "I can barely sleep at all, night after night." -> 3'
    ),
    "feeling tired or having little energy": (
        '- "I have plenty of energy throughout the day." -> 0\n'
        '- "I felt tired on a few days." -> 1\n'
        '- "I am low on energy most of the days." -> 2\n'
        '- "I feel exhausted and drained every single day." -> 3'
    ),
    "poor appetite or overeating": (
        '- "My appetite is normal and eating is fine." -> 0\n'
        '- "On a couple of days my appetite was off." -> 1\n'
        '- "I have been over- or under-eating more than half the days." -> 2\n'
        '- "My eating is completely disrupted every day." -> 3'
    ),
    "feeling bad about yourself, or that you are a failure": (
        '- "I feel good about myself and what I do." -> 0\n'
        '- "I felt down on myself once or twice." -> 1\n'
        '- "I often feel like a failure these days." -> 2\n'
        '- "I feel worthless and like a failure every day." -> 3'
    ),
    "trouble concentrating on things": (
        '- "I can focus well on tasks and reading." -> 0\n'
        '- "My focus slipped on a couple of days." -> 1\n'
        '- "I struggle to concentrate most days." -> 2\n'
        '- "I cannot concentrate on anything at all, every day." -> 3'
    ),
    "moving or speaking slowly, or being restless and fidgety": (
        '- "I move and speak at my normal pace." -> 0\n'
        '- "I felt a bit restless on a day or two." -> 1\n'
        '- "I have been noticeably slowed down or restless most days." -> 2\n'
        '- "I feel slowed down or agitated every single day." -> 3'
    ),
}

def pearson_corr(y_true, y_pred):
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    corr = pearsonr(y_true, y_pred)[0]
    return 0.0 if np.isnan(corr) else corr


def chunk_text(text, chunk_words=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return [""]
    step = max(1, chunk_words - overlap)
    return [
        " ".join(words[i:i + chunk_words])
        for i in range(0, len(words), step)
    ]


def parse_item_score(raw_text):
    # Grab the first number the model emits and clamp to the 0-3 item scale.
    match = re.search(r'\d+', str(raw_text))
    if not match:
        return None
    return int(np.clip(int(match.group()), 0, 3))


class FlanItemScorer:
    def __init__(self, model_name=MODEL_NAME):
        print(f"Loading prompt model '{model_name}' on {DEVICE}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def score_transcript(self, transcript):
        """Split the transcript into word windows and prompt the model once per
        PHQ-8 item per window. For each item we MEAN the per-window scores
        (so the whole interview contributes, not just the opening), then sum the
        8 item means into a single 0-24 PHQ-8 estimate. Items the model fails to
        answer in a window are treated as 0 (symptom not reported)."""
        chunks = chunk_text(transcript)
        # per-item accumulated score and count of windows that produced a score
        item_totals = np.zeros(len(PHQ8_ITEMS), dtype=float)

        for chunk in chunks:
            for item_idx, symptom in enumerate(PHQ8_ITEMS):
                prompt = ITEM_PROMPT_TEMPLATE.format(
                    transcript=chunk,
                    symptom=symptom,
                    examples=ITEM_EXAMPLES[symptom],
                )
                enc = self.tokenizer(
                    prompt, return_tensors='pt', truncation=True, max_length=512
                ).to(DEVICE)
                with torch.no_grad():
                    out = self.model.generate(**enc, max_new_tokens=5)
                text = self.tokenizer.decode(out[0], skip_special_tokens=True)
                score = parse_item_score(text)
                item_totals[item_idx] += score if score is not None else 0

        # Mean per item across windows, then sum the item means.
        items_75 = np.percentile(item_totals, 75)
        total = items_75.sum()
        return float(np.clip(total, 0, 24))


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


def load_data(split_file):
    df_split = pd.read_csv(split_file)
    texts, labels, ids = [], [], []

    for _, row in df_split.iterrows():
        p_id = int(row['Participant_ID'])
        phq_score = row['PHQ_Score']

        transcript_path = os.path.join(DATA_DIR, f"{p_id}_P", f"{p_id}_Transcript.csv")
        if os.path.exists(transcript_path):
            try:
                df_trans = pd.read_csv(transcript_path)
                if 'Text' in df_trans.columns:
                    text_data = df_trans['Text'].dropna().astype(str).tolist()
                    texts.append(" ".join(text_data))
                    labels.append(float(phq_score))
                    ids.append(p_id)
            except Exception as e:
                print(f"Error reading {transcript_path}: {e}")

    return ids, texts, labels


def main():
    print(f"Using device: {DEVICE}")

    print("Loading training data...")
    train_ids, train_texts, train_labels = load_data(os.path.join(LABELS_DIR, 'train_split.csv'))

    print("Loading development data...")
    dev_ids, dev_texts, dev_labels = load_data(os.path.join(LABELS_DIR, 'dev_split.csv'))

    # Zero-shot prompting needs no training, so we score the combined train+dev
    # transcripts (matching the splits used by the other scripts).
    all_texts = train_texts + dev_texts
    all_labels = np.array(list(train_labels) + list(dev_labels), dtype=float)
    print(f"Scoring {len(all_texts)} transcripts via symptom-wise prompting...")

    scorer = FlanItemScorer()

    preds = []
    for idx, text in enumerate(all_texts):
        print(f"  Prompting transcript {idx + 1}/{len(all_texts)}", end='\r')
        preds.append(scorer.score_transcript(text))
        print(str(preds[-1]) + " vs " + str(all_labels[idx]))

    print()
    preds = np.array(preds, dtype=float)

    print(f"\n--- Zero-shot Prompt PHQ-8 Estimation ({MODEL_NAME}) ---")
    print(f"MAE:     {mean_absolute_error(all_labels, preds):.4f}")
    print(f"RMSE:    {np.sqrt(mean_squared_error(all_labels, preds)):.4f}")
    print(f"Pearson: {pearson_corr(all_labels, preds):.4f}")

    baseline_mae = mean_absolute_error(all_labels, [np.mean(all_labels)] * len(all_labels))
    print(f"\nBaseline (predict-the-mean) MAE: {baseline_mae:.4f}")

    plot_predictions(
        all_labels,
        preds,
        f"Zero-shot prompt ({MODEL_NAME})",
        '../media/prompt_zeroshot_best_model_predictions.png'
    )


if __name__ == "__main__":
    main()
