import os
import re
import warnings

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import common


warnings.filterwarnings('ignore')

MODEL_NAME = 'google/flan-t5-base'

CHUNK_WORDS = 300
CHUNK_OVERLAP = 0

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def parse_item_score(raw_text):
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
        chunks = common.chunk_text(transcript, CHUNK_WORDS, CHUNK_OVERLAP)

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

        items_75 = np.percentile(item_totals, 75)
        total = items_75.sum()
        return float(np.clip(total, 0, 24))


def main():
    print(f"Using device: {DEVICE}")

    print("Loading training data...")
    train_ids, train_texts, train_labels = common.load_data(
        os.path.join(common.LABELS_DIR, 'train_split.csv'))

    print("Loading development data...")
    dev_ids, dev_texts, dev_labels = common.load_data(
        os.path.join(common.LABELS_DIR, 'dev_split.csv'))

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
    common.print_point_metrics(all_labels, preds)
    common.print_baseline(all_labels)

    common.plot_predictions(
        all_labels,
        preds,
        f"Zero-shot prompt ({MODEL_NAME})",
        common.media_path('prompt_zeroshot_best_model_predictions.png')
    )


if __name__ == "__main__":
    main()
