import os

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler

import common

common.suppress_expected_warnings()

CHUNK_WORDS = 200
CHUNK_OVERLAP = 50


class BertFeatureExtractor:
    def __init__(self, model_name='bert-base-uncased', batch_size=16):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading BERT model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.batch_size = batch_size

    @torch.no_grad()
    def _embed_chunks(self, chunks):
        vectors = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            out = self.model(**enc)

            token_embeddings = out.last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1).float()
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = (summed / counts).cpu().numpy()
            vectors.append(mean_pooled)
        return np.vstack(vectors)

    def encode(self, texts, show_progress_bar=True):
        doc_vectors = []
        for idx, text in enumerate(texts):
            if show_progress_bar:
                print(f"  Encoding transcript {idx + 1}/{len(texts)}", end='\r')
            chunks = common.chunk_text(text, CHUNK_WORDS, CHUNK_OVERLAP)
            chunk_vectors = self._embed_chunks(chunks)
            doc_vectors.append(chunk_vectors.mean(axis=0))
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

    print(f"Loaded {len(train_texts)} train transcripts, {len(dev_texts)} dev transcripts.")

    extractor = BertFeatureExtractor(model_name='bert-base-uncased')

    print("Encoding train texts with BERT...")
    train_embeddings = extractor.encode(train_texts)

    print("Encoding dev texts with BERT...")
    dev_embeddings = extractor.encode(dev_texts)

    X_train_dev = np.vstack([train_embeddings, dev_embeddings])
    y_train_dev = np.concatenate([train_labels, dev_labels])

    scaler = StandardScaler()
    X_train_dev_scaled = scaler.fit_transform(X_train_dev)

    common.run_regression_pipeline(
        X_train_dev_scaled,
        y_train_dev,
        common.media_path('bert_best_model_predictions.png'),
        summary_title='Cross-Validation Summary on Combined Train+Dev (BERT)',
        label_fn=lambda name: f"BERT + {name}",
    )


if __name__ == "__main__":
    main()
