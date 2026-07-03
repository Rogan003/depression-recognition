import os

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import common


common.suppress_expected_warnings()

common.ensure_media_dir()

CHUNK_WORDS = 150
CHUNK_OVERLAP = 100

NEGATIVE_EMOTION_WORDS = {
    'sad', 'depressed', 'depression', 'anxious', 'anxiety', 'worried', 'worry',
    'lonely', 'alone', 'tired', 'exhausted', 'hopeless', 'helpless', 'cry',
    'crying', 'upset', 'angry', 'afraid', 'fear', 'stress', 'stressed', 'hurt',
    'pain', 'bad', 'worse', 'worst', 'hate', 'guilty', 'ashamed', 'fail',
    'failure', 'empty', 'numb', 'down', 'low', 'struggle', 'struggling',
}


def extract_linguistic_features(text):
    words = text.split()
    if not words:
        return [0, 0]

    word_count = len(words)
    lower_words = [w.lower().strip('.,!?";:()') for w in words]
    unique_word_count = len(set(words))
    neg_emotion_rate = sum(w in NEGATIVE_EMOTION_WORDS for w in lower_words) / word_count

    return [unique_word_count, neg_emotion_rate]


def extract_temporal_features(df_trans):
    num_turns = len(df_trans)
    if num_turns == 0:
        return [0, 0]

    df_trans['duration'] = df_trans['End_Time'] - df_trans['Start_Time']
    total_duration = df_trans['duration'].sum()

    return [num_turns, total_duration]


def load_data_enhanced(split_file):
    df_split = pd.read_csv(split_file)
    data = []

    for _, row in df_split.iterrows():
        p_id = int(row['Participant_ID'])
        phq_score = row['PHQ_Score']

        transcript_path = os.path.join(common.DATA_DIR, f"{p_id}_P", f"{p_id}_Transcript.csv")
        if os.path.exists(transcript_path):
            try:
                df_trans = pd.read_csv(transcript_path)
                if 'Text' in df_trans.columns:
                    text_data = df_trans['Text'].dropna().astype(str).tolist()
                    full_text = " ".join(text_data)

                    ling_features = extract_linguistic_features(full_text)
                    temp_features = extract_temporal_features(df_trans)

                    data.append({
                        'Participant_ID': p_id,
                        'Text': full_text,
                        'PHQ_Score': phq_score,
                        'unique_words': ling_features[0],
                        'neg_emotion_rate': ling_features[1],
                        'num_turns': temp_features[0],
                        'total_duration': temp_features[1]
                    })
            except Exception as e:
                print(f"Error reading {transcript_path}: {e}")

    return pd.DataFrame(data)


def main():
    print("Loading and enhancing data...")
    train_df = load_data_enhanced(os.path.join(common.LABELS_DIR, 'train_split.csv'))
    dev_df = load_data_enhanced(os.path.join(common.LABELS_DIR, 'dev_split.csv'))
    print(f"Loaded {len(train_df)} train, {len(dev_df)} dev samples.")

    full_df = pd.concat([train_df, dev_df], ignore_index=True)

    base_cols = ['unique_words', 'neg_emotion_rate', 'num_turns', 'total_duration']
    X_base = full_df[base_cols].values

    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                                 max_features=100, min_df=3, max_df=0.9)
    X_tfidf = vectorizer.fit_transform(full_df['Text']).toarray()

    print("Encoding text (all-mpnet-base-v2, chunked over full transcript)...")
    embedder = SentenceTransformer('all-mpnet-base-v2')
    X_emb = common.encode_long_texts(embedder, full_df['Text'].tolist(), CHUNK_WORDS, CHUNK_OVERLAP)

    X_hybrid = np.hstack([X_base, X_tfidf, X_emb])
    scaler = StandardScaler()
    X_hybrid_scaled = scaler.fit_transform(X_hybrid)
    y = full_df['PHQ_Score'].values

    print("\nTraining and tuning hybrid models with 5-fold cross-validation...")
    results = common.cross_validate_models(common.default_models(), X_hybrid_scaled, y)

    common.print_cv_summary(results, 'Cross-Validation Summary (Hybrid features, combined Train+Dev)',
                            name_width=16)
    common.print_baseline(y)

    best_result = min(results, key=common.model_score_for_picking)
    print(f"\nBest model: {best_result['name']} (CV MAE {best_result['MAE']:.4f})")
    oof_preds = common.out_of_fold_predictions(best_result['model'], X_hybrid_scaled, y)
    print(f"Out-of-fold R2: {r2_score(y, oof_preds):.4f}")
    common.plot_predictions(
        y, oof_preds, f"Hybrid {best_result['name']}",
        common.media_path('enhanced_best_model_predictions.png'),
        ylabel='PHQ-8 score'
    )

    print("\nLinguistic & Temporal Features Correlation with PHQ-8:")
    for col in base_cols:
        corr = common.pearson_corr(full_df[col].values, full_df['PHQ_Score'].values)
        print(f"  {col}: {corr:.3f}")


if __name__ == "__main__":
    main()
