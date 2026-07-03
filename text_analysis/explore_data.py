import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'dataset', 'wwwedaic', 'data')
LABELS_DIR = os.path.join(BASE_DIR, 'dataset', 'wwwedaic', 'labels')
OUTPUT_DIR = os.path.join(BASE_DIR, 'media', 'text_exploration')


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def tokenize(text):
    return re.findall(r"\b[a-zA-Z][a-zA-Z']*\b", str(text).lower())


def safe_corr(x, y, method='pearson'):
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]

    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan, np.nan

    if method == 'spearman':
        return spearmanr(x, y)
    return pearsonr(x, y)


def load_split(split_name):
    split_path = os.path.join(LABELS_DIR, f'{split_name}_split.csv')
    df_split = pd.read_csv(split_path)
    rows = []

    for _, row in df_split.iterrows():
        participant_id = int(row['Participant_ID'])
        transcript_path = os.path.join(DATA_DIR, f'{participant_id}_P', f'{participant_id}_Transcript.csv')
        row_data = {
            'split': split_name,
            'participant_id': participant_id,
            'y': float(row['PHQ_Score']),
            'transcript_path': transcript_path,
            'transcript_found': os.path.exists(transcript_path),
            'text': '',
            'utterance_count': 0,
            'missing_text_rows': np.nan,
            'read_error': '',
        }

        if row_data['transcript_found']:
            try:
                df_transcript = pd.read_csv(transcript_path)
                if 'Text' not in df_transcript.columns:
                    row_data['transcript_found'] = False
                    row_data['read_error'] = 'Text column missing'
                else:
                    text_series = df_transcript['Text']
                    text_values = text_series.dropna().astype(str).str.strip()
                    text_values = text_values[text_values != '']
                    row_data['text'] = ' '.join(text_values.str.lower().tolist())
                    row_data['utterance_count'] = len(text_values)
                    row_data['missing_text_rows'] = int(text_series.isna().sum())
            except Exception as exc:
                row_data['transcript_found'] = False
                row_data['read_error'] = str(exc)

        rows.append(row_data)

    return pd.DataFrame(rows)


def add_text_features(df):
    df = df.copy()
    token_lists = df['text'].apply(tokenize)
    stop_words = set(ENGLISH_STOP_WORDS)

    df['char_count'] = df['text'].str.len()
    df['word_count'] = token_lists.apply(len)
    df['unique_word_count'] = token_lists.apply(lambda words: len(set(words)))
    df['stopword_count'] = token_lists.apply(lambda words: sum(word in stop_words for word in words))
    df['non_stopword_count'] = df['word_count'] - df['stopword_count']
    df['avg_word_length'] = token_lists.apply(lambda words: np.mean([len(word) for word in words]) if words else 0)
    df['type_token_ratio'] = np.where(df['word_count'] > 0, df['unique_word_count'] / df['word_count'], 0)
    df['stopword_ratio'] = np.where(df['word_count'] > 0, df['stopword_count'] / df['word_count'], 0)
    df['avg_words_per_utterance'] = np.where(df['utterance_count'] > 0, df['word_count'] / df['utterance_count'], 0)
    df['question_mark_count'] = df['text'].str.count(r'\?')
    df['exclamation_count'] = df['text'].str.count(r'!')
    df['ellipsis_count'] = df['text'].str.count(r'\.\.\.')

    return df


def save_summary_tables(df):
    df.to_csv(os.path.join(OUTPUT_DIR, 'train_dev_text_features.csv'), index=False)

    split_summary = df.groupby('split').agg(
        n=('participant_id', 'count'),
        y_mean=('y', 'mean'),
        y_std=('y', 'std'),
        y_min=('y', 'min'),
        y_median=('y', 'median'),
        y_max=('y', 'max'),
        word_count_mean=('word_count', 'mean'),
        word_count_median=('word_count', 'median'),
        utterance_count_mean=('utterance_count', 'mean'),
        missing_transcripts=('transcript_found', lambda values: int((~values).sum())),
    )
    overall = pd.DataFrame({
        'n': [len(df)],
        'y_mean': [df['y'].mean()],
        'y_std': [df['y'].std()],
        'y_min': [df['y'].min()],
        'y_median': [df['y'].median()],
        'y_max': [df['y'].max()],
        'word_count_mean': [df['word_count'].mean()],
        'word_count_median': [df['word_count'].median()],
        'utterance_count_mean': [df['utterance_count'].mean()],
        'missing_transcripts': [int((~df['transcript_found']).sum())],
    }, index=['train_dev'])
    pd.concat([split_summary, overall]).to_csv(os.path.join(OUTPUT_DIR, 'split_summary.csv'))

    numeric_cols = get_numeric_feature_columns()
    df[numeric_cols].describe().T.to_csv(os.path.join(OUTPUT_DIR, 'numeric_feature_summary.csv'))


def get_numeric_feature_columns():
    return [
        'y', 'char_count', 'word_count', 'unique_word_count', 'utterance_count',
        'avg_word_length', 'type_token_ratio', 'stopword_ratio', 'avg_words_per_utterance',
        'question_mark_count', 'exclamation_count', 'ellipsis_count'
    ]


def plot_y_distribution(df):
    plt.figure(figsize=(12, 5))
    sns.histplot(data=df, x='y', hue='split', bins=15, kde=True, alpha=0.55)
    plt.title('Distribution of PHQ / y Values (Train + Dev)')
    plt.xlabel('PHQ score / y')
    plt.ylabel('Participant count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'y_distribution_histogram.png'), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='split', y='y')
    sns.stripplot(data=df, x='split', y='y', color='black', alpha=0.55)
    plt.title('PHQ / y Values by Split')
    plt.xlabel('Split')
    plt.ylabel('PHQ score / y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'y_distribution_by_split_boxplot.png'), dpi=200)
    plt.close()


def plot_feature_vs_y(df, feature, title, x_label, filename):
    pearson_corr, pearson_p = safe_corr(df[feature], df['y'], 'pearson')
    spearman_corr, spearman_p = safe_corr(df[feature], df['y'], 'spearman')

    plt.figure(figsize=(8, 6))
    sns.regplot(data=df, x=feature, y='y', scatter_kws={'alpha': 0.75}, line_kws={'color': 'red'})
    plt.title(
        f'{title}\nPearson r={pearson_corr:.3f} (p={pearson_p:.3g}), '
        f'Spearman ρ={spearman_corr:.3f} (p={spearman_p:.3g})'
    )
    plt.xlabel(x_label)
    plt.ylabel('PHQ score / y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200)
    plt.close()

    return {
        'feature': feature,
        'pearson_r': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_r': spearman_corr,
        'spearman_p': spearman_p,
    }


def plot_required_correlations(df):
    required_features = [
        ('char_count', 'Transcript Character Length vs PHQ / y', 'Character count', 'text_length_vs_y_correlation.png'),
        ('word_count', 'Transcript Word Count vs PHQ / y', 'Word count', 'word_count_vs_y_correlation.png'),
        ('unique_word_count', 'Unique Words vs PHQ / y', 'Unique word count', 'unique_words_vs_y_correlation.png'),
        ('non_stopword_count', 'Non-Stopwords vs PHQ / y', 'Non-stopword count', 'non_stopwords_vs_y_correlation.png'),
    ]
    results = [plot_feature_vs_y(df, *feature_info) for feature_info in required_features]
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'required_correlation_results.csv'), index=False)


def plot_numeric_feature_correlations(df):
    numeric_cols = get_numeric_feature_columns()
    corr = df[numeric_cols].corr(method='spearman')
    corr.to_csv(os.path.join(OUTPUT_DIR, 'numeric_spearman_correlation_matrix.csv'))

    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
    plt.title('Spearman Correlation Matrix for Transcript Features and y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'numeric_feature_correlation_heatmap.png'), dpi=200)
    plt.close()

    y_corr = corr['y'].drop('y').sort_values()
    plt.figure(figsize=(8, 6))
    colors = ['red' if value < 0 else 'blue' for value in y_corr]
    plt.barh(y_corr.index, y_corr.values, color=colors)
    plt.axvline(0, color='black', linewidth=1)
    plt.title('Transcript Feature Correlations with y (Spearman)')
    plt.xlabel('Spearman correlation with y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'numeric_features_vs_y_correlations.png'), dpi=200)
    plt.close()


def plot_length_distributions(df):
    for feature in ['char_count', 'word_count', 'unique_word_count', 'utterance_count']:
        plt.figure(figsize=(10, 5))
        sns.histplot(data=df, x=feature, hue='split', kde=True, bins=20, alpha=0.55)
        plt.title(f'Distribution of {feature.replace("_", " ").title()}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{feature}_distribution.png'), dpi=200)
        plt.close()


def save_top_terms_by_group(df):
    y_median = df['y'].median()
    df = df.copy()
    df['y_group'] = np.where(df['y'] >= y_median, 'higher_y', 'lower_y')
    stop_words = set(ENGLISH_STOP_WORDS)
    rows = []

    for group_name, group_df in df.groupby('y_group'):
        words = []
        for text in group_df['text']:
            words.extend([word for word in tokenize(text) if word not in stop_words and len(word) > 2])
        for word, count in Counter(words).most_common(50):
            rows.append({'y_group': group_name, 'term': word, 'count': count})

    top_terms = pd.DataFrame(rows)
    top_terms.to_csv(os.path.join(OUTPUT_DIR, 'top_words_by_y_group.csv'), index=False)

    for group_name, group_terms in top_terms.groupby('y_group'):
        plot_df = group_terms.head(25).sort_values('count')
        plt.figure(figsize=(9, 8))
        plt.barh(plot_df['term'], plot_df['count'], color='steelblue')
        plt.title(f'Top Words in {group_name.replace("_", " ").title()} Group')
        plt.xlabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'top_words_{group_name}.png'), dpi=200)
        plt.close()


def analyze_count_term_correlations(df, ngram_range=(1, 1), min_df=2, top_n=30):
    vectorizer = CountVectorizer(
        stop_words='english',
        ngram_range=ngram_range,
        min_df=min_df,
        token_pattern=r"\b[a-zA-Z][a-zA-Z']{2,}\b"
    )
    try:
        matrix = vectorizer.fit_transform(df['text'])
    except ValueError as exc:
        print(f'Skipping count-vectorizer term correlations for {ngram_range}: {exc}')
        return
    feature_names = np.array(vectorizer.get_feature_names_out())
    y = df['y'].to_numpy(dtype=float)
    rows = []

    for idx, term in enumerate(feature_names):
        values = matrix[:, idx].toarray().ravel()
        pearson_corr, pearson_p = safe_corr(values, y, 'pearson')
        spearman_corr, spearman_p = safe_corr(values, y, 'spearman')
        rows.append({
            'term': term,
            'document_frequency': int((values > 0).sum()),
            'total_count': int(values.sum()),
            'pearson_r': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_r': spearman_corr,
            'spearman_p': spearman_p,
        })

    name = 'words' if ngram_range == (1, 1) else 'bigrams'
    correlations = pd.DataFrame(rows).dropna(subset=['spearman_r'])
    correlations.to_csv(os.path.join(OUTPUT_DIR, f'{name}_y_correlations_all.csv'), index=False)
    plot_term_correlations(correlations, name, top_n)


def analyze_tfidf_correlations(df, top_n=30):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.93,
        max_features=5000,
        token_pattern=r"\b[a-zA-Z][a-zA-Z']{2,}\b"
    )
    try:
        matrix = vectorizer.fit_transform(df['text']).toarray()
    except ValueError as exc:
        print(f'Skipping TF-IDF term correlations: {exc}')
        return
    feature_names = np.array(vectorizer.get_feature_names_out())
    y = df['y'].to_numpy(dtype=float)
    rows = []

    for idx, term in enumerate(feature_names):
        values = matrix[:, idx]
        pearson_corr, pearson_p = safe_corr(values, y, 'pearson')
        spearman_corr, spearman_p = safe_corr(values, y, 'spearman')
        rows.append({
            'term': term,
            'document_frequency': int((values > 0).sum()),
            'mean_tfidf': float(values.mean()),
            'pearson_r': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_r': spearman_corr,
            'spearman_p': spearman_p,
        })

    correlations = pd.DataFrame(rows).dropna(subset=['spearman_r'])
    correlations.to_csv(os.path.join(OUTPUT_DIR, 'tfidf_y_correlations_all.csv'), index=False)
    plot_term_correlations(correlations, 'tfidf', top_n)


def plot_term_correlations(correlations, name, top_n=30):
    if correlations.empty:
        return

    strongest = correlations.reindex(correlations['spearman_r'].abs().sort_values(ascending=False).index).head(top_n)
    strongest.to_csv(os.path.join(OUTPUT_DIR, f'{name}_strongest_y_correlations.csv'), index=False)

    plot_df = pd.concat([
        correlations.nsmallest(top_n // 2, 'spearman_r'),
        correlations.nlargest(top_n // 2, 'spearman_r')
    ]).sort_values('spearman_r')

    plt.figure(figsize=(10, 10))
    colors = ['red' if value < 0 else 'blue' for value in plot_df['spearman_r']]
    plt.barh(plot_df['term'], plot_df['spearman_r'], color=colors)
    plt.axvline(0, color='black', linewidth=1)
    plt.title(f'Terms Most Correlated with y ({name}, Spearman)')
    plt.xlabel('Spearman correlation with y')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_strongest_y_correlations.png'), dpi=200)
    plt.close()


def main():
    ensure_output_dir()
    print('Loading train and dev data only...')
    train_df = load_split('train')
    dev_df = load_split('dev')
    df = pd.concat([train_df, dev_df], ignore_index=True)
    df = add_text_features(df)

    print(df['y'].describe())

    print(f'Loaded {len(train_df)} train participants and {len(dev_df)} dev participants.')
    print(f'Saving exploration outputs to {OUTPUT_DIR}')

    save_summary_tables(df)
    plot_y_distribution(df)
    plot_required_correlations(df)
    plot_numeric_feature_correlations(df)
    plot_length_distributions(df)
    save_top_terms_by_group(df)
    analyze_count_term_correlations(df, ngram_range=(1, 1), min_df=2, top_n=30)
    analyze_count_term_correlations(df, ngram_range=(2, 2), min_df=2, top_n=30)
    analyze_tfidf_correlations(df, top_n=30)

    print('Done. Key outputs include:')
    print('- y_distribution_histogram.png')
    print('- text_length_vs_y_correlation.png')
    print('- word_count_vs_y_correlation.png')
    print('- words_y_correlations_all.csv')
    print('- tfidf_strongest_y_correlations.png')


if __name__ == '__main__':
    main()