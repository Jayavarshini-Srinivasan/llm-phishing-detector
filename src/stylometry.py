import numpy as np
import pandas as pd
import textstat
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

from .config import DATA_PATH, TEXT_COLUMN

# Load spaCy model once
_nlp = spacy.load("en_core_web_sm")
_stopwords = set(stopwords.words("english"))


def extract_features_for_text(text: str) -> np.ndarray:
    """
    Returns a fixed-length stylometric feature vector (100 dims placeholder).
    You will expand/refine this later; for now, basic features are enough.
    """
    if not isinstance(text, str):
        text = str(text)

    # Basic cleanup
    text_clean = text.strip()
    if not text_clean:
        return np.zeros(100, dtype=np.float32)

    # Token and sentence stats
    sentences = sent_tokenize(text_clean)
    words = word_tokenize(text_clean)

    num_sent = len(sentences)
    num_words = len(words)
    num_chars = len(text_clean)

    avg_sent_len = num_words / num_sent if num_sent > 0 else 0.0
    avg_word_len = num_chars / num_words if num_words > 0 else 0.0

    # Stopword ratio
    stop_count = sum(1 for w in words if w.lower() in _stopwords)
    stop_ratio = stop_count / num_words if num_words > 0 else 0.0

    # Capital letters and punctuation
    num_caps = sum(1 for c in text_clean if c.isupper())
    cap_ratio = num_caps / num_chars if num_chars > 0 else 0.0

    num_excl = text_clean.count("!")
    num_q = text_clean.count("?")
    num_colon = text_clean.count(":")

    excl_ratio = num_excl / num_chars if num_chars > 0 else 0.0
    q_ratio = num_q / num_chars if num_chars > 0 else 0.0
    colon_ratio = num_colon / num_chars if num_chars > 0 else 0.0

    # Readability metrics
    try:
        flesch = textstat.flesch_reading_ease(text_clean)
    except Exception:
        flesch = 0.0

    # Type-token ratio
    words_lower = [w.lower() for w in words if w.isalpha()]
    vocab = set(words_lower)
    ttr = len(vocab) / len(words_lower) if words_lower else 0.0

    # Character entropy
    char_counts = Counter(text_clean)
    total_chars = sum(char_counts.values())
    entropy = 0.0
    if total_chars > 0:
        for c, cnt in char_counts.items():
            p = cnt / total_chars
            entropy -= p * np.log2(p)

    # POS tag distribution via spaCy
    doc = _nlp(text_clean)
    pos_counts = Counter([token.pos_ for token in doc])
    total_pos = sum(pos_counts.values()) or 1
    # Take some common POS tags in fixed order
    pos_tags = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "DET", "NUM", "PROPN", "PUNCT"]
    pos_freqs = [pos_counts[tag] / total_pos for tag in pos_tags]

    # Assemble into vector
    feats = [
        num_sent,
        num_words,
        num_chars,
        avg_sent_len,
        avg_word_len,
        stop_ratio,
        cap_ratio,
        excl_ratio,
        q_ratio,
        colon_ratio,
        flesch,
        ttr,
        entropy,
    ] + pos_freqs

    # Pad/truncate to exactly 100 dims
    if len(feats) < 100:
        feats.extend([0.0] * (100 - len(feats)))
    else:
        feats = feats[:100]

    return np.array(feats, dtype=np.float32)


def build_and_save_stylometry_features(output_path: str = "data/stylometry_features.npy") -> None:
    df = pd.read_csv(DATA_PATH)
    all_feats = []

    for _, row in df.iterrows():
        text = row[TEXT_COLUMN]
        vec = extract_features_for_text(text)
        all_feats.append(vec)

    arr = np.stack(all_feats, axis=0)
    np.save(output_path, arr)
    print(f"Saved stylometric features of shape {arr.shape} to {output_path}")
