"""Aspect-based perceived quietness from guest reviews.

Only sentences that mention a noise-related term contribute, so the score
measures the acoustic aspect rather than overall review polarity. Values are
in [-1, 1] with positive meaning quieter.
"""

import re

import numpy as np
import pandas as pd

ASPECT_TERMS = [
    "quiet", "quietness", "peaceful", "calm", "tranquil", "serene", "silent",
    "silence", "soundproof", "restful", "peace", "noise", "noisy", "loud",
    "traffic", "siren", "sirens", "construction", "party", "parties", "music",
    "thin wall", "thin walls", "honk", "train", "airplane", "highway",
    "street noise", "barking", "could hear", "kept us up", "kept me up",
    "couldn't sleep", "cant sleep", "woke", "disturb",
]

POSITIVE = {
    "quiet", "quietness", "peaceful", "calm", "tranquil", "serene",
    "silent", "silence", "soundproof", "restful", "peace",
}

NEGATIVE = {
    "noisy", "loud", "noise", "traffic", "siren", "sirens", "construction",
    "party", "parties", "honk", "barking", "woke", "disturb", "disturbed",
    "disturbing",
}

NEGATIVE_PHRASES = [
    "could hear", "kept us up", "kept me up", "couldn't sleep",
    "cant sleep", "thin wall", "thin walls", "street noise", "too loud",
]

NEGATORS = {
    "not", "no", "never", "wasn't", "wasnt", "isn't", "isnt",
    "didn't", "didnt", "hardly", "barely",
}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\|\|\|")
TOKEN = re.compile(r"[a-z']+")

NEGATION_WINDOW = 2
SHRINK_K = 3.0  # empirical-Bayes pseudo-count toward the cell mean


def score_sentence(sentence):
    """Polarity of one sentence, or None if it carries no noise aspect."""
    s = sentence.lower()
    if not any(term in s for term in ASPECT_TERMS):
        return None

    tokens = TOKEN.findall(s)
    score = 0
    for i, word in enumerate(tokens):
        value = 1 if word in POSITIVE else (-1 if word in NEGATIVE else 0)
        if not value:
            continue
        if any(t in NEGATORS for t in tokens[max(0, i - NEGATION_WINDOW):i]):
            value = -value
        score += value

    for phrase in NEGATIVE_PHRASES:
        if phrase in s:
            score -= 1

    return float(np.tanh(score)) if score else None


def score_review(text):
    """Mean polarity over aspect-bearing sentences and how many there were."""
    if not isinstance(text, str) or not text.strip():
        return np.nan, 0
    values = [score_sentence(s) for s in SENTENCE_SPLIT.split(text)]
    values = [v for v in values if v is not None]
    if not values:
        return np.nan, 0
    return float(np.mean(values)), len(values)


def score_reviews(reviews, text_col="comments"):
    """Add polarity and aspect-sentence count columns to a review table."""
    scored = reviews.copy()
    pairs = scored[text_col].apply(score_review)
    scored["q"] = [p[0] for p in pairs]
    scored["q_n"] = [p[1] for p in pairs]
    return scored


def per_listing(scored, listing_to_cell, id_col="listing_id", k=SHRINK_K):
    """Listing-level quietness, shrunk toward the H3 cell mean.

    Listings with few aspect mentions borrow strength from their neighbourhood;
    k is the pseudo-count controlling how much.
    """
    signal = scored.dropna(subset=["q"]).copy()
    signal["h3"] = signal[id_col].map(listing_to_cell)
    cell_mean = signal.dropna(subset=["h3"]).groupby("h3")["q"].mean()

    out = signal.groupby(id_col).agg(q_raw=("q", "mean"), n_aspect=("q", "size"))
    out["h3"] = out.index.map(listing_to_cell)
    prior = out["h3"].map(cell_mean).fillna(out["q_raw"])
    out["quietness"] = ((out["n_aspect"] * out["q_raw"] + k * prior)
                        / (out["n_aspect"] + k)).round(4)
    return out.reset_index().rename(columns={id_col: "id"})


def per_cell(scored, listing_to_cell, id_col="listing_id"):
    """Cell-level field, weighting each review by its aspect-sentence count."""
    scored = scored.copy()
    scored["h3"] = scored[id_col].map(listing_to_cell)
    signal = scored.dropna(subset=["q", "h3"]).copy()
    signal["w"] = signal["q_n"].clip(lower=1)
    signal["qw"] = signal["q"] * signal["w"]

    agg = signal.groupby("h3").agg(
        qw=("qw", "sum"),
        w=("w", "sum"),
        n_reviews_aspect=("q", "size"),
        n_sentences_aspect=("q_n", "sum"),
    ).reset_index()
    agg["quietness"] = (agg["qw"] / agg["w"]).round(4)

    total = scored.dropna(subset=["h3"]).groupby("h3").size().rename("n_reviews_total")
    cell = agg.merge(total, on="h3", how="left")
    cell["aspect_coverage"] = (cell["n_reviews_aspect"] / cell["n_reviews_total"]).round(3)
    return cell[["h3", "quietness", "aspect_coverage", "n_reviews_aspect",
                 "n_sentences_aspect", "n_reviews_total"]]


def listing_cell_index(listings, res, id_col="id", lat_col="latitude", lon_col="longitude"):
    """Map listing id -> H3 cell at the given resolution."""
    import h3

    df = listings.dropna(subset=[lat_col, lon_col]).copy()
    df[id_col] = df[id_col].astype(str)
    cells = [h3.latlng_to_cell(a, o, res) for a, o in zip(df[lat_col], df[lon_col])]
    return pd.Series(cells, index=df[id_col]).to_dict()
