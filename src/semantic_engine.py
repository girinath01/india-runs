"""
Semantic Engine v1.0 — Local embedding-based semantic understanding.

Uses sentence-transformers (all-MiniLM-L6-v2) to compute cosine similarity
between candidate text and JD target phrases. This replaces pure keyword
matching with real semantic understanding.

Key capabilities:
  1. Understand synonyms (e.g., "document fetching" ≈ "information retrieval")
  2. Detect negation ("I have NO experience with search" → low score)
  3. Score contextual depth (longer, richer descriptions → higher similarity)
"""

import re
import numpy as np
from functools import lru_cache

# ─────────────────────────────────────────────────────────────────────────────
# LAZY MODEL LOADING — only loads when first called
# ─────────────────────────────────────────────────────────────────────────────

_model = None
_target_embeddings = {}

def _get_model():
    """Lazy-load the sentence-transformers model (singleton)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[Semantic Engine] Model loaded: all-MiniLM-L6-v2")
        except ImportError:
            print("[Semantic Engine] WARNING: sentence-transformers not installed. "
                  "Falling back to regex-only scoring.")
            _model = False  # sentinel: tried and failed
    return _model if _model is not False else None


# ─────────────────────────────────────────────────────────────────────────────
# JD TARGET PHRASES — what the ideal candidate's text would say
# ─────────────────────────────────────────────────────────────────────────────

JD_TARGET_PHRASES = {
    "search": (
        "Built and deployed large-scale search, retrieval, and ranking systems "
        "serving millions of queries. Designed semantic search and hybrid retrieval "
        "pipelines combining BM25 with dense vector embeddings. Optimized search "
        "relevance, query understanding, and indexing infrastructure."
    ),
    "recommendation": (
        "Designed and shipped recommendation and personalization systems with "
        "collaborative filtering and real-time ranking models. Built content ranking, "
        "feed ranking, and product recommendation engines at scale. Implemented "
        "recommendation pipelines serving millions of users."
    ),
    "production": (
        "Shipped machine learning models to production at scale, managing latency, "
        "throughput, and SLA requirements. Deployed models with real-time inference, "
        "A/B testing, and monitoring. Handled production traffic, model serving, "
        "and inference optimization for millions of requests."
    ),
    "ownership": (
        "Architected, led, and owned end-to-end machine learning systems from design "
        "to deployment independently. Drove technical strategy, made key architecture "
        "decisions, and mentored team members. Founded and established new ML "
        "capabilities from zero to one."
    ),
    "impact": (
        "Improved key business metrics like click-through rate, conversion rate, "
        "revenue, and latency by measurable amounts. Reduced costs, improved "
        "precision and recall, scaled systems to millions of users. Demonstrated "
        "10x improvements in throughput and P99 latency reductions."
    ),
    "marketplace": (
        "Built two-sided marketplace matching and talent acquisition platforms with "
        "supply-demand optimization. Designed job matching, candidate ranking, and "
        "hiring platform algorithms. Created HR tech and recruiting technology "
        "for talent marketplace products."
    ),
    "evaluation": (
        "Designed evaluation frameworks using NDCG, MRR, MAP, offline evaluation, "
        "and A/B testing for ranking and retrieval systems. Built offline-to-online "
        "correlation pipelines. Implemented automated evaluation metrics and "
        "experiment analysis for search quality measurement."
    ),
    "disqualifying": (
        "Computer vision, image classification, object detection, YOLO, robotics, "
        "autonomous driving, speech recognition, audio processing, ROS, LIDAR, SLAM, "
        "image segmentation, OpenCV, convolutional neural networks for vision tasks."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# NEGATION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_NEGATION_PATTERNS = re.compile(
    r'\b('
    r'no experience|not experienced|never worked|'
    r'have not|haven\'t|did not|didn\'t|'
    r'without experience|lack of experience|'
    r'no background|not familiar|unfamiliar|'
    r'outside of my|not related to|unrelated|'
    r'no exposure|limited exposure to|'
    r'do not have|don\'t have|'
    r'not involved in|never built|never deployed|'
    r'seeking to learn|want to learn|aspiring|'
    r'transitioning from|career change|switching to|'
    r'no knowledge of|basic understanding only'
    r')\b',
    re.IGNORECASE
)

_WEAK_CONTEXT_PATTERNS = re.compile(
    r'\b('
    r'heard about|read about|studied|coursework|'
    r'personal project|side project|hobby|'
    r'tutorial|online course|certification only|'
    r'theoretical|conceptual understanding|'
    r'exposed to|touched upon|briefly'
    r')\b',
    re.IGNORECASE
)


def detect_negation(sentence: str) -> bool:
    """Check if a sentence contains negation context around domain terms."""
    return bool(_NEGATION_PATTERNS.search(sentence))


def detect_weak_context(sentence: str) -> bool:
    """Check if a sentence only describes weak/surface-level experience."""
    return bool(_WEAK_CONTEXT_PATTERNS.search(sentence))


# ─────────────────────────────────────────────────────────────────────────────
# CORE SIMILARITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _get_target_embedding(domain: str):
    """Get (or compute+cache) the embedding for a JD target phrase."""
    global _target_embeddings
    if domain not in _target_embeddings:
        model = _get_model()
        if model is None:
            return None
        phrase = JD_TARGET_PHRASES.get(domain)
        if phrase is None:
            return None
        _target_embeddings[domain] = model.encode(phrase, normalize_embeddings=True)
    return _target_embeddings[domain]


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two normalized vectors."""
    return float(np.dot(a, b))


def compute_semantic_scores(text: str) -> dict[str, float]:
    """
    Compute semantic similarity scores for ALL domains at once.

    Args:
        text: Full candidate career text (headline + summary + descriptions).

    Returns:
        Dict mapping domain name → similarity score (0.0 to 1.0).
        Returns empty dict if model is not available.
    """
    model = _get_model()
    if model is None:
        return {}

    if not text or len(text.strip()) < 20:
        return {domain: 0.0 for domain in JD_TARGET_PHRASES}

    # Split text into sentences for granular analysis
    sentences = _split_into_chunks(text)
    if not sentences:
        return {domain: 0.0 for domain in JD_TARGET_PHRASES}

    # Batch encode all candidate sentences at once (efficient)
    sentence_embeddings = model.encode(sentences, normalize_embeddings=True,
                                        batch_size=32, show_progress_bar=False)

    results = {}
    for domain in JD_TARGET_PHRASES:
        target_emb = _get_target_embedding(domain)
        if target_emb is None:
            results[domain] = 0.0
            continue

        # Compute similarity for each sentence against this domain's target
        similarities = []
        for i, sent_emb in enumerate(sentence_embeddings):
            sim = _cosine_similarity(sent_emb, target_emb)

            # Apply negation penalty: if a sentence talks about the domain
            # but in a negative way, reduce its contribution
            if sim > 0.3:  # only check sentences that seem relevant
                if detect_negation(sentences[i]):
                    sim *= 0.1  # almost zero out negated mentions
                elif detect_weak_context(sentences[i]):
                    sim *= 0.4  # reduce weight for weak/surface context

            similarities.append(sim)

        if not similarities:
            results[domain] = 0.0
            continue

        # Score = weighted combination of top-K sentence similarities
        # This rewards depth: more relevant sentences → higher score
        sorted_sims = sorted(similarities, reverse=True)
        top_k = sorted_sims[:5]  # top 5 most relevant sentences

        # Weighted: best sentence counts most, diminishing returns
        weights = [1.0, 0.6, 0.4, 0.25, 0.15]
        weighted_sum = sum(s * w for s, w in zip(top_k, weights))
        max_weighted = sum(weights[:len(top_k)])

        results[domain] = min(1.0, weighted_sum / max_weighted) if max_weighted > 0 else 0.0

    return results


def compute_sentence_semantic_scores(sentences: list[str], domain: str) -> list[tuple[str, float]]:
    """
    Score individual sentences against a specific domain.
    Returns list of (sentence, score) sorted by score descending.
    Useful for evidence extraction with semantic ranking.
    """
    model = _get_model()
    if model is None:
        return [(s, 0.0) for s in sentences]

    target_emb = _get_target_embedding(domain)
    if target_emb is None:
        return [(s, 0.0) for s in sentences]

    if not sentences:
        return []

    embeddings = model.encode(sentences, normalize_embeddings=True,
                               batch_size=32, show_progress_bar=False)

    scored = []
    for sent, emb in zip(sentences, embeddings):
        sim = _cosine_similarity(emb, target_emb)
        if detect_negation(sent):
            sim *= 0.1
        elif detect_weak_context(sent):
            sim *= 0.4
        scored.append((sent, float(sim)))

    scored.sort(key=lambda x: -x[1])
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _split_into_chunks(text: str, max_chunk_len: int = 256) -> list[str]:
    """
    Split text into sentence-like chunks suitable for embedding.
    Keeps chunks under max_chunk_len characters to stay within model limits.
    """
    # Split on sentence boundaries
    raw_parts = re.split(r'[.;!\n]+', text)
    chunks = []
    for part in raw_parts:
        part = part.strip()
        if len(part) < 15:
            continue
        # If a chunk is too long, split it further on commas
        if len(part) > max_chunk_len:
            sub_parts = part.split(",")
            current = ""
            for sp in sub_parts:
                if len(current) + len(sp) < max_chunk_len:
                    current += sp + ","
                else:
                    if len(current.strip()) >= 15:
                        chunks.append(current.strip().rstrip(","))
                    current = sp + ","
            if len(current.strip()) >= 15:
                chunks.append(current.strip().rstrip(","))
        else:
            chunks.append(part)
    return chunks


def is_model_available() -> bool:
    """Check if the semantic model can be loaded."""
    return _get_model() is not None
