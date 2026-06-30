"""
Semantic Engine v2.0 (ONNX Accelerated) — Local embedding-based semantic understanding.

Uses highly optimized ONNX Runtime to compute cosine similarity between candidate text 
and JD target phrases without needing GPU.

Key capabilities:
  1. Understand synonyms
  2. Detect negation
  3. Score contextual depth
  4. Runs 2x-3x faster on CPU via ONNX graph execution.
"""

import os
import re
import numpy as np
import torch
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# LAZY MODEL LOADING (ONNX)
# ─────────────────────────────────────────────────────────────────────────────

_model = None
_tokenizer = None
_target_embeddings = {}

# We look for the local ONNX model directory
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "local_onnx_model")

def _get_model_and_tokenizer():
    """Lazy-load the ONNX model and tokenizer."""
    global _model, _tokenizer
    if _model is None:
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            
            model_path = os.path.join(_MODEL_DIR, "model.onnx")
            if not os.path.exists(model_path):
                print(f"[Semantic Engine] WARNING: Local ONNX model not found at {model_path}.")
                _model = False
                return None, None
                
            _tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR)
            
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 4
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            _model = ort.InferenceSession(model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
            print(f"[Semantic Engine] ONNX Model loaded from {_MODEL_DIR}")
        except ImportError as e:
            print(f"[Semantic Engine] WARNING: Failed to load ONNX: {e}")
            _model = False
    return (_model if _model is not False else None), _tokenizer

def _mean_pooling(model_output, attention_mask):
    """Mean Pooling - Takes attention mask into account for correct averaging."""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def _encode(sentences: list[str]) -> np.ndarray:
    """Tokenize, infer via ONNX, pool, and normalize."""
    model, tokenizer = _get_model_and_tokenizer()
    if not model:
        return np.zeros((len(sentences), 384))
        
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='np')
    
    inputs = {
        "input_ids": encoded_input["input_ids"].astype(np.int64),
        "attention_mask": encoded_input["attention_mask"].astype(np.int64),
        "token_type_ids": encoded_input["token_type_ids"].astype(np.int64)
    }
    
    outputs = model.run(None, inputs)
    token_embeddings = torch.tensor(outputs[0])
    attention_mask = torch.tensor(encoded_input['attention_mask'])
        
    sentence_embeddings = _mean_pooling([token_embeddings], attention_mask)
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.numpy()

# ─────────────────────────────────────────────────────────────────────────────
# JD TARGET PHRASES
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
    return bool(_NEGATION_PATTERNS.search(sentence))


def detect_weak_context(sentence: str) -> bool:
    return bool(_WEAK_CONTEXT_PATTERNS.search(sentence))


# ─────────────────────────────────────────────────────────────────────────────
# CORE SIMILARITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _get_target_embedding(domain: str):
    """Get (or compute+cache) the embedding for a JD target phrase."""
    global _target_embeddings
    if domain not in _target_embeddings:
        phrase = JD_TARGET_PHRASES.get(domain)
        if phrase is None:
            return None
        _target_embeddings[domain] = _encode([phrase])[0]
    return _target_embeddings[domain]


def _cosine_similarity(a, b) -> float:
    return float(np.dot(a, b))


def compute_semantic_scores(text: str) -> dict[str, float]:
    """Compute semantic similarity scores for ALL domains at once."""
    if _get_model_and_tokenizer()[0] is None:
        return {}

    if not text or len(text.strip()) < 20:
        return {domain: 0.0 for domain in JD_TARGET_PHRASES}

    # Split text into sentences for granular analysis
    sentences = _split_into_chunks(text)
    if not sentences:
        return {domain: 0.0 for domain in JD_TARGET_PHRASES}

    # Encode all sentences efficiently in ONNX
    sentence_embeddings = _encode(sentences)

    results = {}
    for domain in JD_TARGET_PHRASES:
        target_emb = _get_target_embedding(domain)
        if target_emb is None:
            results[domain] = 0.0
            continue

        similarities = []
        for i, sent_emb in enumerate(sentence_embeddings):
            sim = _cosine_similarity(sent_emb, target_emb)
            if sim > 0.3:
                if detect_negation(sentences[i]):
                    sim *= 0.1
                elif detect_weak_context(sentences[i]):
                    sim *= 0.4
            similarities.append(sim)

        if not similarities:
            results[domain] = 0.0
            continue

        sorted_sims = sorted(similarities, reverse=True)
        top_k = sorted_sims[:5]

        weights = [1.0, 0.6, 0.4, 0.25, 0.15]
        weighted_sum = sum(s * w for s, w in zip(top_k, weights))
        max_weighted = sum(weights[:len(top_k)])

        results[domain] = min(1.0, weighted_sum / max_weighted) if max_weighted > 0 else 0.0

    return results


def compute_sentence_semantic_scores(sentences: list[str], domain: str) -> list[tuple[str, float]]:
    """Score individual sentences against a specific domain."""
    if _get_model_and_tokenizer()[0] is None or not sentences:
        return [(s, 0.0) for s in sentences]

    target_emb = _get_target_embedding(domain)
    if target_emb is None:
        return [(s, 0.0) for s in sentences]

    embeddings = _encode(sentences)

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
    raw_parts = re.split(r'[.;!\n]+', text)
    chunks = []
    for part in raw_parts:
        part = part.strip()
        if len(part) < 15:
            continue
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
    return _get_model_and_tokenizer()[0] is not None
