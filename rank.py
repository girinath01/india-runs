#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redrob Hackathon - Candidate Ranker v6.0  (Bug Hunters)
JD: Senior AI Engineer -- Founding Team @ Redrob AI

Architecture (v6.0 — Two-Stage Scored Pipeline):

  Stage 1  — JSONL Streaming Reader   (iter_candidates)
  Stage 2  — Candidate Normalizer     (normalize → Candidate)
  Stage 3  — Feature Extraction        (extract_features → FeatureVector)
             ├── CareerAnalyzer        (0–25 pts)
             ├── JDIntentAnalyzer      (0–20 pts, semantic buckets)
             ├── ProductionAnalyzer    (0–18 pts)
             ├── OwnershipAnalyzer     (0–12 pts, OWNER/LEAD/CONTRIBUTOR/SUPPORT)
             ├── ImpactAnalyzer        (0–10 pts, Impact objects)
             ├── EvalAnalyzer          (0–8  pts)
             ├── CompanyAnalyzer       (0–6  pts, typed profile)
             ├── TrajectoryAnalyzer    (0–6  pts)
             ├── SkillAnalyzer         (0–5  pts)
             ├── EvidenceAnalyzer      (Evidence objects, provenance sentences)
             └── HiringReadiness       (-3 to +8 pts)
  Stage 4  — Dedicated Penalty Stage   (0–50 pts subtracted)
  Stage 5  — Two-Stage Scorer          (technical×0.90 + hiring×0.10 + synergy − penalty)
  Stage 6  — Top-K Priority Queue      (Pass 1 fast filter, heapq, O(N log K))
  Stage 7  — Evidence-Driven Reasoning  (top-3 evidence → JD mapping → biggest gap)
  Output   — submission.csv

Score Budget (v6.0 — integer ranges):
  ┌─ Technical Fit ──────────────────────────────┐
  │  Career                 0–25                 │
  │  JD Intent              0–20  (semantic)     │
  │  Production             0–18  (shipper)      │
  │  Ownership              0–12  (verb-class)   │
  │  Impact                 0–10  (metrics)      │
  │  Evaluation             0–8   (NDCG/A-B)     │
  │  Company                0–6   (type profile)  │
  │  Trajectory             0–6   (progression)  │
  │  Skills                 0–5   (supporting)   │
  │  Synergy bonuses        0–28  (combinations) │
  │  ────────────────────────────────────────     │
  │  Technical Max         ~138                   │
  └──────────────────────────────────────────────┘
  ┌─ Hiring Readiness ──────────────────────────┐
  │  Notice period         -1 to +2              │
  │  Activity               0 to +3              │
  │  Recruiter response     0 to +2              │
  │  Relocation            -1 to +1              │
  │  Hiring Max             +8                   │
  └──────────────────────────────────────────────┘
  ┌─ Penalties ─────────────────────────────────┐
  │  Dedicated stage        0–50  (subtracted)  │
  └──────────────────────────────────────────────┘

  final_score = technical × 0.90 + hiring × 0.10 − penalties

  → Notice period can NEVER dominate technical merit.
  → Synergy bonuses reward JD-valued combinations.
  → Evidence objects provide provenance for reasoning.

Design principles (carried from v5):
  1. Retrieval/Search/Ranking/VectorDB >> trendy LLM tooling  (JD warns)
  2. Shippers > Researchers — production deployment is first-class
  3. Technical fit first; availability is secondary (v6 fix)
  4. Two-pass: 100K → fast filter (top 12000) → deep score → 100
  5. No external deps — pure stdlib → zero import errors
  6. Evidence objects with provenance drive both scoring AND reasoning

PRESERVED from v4.1/v5.0:
  [B1]–[B10] All original bug fixes preserved
  [B2]  Streaming iterator — never loads full 465 MB
  [B3]  len(name) >= 3 guard
  [B7]  Multi-format date parsing
  [B8]  JSON-array branch
  [B9]  NON_TECH_TITLES frozenset
  [B10] Score clamp

Usage:
  python rank.py --candidates ./candidates.jsonl.gz --out ./bug_hunters.csv
  python rank.py --candidates ./sample_candidates.json --out ./test_out.csv
"""

import json
import sys
import math
import re
import csv
import gzip
import heapq
import hashlib
import argparse
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# TODAY — fixed for reproducible ranking in graded sandbox
# ─────────────────────────────────────────────────────────────────────────────
TODAY = date(2026, 6, 15)


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def candidate_id_num(candidate_id: str) -> int:
    match = re.search(r"\d+$", candidate_id or "")
    return int(match.group(0)) if match else 0


# ─────────────────────────────────────────────────────────────────────────────
# TYPED DATACLASSES  (v6.0 — integer-range scoring + Evidence objects)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Evidence:
    """A single piece of extracted evidence with provenance."""
    category: str      # "retrieval", "production", "ownership", "impact", etc.
    sentence: str      # actual text snippet that produced this evidence
    priority: int = 5  # higher = more JD-relevant (used for reasoning sort)


@dataclass
class Impact:
    """A single measurable outcome extracted from the profile."""
    metric: str        # "CTR", "Latency", "Scale", "Revenue", etc.
    improvement: str   # "12%", "40% reduction", "8M users", etc.


@dataclass
class Candidate:
    """Normalised candidate — hides all JSON schema complexity from the pipeline."""
    id: str
    headline: str
    summary: str
    current_title: str
    current_company: str
    location: str
    country: str
    years_of_experience: float
    career: list       # raw career_history list
    skills: list       # raw skills list
    signals: dict      # raw redrob_signals dict


@dataclass
class CareerFeatures:
    """Output of CareerAnalyzer. Integer range 0–25."""
    score: int                       # 0–25
    years_exp: float
    actual_yoe: float
    career_depth: float              # profile richness (absorbs ConfidenceAnalyzer)
    career_consistency: float        # 0–1  (1 = stable, 0 = job-hopper)
    specialization: str              # "RETRIEVAL", "ML", "BACKEND", "DATA", "GENERALIST"
    promotion_count: int
    product_years: float
    startup_years: float
    service_years: float
    ml_years: float
    evidence: list = field(default_factory=list)


@dataclass
class CompanyFeatures:
    """Output of CompanyAnalyzer. Returns profile + score 0–6."""
    score: int                       # 0–6
    company_type: str                # "MARKETPLACE", "SEARCH", "PRODUCT", "STARTUP", "CONSULTING", "OTHER"
    search_exposure: bool
    ranking_exposure: bool
    startup: bool
    elite_company: bool
    consult_fraction: float
    founder_mindset: bool
    evidence: list = field(default_factory=list)


@dataclass
class SkillFeatures:
    """Output of SkillAnalyzer. Supporting evidence only, 0-3. JD: skills confirm experience, don't substitute."""
    score: int                       # 0-3 (reduced to prevent keyword overweight)
    tier1_count: int
    tier2_count: int
    disq_fraction: float
    evidence: list = field(default_factory=list)


@dataclass
class EvidenceFeatures:
    """Output of EvidenceAnalyzer. Extracts provenance Evidence objects."""
    retrieval: bool
    recommendation: bool
    ranking: bool
    search_relevance: bool
    marketplace: bool
    production_deployed: bool
    evidence: list = field(default_factory=list)


@dataclass
class OwnershipFeatures:
    """Output of OwnershipAnalyzer. OWNER/LEAD/CONTRIBUTOR/SUPPORT classification."""
    score: int                       # 0–12
    level: str                       # "OWNER", "LEAD", "CONTRIBUTOR", "SUPPORT", "UNKNOWN"
    evidence: list = field(default_factory=list)


@dataclass
class ImpactFeatures:
    """Output of ImpactAnalyzer. Extracts Impact objects with metric+improvement."""
    score: int                       # 0–10
    impacts: list = field(default_factory=list)     # list of Impact objects
    evidence: list = field(default_factory=list)


@dataclass
class ProductionFeatures:
    """Output of ProductionAnalyzer. 0–18, evidence of deployed systems."""
    score: int                       # 0–18
    ship_count: int                  # number of shipping signals found
    research_only: bool
    evidence: list = field(default_factory=list)


@dataclass
class JDIntentFeatures:
    """Output of JDIntentAnalyzer. Semantic bucket detection, 0–20."""
    score: int                       # 0–20
    recommendation_hit: bool
    search_hit: bool
    marketplace_hit: bool
    evaluation_hit: bool
    vector_hit: bool
    hybrid_bonus: int
    ltr_bonus: int
    evidence: list = field(default_factory=list)


@dataclass
class EvalFeatures:
    """Output of EvalAnalyzer. 0–8, evaluation methodology."""
    score: int                       # 0–8
    has_eval: bool
    eval_methods: list = field(default_factory=list)
    evidence: list = field(default_factory=list)


@dataclass
class TrajectoryFeatures:
    """Output of TrajectoryAnalyzer. Career progression, 0–6."""
    score: int                       # 0–6
    penalty: int
    reward: int


@dataclass
class DomainTenure:
    """Output of DomainTenureAnalyzer. 0-10.
    Measures YEARS spent specifically in retrieval/search/recommendation.
    JD requires: 3+ years in this domain. Directly scored against that.
    """
    score: int           # 0-10
    domain_months: int   # raw months in retrieval/search/recommendation
    domain_years: float  # domain_months / 12.0


@dataclass
class HiringReadiness:
    """Output of HiringReadinessAnalyzer v6.1. -3 to +8. Uses all 21 Redrob signals."""
    score: int                       # -3 to +8  (final clamped total)
    # Group 1: Availability
    notice_bonus: int                # -1 to +2
    otw_pts: int                     # 0 to +1
    work_mode_pts: int               # -1 to 0
    relocation: int                  # -1 to +1
    # Group 2: Engagement
    activity_score: int              # -1 to +2
    recruiter_response: int          # -1 to +1
    response_time_pts: int           # 0 to +1
    applications_pts: int            # 0 to +1
    # Group 3: Trust & Market Demand
    interview_pts: int               # -1 to +1
    offer_pts: int                   # -1 to +1
    saved_pts: int                   # 0 to +1
    views_pts: int                   # 0 to +1
    completeness_pts: int            # -1 to +1
    trust_pts: int                   # -1 to +1
    # Group 4: Skill Validation
    assessment_pts: int              # 0 to +2
    github_pts: int                  # 0 to +1
    endorsement_pts: int             # 0 to +1
    search_pts: int                  # 0 to +1
    # Legacy fields (used in generate_reasoning)
    open_to_work: bool


@dataclass
class FeatureVector:
    """Typed aggregate. Scorer depends ONLY on this, not raw JSON."""
    career:     CareerFeatures
    company:    CompanyFeatures
    skills:     SkillFeatures
    evidence:   EvidenceFeatures
    ownership:  OwnershipFeatures
    impact:     ImpactFeatures
    production: ProductionFeatures
    jd_intent:  JDIntentFeatures
    evaluation: EvalFeatures
    trajectory:    TrajectoryFeatures
    hiring:        HiringReadiness
    domain_tenure: DomainTenure
    all_evidence: list = field(default_factory=list)  # merged evidence for reasoning


# ─────────────────────────────────────────────────────────────────────────────
# TIERED SKILL DEFINITIONS  (from JD analysis) — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

TIER1_SKILLS = frozenset({
    "retrieval", "information retrieval", "dense retrieval", "sparse retrieval",
    "hybrid retrieval", "hybrid search", "semantic search", "vector search",
    "search engine", "search ranking", "search system", "search infrastructure",
    "recommendation system", "recommender", "recommender system", "ranking system",
    "learning to rank", "ltr", "lambdamart", "lambdarank", "ranknet",
    "bi encoder", "cross encoder",
    "faiss", "elasticsearch", "opensearch", "qdrant", "milvus",
    "weaviate", "pinecone", "pgvector", "vespa", "annoy", "scann", "chroma",
    "vector database", "vector db", "vector store", "vector index",
    "ann", "approximate nearest neighbor", "hnsw",
    "reranking", "re ranking", "reranker", "colbert",
    "bm25", "tfidf", "tf idf",
    "ndcg", "mrr", "mean reciprocal rank", "map", "mean average precision",
    "a b testing", "ab testing", "offline evaluation", "online evaluation",
    "eval framework", "evaluation framework",
})

TIER2_SKILLS = frozenset({
    "rag", "retrieval augmented generation",
    "embeddings", "embedding", "sentence transformers", "sentence transformer",
    "text embedding", "openai embeddings", "bge", "e5 model", "dense embedding",
    "nlp", "natural language processing", "language model", "text classification",
    "bert", "transformers", "huggingface", "hugging face",
    "pytorch", "tensorflow", "keras", "jax",
    "mlops", "model deployment", "model serving", "inference",
    "feature engineering", "feature store",
    "distributed systems", "distributed training",
    "mlflow", "wandb", "ray", "dask",
    "xgboost", "lightgbm", "gradient boosting",
    "scikit learn", "sklearn",
    "python", "fastapi", "flask", "docker", "kubernetes",
    "spark", "kafka", "airflow",
    "aws", "gcp", "azure",
})

TIER3_SKILLS = frozenset({
    "langchain", "lang chain",
    "prompt engineering", "instruction tuning",
    "qlora", "lora", "peft", "fine tuning", "finetuning",
    "llm", "large language model", "generative ai",
    "openai", "chatgpt", "gpt",
    "rlhf",
})

DISQUALIFYING_SKILLS = frozenset({
    "computer vision", "image classification", "object detection",
    "yolo", "convolutional", "image segmentation",
    "opencv",
    "speech recognition", "asr", "tts", "text to speech", "speech synthesis",
    "audio processing", "sound classification", "voice recognition",
    "robotics", "ros", "autonomous driving", "lidar", "slam",
    "photoshop", "illustrator", "figma", "canva",
    "accounting", "tally", "gst", "salesforce", "crm",
    "seo", "content writing", "copywriting",
    "six sigma", "lean", "kaizen",
    "solidworks", "autocad",
})

TIER1_REGEX = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(TIER1_SKILLS, key=len, reverse=True))) + r')\b')
TIER2_REGEX = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(TIER2_SKILLS, key=len, reverse=True))) + r')\b')
TIER3_REGEX = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(TIER3_SKILLS, key=len, reverse=True))) + r')\b')
DISQ_REGEX  = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(DISQUALIFYING_SKILLS, key=len, reverse=True))) + r')\b')

# Module-level compiled regexes — UNCHANGED
PRIMARY_CORE_RE    = re.compile(r'\b(information retrieval|learning to rank|ltr|lambdamart|bm25|semantic search|candidate matching|search quality)\b')
SECONDARY_CORE_RE  = re.compile(r'\b(ranking system|recommendation system|candidate ranking|personalization|relevance|matching engine|elasticsearch|opensearch)\b')
EXPLICIT_VECTOR_RE = re.compile(r'\b(faiss|pinecone|qdrant|weaviate|milvus|pgvector)\b')
VECTOR_TEXT_RE     = re.compile(r'\b(vector search|vector database|chroma|ann|hnsw)\b')
# HR_TEXT_RE superseded by MARKETPLACE_RE in v6.0 — kept for schema reference only
# HR_TEXT_RE       = re.compile(r'\b(hr tech|recruiting tech|talent acquisition platform|marketplace product|job board)\b')
PROD_TEXT_RE       = re.compile(r'\b(scale|shipped|deployed|productionized|enterprise|latency|qps|inference optimization|tensorrt|vllm|distributed systems|ray|spark)\b')
# INFRA_TEXT_RE superseded by SEARCH_RE in v6.0 — kept for schema reference only
# INFRA_TEXT_RE      = re.compile(r'\b(search infra|retrieval pipeline|relevance|indexing|ranking optimization)\b')
EVAL_TEXT_RE       = re.compile(r'\b(ndcg|mrr|mean average precision|a/b test|ab testing|offline evaluation|online evaluation)\b')
RESEARCH_HEAVY_RE  = re.compile(r'\b(paper|publication|thesis|arxiv|academic)\b')

# ─────────────────────────────────────────────────────────────────────────────
# V6.0 — SEMANTIC BUCKET REGEXES (Step 4: JDIntentAnalyzer)
# ─────────────────────────────────────────────────────────────────────────────

RECOMMENDATION_RE = re.compile(
    r'\b(recommendation|personalization|feed ranking|product ranking|'
    r'collaborative filtering|content ranking|recommender|ranking model|'
    r'recommendation engine|recommendation system|recommender system)\b'
)
SEARCH_RE = re.compile(
    r'\b(semantic search|retrieval|hybrid search|information retrieval|'
    r'dense retrieval|query understanding|search engine|search ranking|'
    r'search quality|search system|search infrastructure|search platform|'
    r'retrieval system|retrieval pipeline)\b'
)
MARKETPLACE_RE = re.compile(
    r'\b(marketplace|two.?sided|supply.?demand|talent marketplace|'
    r'job matching|candidate matching|hiring platform|hr tech|'
    r'recruiting tech|talent acquisition|marketplace product|job board)\b'
)

# ─────────────────────────────────────────────────────────────────────────────
# V6.0 — OWNERSHIP VERB CATEGORIES (Step 7: OWNER/LEAD/CONTRIBUTOR/SUPPORT)
# ─────────────────────────────────────────────────────────────────────────────

OWNER_VERBS_RE = re.compile(
    r'\b(architected|designed|invented|pioneered|founded|created from scratch)\b'
)
LEAD_VERBS_RE = re.compile(
    r'\b(led|owned|drove|spearheaded|established|launched|initiated|'
    r'built and deployed|built and launched)\b'
)
CONTRIBUTOR_VERBS_RE = re.compile(
    r'\b(built|implemented|developed|created|wrote|shipped|engineered|'
    r'designed and implemented|coded|constructed|delivered)\b'
)
SUPPORT_VERBS_RE = re.compile(
    r'\b(worked on|assisted|supported|helped|participated|contributed to|'
    r'was part of|involved in|collaborated on)\b'
)

# v5 ownership regexes preserved for backward compat (used in evidence analyzer)
OWNERSHIP_HIGH_RE = re.compile(
    r'\b(designed|architected|led|owned|created|implemented|launched|drove|'
    r'founded|established|spearheaded|initiated|invented|pioneered|'
    r'built and deployed|built and launched)\b'
)
# OWNERSHIP_LOW_RE superseded by SUPPORT_VERBS_RE in v6.0 — kept for backward compat
# OWNERSHIP_LOW_RE = re.compile(
#     r'\b(worked on|assisted|supported|helped|participated|contributed to|'
#     r'was part of|involved in|collaborated on)\b'
# )

# ImpactAnalyzer — measurable outcomes (Step 9)
IMPACT_METRICS_RE = re.compile(
    r'\b(reduced latency|improved ctr|click.?through|conversion rate|revenue|'
    r'throughput|scaled to \d+[mk]?|million users|billion|10[xX]|'
    r'latency reduction|cost reduction|precision|recall|f1 score|'
    r'a/?b test|experiment|rollout|lifted|increased by|decreased by|'
    r'saved \$|reduced \$|qps improvement|p99|p50|nps improvement)\b'
)

# Impact value extraction — captures numbers/percentages near metric words
IMPACT_VALUE_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(%|x|[mk]\b|million|billion|users|queries|rps|qps|ms|seconds)',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────────────────────
# COMPANY / TITLE / LOCATION CONSTANTS — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

CONSULTING_INDUSTRIES = frozenset({
    "it services", "it consulting", "consulting",
    "business process outsourcing", "bpo", "outsourcing",
    "staffing", "managed services", "it staffing",
})
CONSULTING_NAMES = frozenset({
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "l&t infotech", "lti", "cognizant technology solutions", "atos",
    "niit technologies", "persistent systems", "cyient", "zensar",
    "birlasoft", "coforge", "kpit", "mastech", "firstsource", "wns", "genpact",
})
CONSULTING_RE = re.compile(r'\b(?:' + '|'.join(map(re.escape, CONSULTING_INDUSTRIES | CONSULTING_NAMES)) + r')\b')

PRODUCT_TECH_COMPANIES = frozenset({
    "swiggy", "zomato", "razorpay", "meesho", "flipkart", "ola", "gojek",
    "grab", "zepto", "uber", "doordash", "instacart",
})

ELITE_SEARCH_COMPANIES = frozenset({
    "google", "meta", "facebook", "linkedin", "netflix", "pinterest", "airbnb", "amazon",
})

SEARCH_COMPANIES = frozenset({
    "linkedin", "google", "meta", "amazon", "airbnb", "pinterest", "spotify", "netflix",
})

STRONG_TITLE_SCORES: dict[str, float] = {
    "search engineer": 1.0, "ranking engineer": 1.0,
    "relevance engineer": 1.0, "search scientist": 1.0,
    "recommendation engineer": 1.0, "recommendation systems engineer": 1.0,
    "applied ml engineer": 0.95, "applied ml": 0.95,
    "nlp engineer": 0.90, "ai engineer": 0.88,
    "ml engineer": 0.88, "machine learning engineer": 0.88,
    "applied scientist": 0.82, "research engineer": 0.75,
    "principal engineer": 0.82, "staff engineer": 0.82,
    "senior engineer": 0.72, "data engineer": 0.45,
}

NON_TECH_TITLES = frozenset({
    "marketing manager", "hr manager", "human resources", "operations manager",
    "business analyst", "content writer", "sales executive", "accountant",
    "project manager", "graphic designer", "customer support", "customer service",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "finance manager", "account manager", "talent acquisition", "recruiter",
    "data entry", "tester", "manual tester", "business development",
    "scrum master", "product manager", "sales", "marketing",
})

PREFERRED_LOCATIONS = frozenset({
    "pune", "noida", "delhi", "delhi ncr", "ncr", "new delhi",
    "gurgaon", "gurugram", "faridabad", "greater noida",
    "hyderabad", "mumbai", "bangalore", "bengaluru",
})

PROFICIENCY_WEIGHTS = {
    "expert": 1.0, "advanced": 0.75, "intermediate": 0.45, "beginner": 0.15,
}

SHIP_SIGNALS = frozenset({
    "production", "shipped", "launched", "deployed", "live system",
    "real users", "at scale", "million", "billion",
    "latency", "throughput", "serving", "inference", "qps", "rps",
    "a/b test", "experiment", "rollout",
    "ranking system", "retrieval system", "recommendation engine",
    "search engine", "embedding service", "vector index", "reranker",
    "search platform", "recommendation platform",
    "sla", "uptime", "oncall", "latency reduction", "cost reduction",
    "serving system", "online inference", "production traffic",
})
SHIP_SIGNALS_RE = re.compile(r'\b(?:' + '|'.join(map(re.escape, SHIP_SIGNALS)) + r')\b')

RESEARCH_SIGNALS = frozenset({
    "paper", "publication", "conference", "arxiv", "journal",
    "thesis", "academic", "phd candidate", "research lab",
    "pure research", "benchmark", "theoretical",
})
RESEARCH_SIGNALS_RE = re.compile(r'\b(?:' + '|'.join(map(re.escape, RESEARCH_SIGNALS)) + r')\b')

# Production-scale signals — differentiate 100 users from 10M users
SCALE_RE = re.compile(
    r'\\b(?:'
    r'\\d+(?:\\.\\d+)?\\s*(?:million|billion|m\\+?|b\\+?)\\s*(?:users?|requests?|queries?|candidates?|records?|dau|mau)\\b'
    r'|(?:millions?|billions?)\\s+of\\s+(?:users?|requests?|queries?|records?|profiles?)'
    r'|10[kmb]\\+?\\s*(?:users?|requests?|queries?)'
    r'|at\\s+(?:massive|web|production)\\s+scale'
    r'|high[- ](?:traffic|throughput|volume|qps)'
    r'|(?:1|2|5|10|50|100|500)[kmb]\\s+(?:rps|qps|tps)'
    r')',
    re.IGNORECASE
)

FOUNDING_MINDSET_RE = re.compile(
    r'\b(0\s*to\s*1|0->1|greenfield|from scratch|built v1|first engineer|founding|early stage|startup)\b',
    re.IGNORECASE
)

PASS2_POOL_SIZE = 12000
FAST_WORDS_RE = re.compile(
    r'\b(retrieval|recommendation|ranking|search|vector|embedding|nlp|rag|ndcg|'
    r'faiss|qdrant|pinecone|milvus|elasticsearch|weaviate|retrieval platform|'
    r'relevance|matching engine|candidate matching|recommendation pipeline|'
    r'search quality|personalization|ranking model)\b'
)
SHIP_FAST_RE = re.compile(r'\b(shipped|deployed|production|at scale|million|serving|built and)\b')

# ─────────────────────────────────────────────────────────────────────────────
# SPECIALIZATION PATHS (Step 8: CareerAnalyzer)
# ─────────────────────────────────────────────────────────────────────────────

_SPEC_RETRIEVAL_KW = ("retrieval", "search", "ranking", "recommendation", "matching",
                      "relevance", "reranking", "information retrieval", "vector")
_SPEC_ML_KW        = ("machine learning", "ml ", " ml", "deep learning", "data science",
                      "ai ", " ai", "nlp", "natural language")
_SPEC_BACKEND_KW   = ("backend", "software engineer", "platform", "infrastructure",
                      "microservice", "api", "distributed system")
_SPEC_DATA_KW      = ("data engineer", "data pipeline", "etl", "analytics",
                      "data warehouse", "big data", "spark")


# ─────────────────────────────────────────────────────────────────────────────
# SKILL TAXONOMY — authoritative mapping used by generate_reasoning — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────
SKILL_RANKING = {
    "learning to rank", "ltr", "lambdamart", "xgboost ranking",
    "recommendation systems", "recommender", "recommendation engine",
    "recommendation system", "collaborative filtering", "matrix factorization",
    "bpr", "svd", "als",
}
SKILL_SEARCH = {
    "opensearch", "elasticsearch", "solr", "bm25", "lucene",
    "search ranking", "information retrieval", "tfidf", "tf-idf",
    "inverted index", "full-text search", "keyword search",
}
SKILL_VECTOR = {
    "faiss", "pinecone", "qdrant", "weaviate", "milvus",
    "pgvector", "chroma", "vespa", "annoy", "scann",
    "hnsw", "approximate nearest neighbor", "ann search",
    "vector database", "vector db", "vector store",
    "vector search", "dense retrieval",
}
SKILL_SEMANTIC = {
    "semantic search", "sentence transformers", "bi-encoder", "cross-encoder",
    "embeddings", "text embeddings", "bert", "nlp", "natural language processing",
    "neural retrieval", "dense passage retrieval", "dpr",
}
SKILL_RAG = {
    "rag", "retrieval-augmented generation", "langchain", "llamaindex",
    "llm", "gpt", "openai", "llama", "generative ai", "prompt engineering",
}


def _categorise_skill(skill_name: str) -> str | None:
    s = re.sub(r"[-_/]", " ", skill_name.lower()).strip()
    if any(kw in s for kw in SKILL_RANKING):  return "RANKING"
    if any(kw in s for kw in SKILL_SEARCH):   return "SEARCH"
    if any(kw in s for kw in SKILL_VECTOR):   return "VECTOR"
    if any(kw in s for kw in SKILL_SEMANTIC): return "SEMANTIC"
    if any(kw in s for kw in SKILL_RAG):      return "RAG"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DATE PARSING  [B7] — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────
def parse_date(s: str) -> date | None:
    if not s:
        return None
    s = s.replace("/", "-")
    for fmt in ("%Y-%m-%d", "%Y-%m", "%d-%m-%Y", "%Y"):
        try:
            return datetime.strptime(s[:10].strip(), fmt).date()
        except ValueError:
            continue
    return None


def calculate_actual_yoe(career: list) -> float:
    """Calculate non-overlapping years of experience."""
    intervals = []
    for job in career:
        start_str = job.get("start_date", "")
        end_str   = job.get("end_date", "")
        start_dt  = parse_date(start_str)
        end_dt = None
        if end_str:
            end_dt = parse_date(end_str)
        dur = int(job.get("duration_months", 0) or 0)
        if start_dt and not end_dt and dur > 0:
            try:
                y, m = divmod(start_dt.month - 1 + dur, 12)
                end_dt = start_dt.replace(year=start_dt.year + y, month=m + 1)
            except ValueError:
                pass
        if not start_dt and dur > 0:
            intervals.append((-1, dur))
            continue
        if start_dt:
            if not end_dt:
                end_dt = TODAY
            start_m = start_dt.year * 12 + start_dt.month
            end_m   = end_dt.year * 12 + end_dt.month
            if end_m > start_m:
                intervals.append((start_m, end_m))

    valid_intervals = [i for i in intervals if i[0] != -1]
    valid_intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in valid_intervals:
        if not merged:
            merged.append(interval)
        else:
            last = merged[-1]
            if interval[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], interval[1]))
            else:
                merged.append(interval)
    total_months = sum(i[1] - i[0] for i in merged)
    for i in intervals:
        if i[0] == -1:
            total_months += i[1]
    return total_months / 12.0


# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTION — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────
def is_honeypot(candidate: dict) -> bool:
    profile     = candidate.get("profile", {}) or {}
    career      = candidate.get("career_history", []) or []
    skills      = candidate.get("skills", []) or []
    claimed_yoe = float(profile.get("years_of_experience", 0) or 0)
    actual_yoe  = calculate_actual_yoe(career)

    for job in career:
        start_dt = parse_date(job.get("start_date", ""))
        stated   = int(job.get("duration_months", 0) or 0)
        if start_dt and stated > 0:
            max_possible = (TODAY.year - start_dt.year) * 12 + (TODAY.month - start_dt.month) + 3
            if stated > max_possible + 6:
                return True

    if actual_yoe > claimed_yoe + 3.0 and actual_yoe > 10.0:
        return True

    if skills and claimed_yoe <= 10.0 and actual_yoe <= 10.0:
        expert_advanced = [s for s in skills if str(s.get("proficiency", "")).lower() in ("expert", "advanced")]
        zero_evidence   = sum(1 for s in expert_advanced if float(s.get("duration_months", 0) or 0) <= 0.0)
        if len(expert_advanced) >= 20 and zero_evidence >= 10:
            return True
        if len(expert_advanced) >= 30 and zero_evidence >= 20:
            return True

    expert_advanced = [s for s in skills if s.get("proficiency") in ("expert", "advanced")]
    if len(expert_advanced) >= 20:
        zero_evidence = sum(
            1 for s in expert_advanced
            if int(s.get("duration_months", 0) or 0) == 0 and int(s.get("endorsements", 0) or 0) == 0
        )
        if zero_evidence >= 10:
            return True
        names = " ".join(s.get("name", "").lower() for s in expert_advanced)
        has_non_tech = any(re.search(rf"\b{kw}\b", names) for kw in ("accounting", "tally", "sales", "hr", "marketing", "seo", "content writing"))
        has_tech     = any(re.search(rf"\b{kw}\b", names) for kw in ("machine learning", "backend", "react", "python", "aws"))
        if has_non_tech and has_tech:
            return True

    claimed = float(profile.get("years_of_experience", 0) or 0)
    if 0 < claimed < 10 and any(int(j.get("duration_months", 0) or 0) > 240 for j in career):
        return True

    for s in skills:
        name = s.get("name", "").lower()
        dur  = int(s.get("duration_months", 0) or 0)
        if dur > 36 and ("langchain" in name or "openai" in name or "chatgpt" in name or "llama" in name):
            return True
        if dur > 60 and ("qdrant" in name or "weaviate" in name or "pinecone" in name):
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — FAST FILTER (100K → 12K) — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────
def fast_score(candidate: dict) -> float:
    profile = candidate.get("profile", {}) or {}
    skills  = candidate.get("skills", []) or []
    career  = candidate.get("career_history", []) or []
    title   = profile.get("current_title", "").lower()

    score_penalty = 0.0
    if any(title == t or title.startswith(t + " ") or title.startswith(t + ",") for t in NON_TECH_TITLES):
        score_penalty = 20.0

    t1_count = t2_count = 0
    for s in skills:
        name = s.get("name", "").lower().replace("-", " ").replace("_", " ").replace("/", " ").strip()
        if len(name) < 3:
            continue
        if TIER1_REGEX.search(name):
            t1_count += 1
        elif TIER2_REGEX.search(name):
            t2_count += 1

    title_hit = any(t in title for t in STRONG_TITLE_SCORES)

    headline    = profile.get("headline", "") or ""
    summary     = profile.get("summary", "")  or ""
    career_desc = ""
    elite_hit   = False
    for j in career[:3]:
        company = (j.get("company", "") or "").lower()
        if any(elite in company for elite in ELITE_SEARCH_COMPANIES):
            elite_hit = True
        career_desc += (j.get("description", "") or "") + " "
    career_desc = career_desc[:4000]
    combined    = (headline + " " + summary + " " + career_desc).lower()

    text_hit  = bool(FAST_WORDS_RE.search(combined))
    ship_hits = len(set(SHIP_FAST_RE.findall(combined)))

    score = t1_count * 2.5 + t2_count * 0.5
    if title_hit:  score += 4.0
    if text_hit:   score += 1.0
    if elite_hit:  score += 3.0
    if PRIMARY_CORE_RE.search(combined):
        score += 3.0
    score += min(ship_hits * 0.5, 2.0)
    return score - score_penalty


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE NORMALIZER — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

def normalize(raw: dict) -> Candidate:
    profile = raw.get("profile", {}) or {}
    return Candidate(
        id=raw.get("candidate_id", ""),
        headline=profile.get("headline", "") or "",
        summary=profile.get("summary", "") or "",
        current_title=profile.get("current_title", "") or "",
        current_company=profile.get("current_company", "") or "",
        location=profile.get("location", "") or "",
        country=profile.get("country", "") or "",
        years_of_experience=float(profile.get("years_of_experience", 0) or 0),
        career=raw.get("career_history", []) or [],
        skills=raw.get("skills", []) or [],
        signals=raw.get("redrob_signals", {}) or {},
    )


# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE EXTRACTION HELPERS  (v6.0 — shared by multiple analyzers)
# ─────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into rough sentences for evidence extraction."""
    parts = re.split(r'[.;!\n]+', text)
    return [p.strip() for p in parts if len(p.strip()) >= 15]


def _get_career_text(c: Candidate, max_jobs: int = 4) -> str:
    """Concatenate headline + summary + top career descriptions."""
    parts = [c.headline, c.summary]
    for job in c.career[:max_jobs]:
        parts.append((job.get("title", "") or "") + " " + (job.get("description", "") or ""))
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — FEATURE EXTRACTION  (v6.0 — all analyzers rewritten)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_career(c: Candidate) -> CareerFeatures:
    """
    CareerAnalyzer v6.0 — integer range 0–25.

    Components:
      YoE band fit     0–8   (sweet spot 5-9)
      Consistency      0–4   (fraction in long-tenure roles)
      Promotions       0–3
      ML depth         0–4   (years in ML-relevant roles)
      Specialization   0–3   (RETRIEVAL > ML > BACKEND > GENERALIST)
      Career depth     0–3   (profile richness, absorbs ConfidenceAnalyzer)
    """
    actual_yoe = calculate_actual_yoe(c.career)
    yoe = c.years_of_experience
    evidence = []

    # ── YoE band (0–8) ─────────────────────────────────────────────────
    if   yoe < 3.5:  yoe_pts = 3
    elif yoe < 5:    yoe_pts = 5
    elif yoe <= 9:   yoe_pts = 8   # JD sweet spot
    elif yoe <= 12:  yoe_pts = 6
    else:            yoe_pts = 4

    # ── Career consistency (0–4) ────────────────────────────────────────
    total_dur = sum(max(1, int(j.get("duration_months", 1) or 1)) for j in c.career)
    long_dur  = sum(
        max(1, int(j.get("duration_months", 1) or 1))
        for j in c.career if int(j.get("duration_months", 1) or 1) >= 18
    )
    consistency = long_dur / total_dur if total_dur > 0 else 1.0
    consistency_pts = min(4, int(consistency * 4))

    # ── Promotions (0–3) ───────────────────────────────────────────────
    _levels = {
        "intern": 1, "trainee": 1, "junior": 2, "jr": 2,
        "senior": 4, "lead": 5, "staff": 5, "principal": 5,
    }
    promotion_count = 0
    if len(c.career) >= 2:
        for i in range(len(c.career) - 1):
            curr_co = (c.career[i].get("company",     "") or "").lower()
            prev_co = (c.career[i + 1].get("company", "") or "").lower()
            if curr_co and curr_co == prev_co:
                curr_t  = (c.career[i].get("title",     "") or "").lower()
                prev_t  = (c.career[i + 1].get("title", "") or "").lower()
                curr_lvl = max((_levels[k] for k in _levels if k in curr_t), default=3)
                prev_lvl = max((_levels[k] for k in _levels if k in prev_t), default=3)
                if curr_lvl > prev_lvl:
                    promotion_count += 1
    promo_pts = min(3, promotion_count)

    # ── ML years depth (0–4) ───────────────────────────────────────────
    ml_months = 0
    for job in c.career:
        t = (job.get("title", "") or "").lower()
        d = (job.get("description", "") or "").lower()
        if any(kw in t or kw in d for kw in _SPEC_ML_KW + _SPEC_RETRIEVAL_KW):
            ml_months += int(job.get("duration_months", 0) or 0)
    ml_years = ml_months / 12.0
    ml_pts = min(4, int(ml_years * 0.8))

    # ── Specialization (0–3) ──────────────────────────────────────────
    spec_counts = {"RETRIEVAL": 0, "ML": 0, "BACKEND": 0, "DATA": 0}
    for job in c.career[:5]:
        t = (job.get("title", "") or "").lower()
        d = (job.get("description", "") or "").lower()[:500]
        td = t + " " + d
        dur = max(1, int(job.get("duration_months", 1) or 1))
        if any(kw in td for kw in _SPEC_RETRIEVAL_KW):
            spec_counts["RETRIEVAL"] += dur
        elif any(kw in td for kw in _SPEC_ML_KW):
            spec_counts["ML"] += dur
        elif any(kw in td for kw in _SPEC_BACKEND_KW):
            spec_counts["BACKEND"] += dur
        elif any(kw in td for kw in _SPEC_DATA_KW):
            spec_counts["DATA"] += dur

    if any(spec_counts.values()):
        specialization = max(spec_counts, key=spec_counts.get)
        if spec_counts[specialization] == 0:
            specialization = "GENERALIST"
    else:
        specialization = "GENERALIST"

    spec_bonus = {"RETRIEVAL": 3, "ML": 2, "BACKEND": 1, "DATA": 1, "GENERALIST": 0}
    spec_pts = spec_bonus.get(specialization, 0)

    if specialization == "RETRIEVAL":
        evidence.append(Evidence("career", f"Specialization in retrieval/search/ranking across career", priority=8))

    # ── Career depth / profile richness (0–3) — absorbs ConfidenceAnalyzer
    depth_score = 0.0
    if c.headline and len(c.headline) > 20: depth_score += 0.5
    if c.summary  and len(c.summary)  > 50: depth_score += 0.5
    if len(c.career) >= 2: depth_score += 0.5
    if len(c.career) >= 4: depth_score += 0.5
    filled_descs = sum(1 for j in c.career if (j.get("description", "") or "").strip())
    if filled_descs >= 2: depth_score += 0.5
    if len(c.skills) >= 5: depth_score += 0.5
    depth_pts = min(3, int(depth_score))

    # ── Product / Startup / Service year classification ────────────────
    product_months = startup_months = service_months = 0
    for job in c.career:
        industry = (job.get("industry",     "") or "").lower()
        company  = (job.get("company",      "") or "").lower()
        j_size   = (job.get("company_size", "") or "")
        dur      = max(1, int(job.get("duration_months", 1) or 1))
        is_consulting = bool(CONSULTING_RE.search(industry) or CONSULTING_RE.search(company))
        if is_consulting:
            service_months += dur
        elif j_size in ("1-10", "11-50", "51-200"):
            startup_months += dur
        else:
            product_months += dur

    if 5 <= yoe <= 9:
        evidence.append(Evidence("career", f"{yoe:g} years experience in JD's 5-9 year sweet spot", priority=6))

    score = min(25, yoe_pts + consistency_pts + promo_pts + ml_pts + spec_pts + depth_pts)

    return CareerFeatures(
        score=score,
        years_exp=yoe,
        actual_yoe=actual_yoe,
        career_depth=depth_score,
        career_consistency=consistency,
        specialization=specialization,
        promotion_count=promotion_count,
        product_years=product_months / 12.0,
        startup_years=startup_months / 12.0,
        service_years=service_months / 12.0,
        ml_years=ml_years,
        evidence=evidence,
    )


def analyze_company(c: Candidate) -> CompanyFeatures:
    """
    CompanyAnalyzer v6.0 — returns typed profile + score 0–6.

    Returns company *type* (MARKETPLACE/SEARCH/PRODUCT/STARTUP/CONSULTING/OTHER),
    not just a score. The scorer decides how much each type matters.
    """
    evidence = []
    elite_hit = False
    search_exposure = False
    ranking_exposure = False
    is_startup = False
    total_months = consulting_months = 0
    best_type = "OTHER"
    founder_mindset = False

    for job in c.career:
        comp     = (job.get("company",      "") or "").lower()
        industry = (job.get("industry",     "") or "").lower()
        j_size   = (job.get("company_size", "") or "")
        desc     = (job.get("description",  "") or "").lower()[:500]
        dur      = max(1, int(job.get("duration_months", 1) or 1))
        total_months += dur

        is_consulting = bool(CONSULTING_RE.search(industry) or CONSULTING_RE.search(comp))
        if FOUNDING_MINDSET_RE.search(desc) or FOUNDING_MINDSET_RE.search(comp):
            founder_mindset = True
        if is_consulting:
            consulting_months += dur

        if any(ec in comp for ec in ELITE_SEARCH_COMPANIES):
            elite_hit = True
            evidence.append(Evidence("company", f"Worked at {comp.title()} (elite search/recommendation company)", priority=7))

        if any(pc in comp for pc in PRODUCT_TECH_COMPANIES):
            evidence.append(Evidence("company", f"Worked at {comp.title()} (product tech company)", priority=5))

        if j_size in ("1-10", "11-50", "51-200"):
            is_startup = True

        # Detect search/ranking exposure from descriptions
        if SEARCH_RE.search(desc) or PRIMARY_CORE_RE.search(desc):
            search_exposure = True
        if RECOMMENDATION_RE.search(desc) or SECONDARY_CORE_RE.search(desc):
            ranking_exposure = True
        if MARKETPLACE_RE.search(desc):
            best_type = "MARKETPLACE"

    consult_frac = consulting_months / total_months if total_months > 0 else 0.0

    # Classify company type
    if best_type != "MARKETPLACE":
        if search_exposure and ranking_exposure:
            best_type = "SEARCH"
        elif any(any(pc in (j.get("company", "") or "").lower() for pc in PRODUCT_TECH_COMPANIES) for j in c.career):
            best_type = "PRODUCT"
        elif elite_hit:
            best_type = "SEARCH"
        elif is_startup:
            best_type = "STARTUP"
        elif consult_frac >= 0.80:
            best_type = "CONSULTING"

    # Score: 0–6
    type_scores = {
        "MARKETPLACE": 4, "SEARCH": 4, "PRODUCT": 3,
        "STARTUP": 3, "OTHER": 2, "CONSULTING": 0,
    }
    score = type_scores.get(best_type, 2)
    if elite_hit:
        score = min(6, score + 1)
    if search_exposure and best_type != "CONSULTING":
        score = min(6, score + 1)

    # Consulting penalty
    if consult_frac >= 0.95:
        score = max(0, score - 3)
    elif consult_frac > 0.90:
        score = max(0, score - 2)

    return CompanyFeatures(
        score=min(6, max(0, score)),
        company_type=best_type,
        search_exposure=search_exposure,
        ranking_exposure=ranking_exposure,
        startup=is_startup,
        elite_company=elite_hit,
        consult_fraction=consult_frac,
        founder_mindset=founder_mindset,
        evidence=evidence,
    )


def analyze_skills(c: Candidate) -> SkillFeatures:
    """
    SkillAnalyzer v6.0 — supporting evidence only, 0–5.
    Skills confirm experience; they don't substitute for it.
    """
    evidence = []
    t1_count = t2_count = t3_count = 0
    disq_count = 0
    total_count = 0
    tier1_names = []

    for skill in c.skills:
        name = re.sub(r"[-_/]", " ", skill.get("name", "").lower()).strip()
        if len(name) < 3:
            continue
        total_count += 1

        if TIER1_REGEX.search(name):
            t1_count += 1
            tier1_names.append(name)
        elif TIER2_REGEX.search(name):
            t2_count += 1
        elif TIER3_REGEX.search(name):
            t3_count += 1
        if DISQ_REGEX.search(name):
            disq_count += 1

    disq_frac = disq_count / total_count if total_count > 0 else 0.0

    # Score: 0–5
    score = 0
    if t1_count >= 5:   score = 3
    elif t1_count >= 3: score = 2
    elif t1_count >= 1: score = 1

    if t2_count >= 5:   score += 2
    elif t2_count >= 2: score += 1

    # T3-only penalty: LangChain-only gets almost nothing
    if t1_count == 0 and t2_count == 0 and t3_count > 0:
        score = max(0, score)  # already 0

    # Domain disqualification
    if disq_frac > 0.60:
        score = 0
    elif disq_frac > 0.40:
        score = max(0, score - 2)

    score = min(1, max(0, score))  # capped at 1: skills mean almost nothing without experience: skills confirm, don't substitute experience

    if tier1_names:
        evidence.append(Evidence("skill", f"Core skills: {', '.join(tier1_names[:4])}", priority=5))

    return SkillFeatures(
        score=score,
        tier1_count=t1_count,
        tier2_count=t2_count,
        disq_fraction=disq_frac,
        evidence=evidence,
    )


def analyze_evidence(c: Candidate) -> EvidenceFeatures:
    """
    EvidenceAnalyzer v6.0 — extracts Evidence objects with provenance sentences.

    For each career description sentence, detects domain + verb co-occurrence
    and creates an Evidence object linking the finding to the source text.
    """
    full_text = _get_career_text(c).lower()
    sentences = _split_sentences(full_text)
    evidence = []

    # Domain signals
    retrieval      = bool(SEARCH_RE.search(full_text))
    recommendation = bool(RECOMMENDATION_RE.search(full_text))
    ranking        = bool(re.search(r'\b(ranking|learning to rank|ltr|lambdamart|reranking)\b', full_text))
    search_rel     = bool(re.search(r'\b(search relevance|relevance|search quality|query understanding)\b', full_text))
    marketplace    = bool(MARKETPLACE_RE.search(full_text))
    prod_deployed  = bool(re.search(r'\b(production|deployed|live|real users|at scale|serving|online)\b', full_text))

    # Extract evidence with provenance sentences
    for sentence in sentences:
        s_lower = sentence.lower()

        # Check for domain + verb co-occurrence (strongest signal)
        has_ownership = bool(OWNERSHIP_HIGH_RE.search(s_lower))
        has_prod      = bool(PROD_TEXT_RE.search(s_lower))

        if SEARCH_RE.search(s_lower):
            pri = 10 if has_ownership else 8 if has_prod else 6
            evidence.append(Evidence("retrieval", sentence[:120], priority=pri))
        elif RECOMMENDATION_RE.search(s_lower):
            pri = 10 if has_ownership else 8 if has_prod else 6
            evidence.append(Evidence("recommendation", sentence[:120], priority=pri))
        elif MARKETPLACE_RE.search(s_lower):
            pri = 9 if has_ownership else 7
            evidence.append(Evidence("marketplace", sentence[:120], priority=pri))
        elif has_prod and (TIER1_REGEX.search(s_lower) or TIER2_REGEX.search(s_lower)):
            evidence.append(Evidence("production", sentence[:120], priority=7))
        elif has_ownership:
            evidence.append(Evidence("ownership", sentence[:120], priority=5))

    return EvidenceFeatures(
        retrieval=retrieval,
        recommendation=recommendation,
        ranking=ranking,
        search_relevance=search_rel,
        marketplace=marketplace,
        production_deployed=prod_deployed,
        evidence=evidence,
    )


def analyze_ownership(c: Candidate) -> OwnershipFeatures:
    """
    OwnershipAnalyzer v6.0 — classifies into OWNER/LEAD/CONTRIBUTOR/SUPPORT.

    OWNER:       Architected / Designed / Invented / Pioneered (score 12)
    LEAD:        Led / Owned / Drove / Spearheaded             (score 9)
    CONTRIBUTOR: Built / Implemented / Developed / Shipped     (score 5)
    SUPPORT:     Worked on / Assisted / Supported              (score 2)
    UNKNOWN:     No verbs found                                (score 3)
    """
    text = _get_career_text(c).lower()
    sentences = _split_sentences(text)
    evidence = []

    owner_count  = len(OWNER_VERBS_RE.findall(text))
    lead_count   = len(LEAD_VERBS_RE.findall(text))
    contrib_count = len(CONTRIBUTOR_VERBS_RE.findall(text))
    support_count = len(SUPPORT_VERBS_RE.findall(text))

    # Classify ownership level
    if owner_count >= 2 or (owner_count >= 1 and lead_count >= 1):
        level = "OWNER"
        score = 12
    elif lead_count >= 2 or (lead_count >= 1 and contrib_count >= 2):
        level = "LEAD"
        score = 9
    elif contrib_count >= 2:
        level = "CONTRIBUTOR"
        score = 5
    elif support_count >= 1:
        level = "SUPPORT"
        score = 2
    else:
        level = "UNKNOWN"
        score = 3

    # Extract the best ownership evidence sentence
    for sentence in sentences:
        s_lower = sentence.lower()
        if OWNER_VERBS_RE.search(s_lower) or LEAD_VERBS_RE.search(s_lower):
            evidence.append(Evidence("ownership", sentence[:120], priority=7))
            break

    return OwnershipFeatures(score=score, level=level, evidence=evidence)


def analyze_impact(c: Candidate) -> ImpactFeatures:
    """
    ImpactAnalyzer v6.0 — extracts Impact objects (metric, improvement), 0–10.
    """
    text = _get_career_text(c).lower()
    sentences = _split_sentences(text)
    evidence = []
    impacts = []

    metric_hits = set(IMPACT_METRICS_RE.findall(text))
    unique_count = len(metric_hits)

    # Extract Impact objects with values
    for sentence in sentences:
        s_lower = sentence.lower()
        if IMPACT_METRICS_RE.search(s_lower):
            # Try to extract the metric name and value
            metric_match = IMPACT_METRICS_RE.search(s_lower)
            value_match  = IMPACT_VALUE_RE.search(s_lower)
            if metric_match:
                metric_name = metric_match.group(1)
                improvement = ""
                if value_match:
                    improvement = value_match.group(0)
                impacts.append(Impact(metric=metric_name, improvement=improvement))
                evidence.append(Evidence("impact", sentence[:120], priority=6))

    # Score: 0–10 (diminishing returns)
    if   unique_count >= 5: score = 10
    elif unique_count >= 4: score = 8
    elif unique_count >= 3: score = 7
    elif unique_count >= 2: score = 5
    elif unique_count >= 1: score = 3
    else:                   score = 0

    return ImpactFeatures(score=score, impacts=impacts, evidence=evidence)


def analyze_production(c: Candidate) -> ProductionFeatures:
    """
    ProductionAnalyzer v6.0 — evidence of deployed systems, 0–18.
    Shippers > Researchers. Temporal decay weights recent production heavier.
    """
    evidence = []
    ship_total = 0.0
    research_total = 0.0

    for job in c.career[:5]:
        end_str = str(job.get("end_date", str(TODAY))).split("T")[0]
        try:
            end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
        except ValueError:
            end_date = TODAY
        years_ago = max(0.0, (TODAY - end_date).days / 365.25)
        decay     = max(0.05, math.exp(-0.35 * years_ago))

        job_text = ((job.get("title", "") or "") + " " + (job.get("description", "") or "")).lower()
        j_size   = (job.get("company_size", "") or "")
        dur      = max(1, int(job.get("duration_months", 1) or 1))

        size_mult = 1.25 if j_size in ("1-10", "11-50", "51-200") else 1.10 if j_size in ("201-500", "501-1000") else 1.00

        has_domain   = bool(TIER1_REGEX.search(job_text) or TIER2_REGEX.search(job_text))
        ship_hits    = len(set(SHIP_SIGNALS_RE.findall(job_text))) if has_domain else 0
        research_hits = len(set(RESEARCH_SIGNALS_RE.findall(job_text)))
        pr_hits      = len(set(PROD_TEXT_RE.findall(job_text)))
        p_hits       = len(set(PRIMARY_CORE_RE.findall(job_text)))
        ex_hits      = len(set(EXPLICIT_VECTOR_RE.findall(job_text)))
        scale_hits   = len(set(SCALE_RE.findall(job_text))) if has_domain else 0
        built_vector = bool(re.search(r'\b(built|implemented|shipped|deployed|created|developed)\b.{0,80}?\b(faiss|pinecone|qdrant|weaviate|milvus|pgvector|chroma)\b', job_text))
        if built_vector:
            ship_total += 4.0 * decay * size_mult  # Built FAISS > FAISS skill

        # Co-occurrence: shipping IR/Vector systems at scale -> massive boost
        if pr_hits > 0 and (p_hits > 0 or ex_hits > 0):
            ship_total += (ship_hits + pr_hits) * decay * size_mult * 2.0
        elif has_domain and ship_hits > 0:
            ship_total += ship_hits * decay * size_mult * 1.5
        elif has_domain:
            ship_total += pr_hits * decay * size_mult * 0.5
        # Scale bonus: mentions of scale (millions of users, high QPS) are premium signal
        if scale_hits > 0 and has_domain:
            ship_total += scale_hits * decay * size_mult * 1.2

        research_total += research_hits * decay

        if (ship_hits > 0 or pr_hits > 0) and has_domain:
            desc_snip = (job.get("description", "") or "")[:100]
            evidence.append(Evidence("production", desc_snip, priority=8))

    research_only = research_total > 1.0 and ship_total <= 0.5

    # Map to 0–18 with better scaling
    raw = min(18.0, ship_total * 1.2)
    if research_only:
        raw = max(0, raw - 5)

    # Title boost for strong production titles
    title = c.current_title.lower()
    title_prod = 0
    for t, sc in STRONG_TITLE_SCORES.items():
        if t in title:
            title_prod = max(title_prod, int(sc * 4))
    raw = min(18, raw + title_prod * 0.5)

    score = max(0, min(18, int(raw)))

    return ProductionFeatures(
        score=score,
        ship_count=int(ship_total),
        research_only=research_only,
        evidence=evidence,
    )


def analyze_jd_intent(c: Candidate) -> JDIntentFeatures:
    """
    JDIntentAnalyzer v6.0 — semantic bucket detection, 0–20.

    Instead of counting keyword hits, detect problem-domain alignment:
      RECOMMENDATION hit: +7    (personalization, feed ranking, recommender)
      SEARCH hit:         +6    (semantic search, retrieval, hybrid search)
      MARKETPLACE hit:    +5    (matching, two-sided, talent marketplace)
      EVALUATION hit:     +3    (NDCG, A/B testing, offline eval)
      VECTOR hit:         +2    (FAISS, Qdrant, Pinecone, etc.)
      hybrid_bonus:       +2    (traditional IR + dense vectors in same role)
      ltr_bonus:          +1    (gradient boosting + search/ranking)
    """
    full_text = _get_career_text(c).lower()
    skill_names = " ".join(s.get("name", "").lower() for s in c.skills)
    combined = full_text + " " + skill_names
    evidence = []

    # Recency-weighted domain hits: recent experience counts more than old
    # Each job contributes with decay weight: 1.0 -> 0.3 over 7 years
    _rec_score = _srch_score = _mkt_score = 0.0
    for _job in c.career[:7]:
        _end_str = str(_job.get("end_date", str(TODAY))).split("T")[0]
        try:    _end_dt = datetime.strptime(_end_str, "%Y-%m-%d").date()
        except: _end_dt = TODAY
        _yrs_ago = max(0.0, (TODAY - _end_dt).days / 365.25)
        _w = max(0.0, 1.0 - 0.25 * _yrs_ago)  # 1.0 at current, 0.3 at 7+ yrs
        _jt = ((_job.get("title", "") or "") + " " + (_job.get("description", "") or "")).lower()
        if RECOMMENDATION_RE.search(_jt): _rec_score  += _w
        if SEARCH_RE.search(_jt):         _srch_score += _w
        if MARKETPLACE_RE.search(_jt):    _mkt_score  += _w
    recommendation_hit = bool(RECOMMENDATION_RE.search(combined)) or _rec_score  >= 0.5
    search_hit         = bool(SEARCH_RE.search(combined))         or _srch_score >= 0.5
    marketplace_hit    = bool(MARKETPLACE_RE.search(combined))    or _mkt_score  >= 0.5
    evaluation_hit     = bool(EVAL_TEXT_RE.search(combined))
    vector_hit         = bool(EXPLICIT_VECTOR_RE.search(combined) or VECTOR_TEXT_RE.search(combined))
    # Store recency scores for scoring (higher = more recent domain experience)
    _rec_weight  = min(1.0, _rec_score  / 2.0)  # normalize: 2 recent jobs = max
    _srch_weight = min(1.0, _srch_score / 2.0)

    # Hybrid search: traditional IR + dense vectors in same profile
    hybrid_bonus = 0
    if PRIMARY_CORE_RE.search(combined) and EXPLICIT_VECTOR_RE.search(combined):
        hybrid_bonus = 2

    # LTR: gradient boosting alongside search/ranking
    ltr_bonus = 0
    if any(kw in combined for kw in ("xgboost", "lightgbm", "gradient boosting")):
        if any(kw in combined for kw in ("search", "ranking", "relevance", "recommend")):
            ltr_bonus = 1

    # Search-company bonus: detect via career history
    for job in c.career:
        comp = (job.get("company", "") or "").lower()
        if any(sc in comp for sc in SEARCH_COMPANIES):
            if not search_hit:
                search_hit = True  # implicit search experience

    # Score: sum of bucket hits, capped at 20
    score = 0
    if recommendation_hit:
        # Recency bonus: add up to 2 extra points for CURRENT recommendation work
        score += 7 + int(_rec_weight * 2)  # Balanced with Search and Marketplace
        evidence.append(Evidence("recommendation", "Profile demonstrates recommendation/personalization experience", priority=9))
    if search_hit:
        # Recency bonus: add up to 2 extra points for CURRENT search/retrieval work
        score += 7 + int(_srch_weight * 2)  # Balanced with Recommendation
        evidence.append(Evidence("retrieval", "Profile demonstrates search/retrieval system experience", priority=9))
    if marketplace_hit:
        score += 7
        evidence.append(Evidence("marketplace", "Profile shows marketplace/matching platform experience", priority=8))
    if evaluation_hit:
        score += 3
    if vector_hit:
        score += 2
    score += hybrid_bonus
    score += ltr_bonus

    # Generic ML penalty: "ML Engineer" with no retrieval/search evidence
    title_lower = c.current_title.lower()
    generic_ml  = any(kw in title_lower for kw in ("data scientist", "machine learning", "ml engineer", "data engineer"))
    if generic_ml and not search_hit and not recommendation_hit and not vector_hit:
        score = max(0, score - 3)

    score = min(24, max(0, score))  # cap raised to 24 to allow recency bonuses

    return JDIntentFeatures(
        score=score,
        recommendation_hit=recommendation_hit,
        search_hit=search_hit,
        marketplace_hit=marketplace_hit,
        evaluation_hit=evaluation_hit,
        vector_hit=vector_hit,
        hybrid_bonus=hybrid_bonus,
        ltr_bonus=ltr_bonus,
        evidence=evidence,
    )


def analyze_evaluation(c: Candidate) -> EvalFeatures:
    """
    EvalAnalyzer v6.0 — evaluation methodology detection, 0–8.

    JD: "Hands-on experience designing evaluation frameworks for ranking systems
         — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation."
    """
    full_text = _get_career_text(c).lower()
    skill_names = " ".join(s.get("name", "").lower() for s in c.skills)
    combined = full_text + " " + skill_names
    evidence = []

    eval_methods = set()
    eval_kw_map = {
        "ndcg":              "NDCG",
        "mrr":               "MRR",
        "mean average precision": "MAP",
        "a/b test":          "A/B Testing",
        "ab testing":        "A/B Testing",
        "offline evaluation": "Offline Eval",
        "online evaluation":  "Online Eval",
        "evaluation framework": "Eval Framework",
        "eval framework":     "Eval Framework",
    }

    for kw, method in eval_kw_map.items():
        if kw in combined:
            eval_methods.add(method)

    has_eval = len(eval_methods) > 0

        # Score: 0-15 (Search Quality metrics are nearly mandatory for top ranks)
    if   len(eval_methods) >= 4: score = 15
    elif len(eval_methods) >= 3: score = 12
    elif len(eval_methods) >= 2: score = 8
    elif len(eval_methods) >= 1: score = 4
    else:                        score = 0

    if eval_methods:
        evidence.append(Evidence("evaluation", f"Evaluation methods: {', '.join(sorted(eval_methods))}", priority=7))

    return EvalFeatures(
        score=score,
        has_eval=has_eval,
        eval_methods=sorted(eval_methods),
        evidence=evidence,
    )


def analyze_trajectory(c: Candidate) -> TrajectoryFeatures:
    """
    TrajectoryAnalyzer v6.0 — career progression quality, 0–6.
    """
    if not c.career:
        return TrajectoryFeatures(score=3, penalty=0, reward=0)

    levels = {
        "intern": 1, "trainee": 1, "junior": 2, "jr": 2,
        "senior": 4, "lead": 5, "staff": 5, "principal": 5,
    }

    trajectory = []
    for job in reversed(c.career):
        title = (job.get("title", "") or "").lower()
        lvl = 0
        for k, v in levels.items():
            if k in title:
                lvl = max(lvl, v)   # always take the highest level found
        if lvl == 0:
            lvl = 3
        dur = max(1, int(job.get("duration_months", 1) or 1))
        trajectory.append((lvl, dur))

    penalty = 0
    reward  = 0
    for i in range(1, len(trajectory)):
        prev_lvl, prev_dur = trajectory[i - 1]
        curr_lvl, curr_dur = trajectory[i]
        # Title chasing: rapid promotion with short tenure
        if curr_lvl > prev_lvl and prev_dur < 18 and curr_dur < 18:
            penalty += 2
        # Impossible jump
        if curr_lvl >= 4 and prev_lvl <= 2 and (prev_dur + curr_dur) < 24:
            penalty += 3
        # Earned promotion: long tenure → level up
        if curr_lvl > prev_lvl and prev_dur >= 24:
            reward += 2

    penalty = min(6, penalty)
    reward  = min(6, reward)
    score   = max(0, min(6, 3 + reward - penalty))   # base 3, adjusted by trajectory

    return TrajectoryFeatures(score=score, penalty=penalty, reward=reward)


def analyze_hiring_readiness(c: Candidate) -> HiringReadiness:
    """
    HiringReadinessAnalyzer v6.1 -- uses all 21 Redrob signals. Range: -3 to +8.

    Group 1: Availability     (-2 to +4)  -- notice, open_to_work, work_mode, relocation
    Group 2: Engagement       (-2 to +5)  -- activity, rr, response_time, applications
    Group 3: Trust & Demand   (-3 to +5)  -- interview, offer, saved, views, completeness, verification
    Group 4: Skill Validation ( 0 to +5)  -- assessments, github, endorsements, search_appearance
    Total clamped to -3..+8.
    """
    signals = c.signals

    # ── Group 1: Availability (-2 to +4) ────────────────────────────
    # 1. Notice period
    notice = int(signals.get("notice_period_days", 60) or 60)
    if   notice <= 15: notice_bonus = 2
    elif notice <= 30: notice_bonus = 1
    elif notice <= 60: notice_bonus = 0
    else:              notice_bonus = -1

    # 2. Open-to-work flag
    otw = bool(signals.get("open_to_work_flag", False))
    otw_pts = 1 if otw else 0

    # 3. Preferred work mode (founding team role = onsite/hybrid expected)
    work_mode = (signals.get("preferred_work_mode", "") or "").lower()
    if work_mode in ("onsite", "hybrid", "flexible"): work_mode_pts = 0
    elif work_mode == "remote":                        work_mode_pts = -1
    else:                                              work_mode_pts = 0

    # 4. Relocation / location fit
    location    = c.location.lower()
    country     = c.country.lower()
    willing     = bool(signals.get("willing_to_relocate", False))
    is_india    = country in ("india", "in") or "india" in location
    is_preferred = any(city in location for city in PREFERRED_LOCATIONS)
    if is_preferred:        relocation = 1
    elif is_india:          relocation = 0
    elif willing:           relocation = -1
    else:                   relocation = -1

    availability = notice_bonus + otw_pts + work_mode_pts + relocation  # -3 to +4

    # ── Group 2: Engagement (-2 to +5) ──────────────────────────────
    # 5. Last active date
    activity_score = 0
    last_dt = parse_date(signals.get("last_active_date", ""))
    if last_dt:
        days = (TODAY - last_dt).days
        if   days <=  3: activity_score = 2
        elif days <=  7: activity_score = 1
        elif days <= 30: activity_score = 0
        else:            activity_score = -1   # inactive > 30 days is a flag

    # 6. Recruiter response rate
    rr = float(signals.get("recruiter_response_rate", 0.0) or 0.0)
    if   rr >= 0.7: recruiter_response = 1
    elif rr >= 0.3: recruiter_response = 0
    else:           recruiter_response = -1    # <30% response is a real flag

    # 7. Average response time to recruiter messages
    avg_rt = float(signals.get("avg_response_time_hours", 48) or 48)
    response_time_pts = 1 if avg_rt <= 4 else 0   # responds same-day

    # 8. Applications submitted (actively job-seeking)
    apps = int(signals.get("applications_submitted_30d", 0) or 0)
    applications_pts = 1 if apps >= 3 else 0

    engagement = activity_score + recruiter_response + response_time_pts + applications_pts  # -2 to +5

    # ── Group 3: Trust & Market Demand (-3 to +5) ────────────────────
    # 9. Interview completion (ghost risk)
    icr_raw = signals.get("interview_completion_rate", None)
    icr = float(icr_raw) if icr_raw is not None else 1.0
    if   icr >= 0.8: interview_pts = 1
    elif icr >= 0.5: interview_pts = 0
    else:            interview_pts = -1   # high ghost risk

    # 10. Offer acceptance rate (seriousness)
    oar_raw = signals.get("offer_acceptance_rate", None)
    oar = float(oar_raw) if oar_raw is not None else -1.0
    if   oar >= 0.5: offer_pts = 1
    elif oar == -1:  offer_pts = 0    # no prior offers -- neutral
    elif oar >= 0.2: offer_pts = 0
    else:            offer_pts = -1   # accepts then backs out

    # 11. Saved by recruiters in last 30d (market demand)
    saved = int(signals.get("saved_by_recruiters_30d", 0) or 0)
    saved_pts = 1 if saved >= 5 else 0

    # 12. Profile views in last 30d (recruiter interest)
    views = int(signals.get("profile_views_received_30d", 0) or 0)
    views_pts = 1 if views >= 10 else 0

    # 13. Profile completeness (how seriously are they job-seeking?)
    completeness = int(signals.get("profile_completeness_score", 50) or 50)
    if   completeness >= 80: completeness_pts = 1
    elif completeness >= 40: completeness_pts = 0
    else:                    completeness_pts = -1

    # 14. Verification trust (email + phone + linkedin)
    v_email = bool(signals.get("verified_email",   False))
    v_phone = bool(signals.get("verified_phone",   False))
    v_li    = bool(signals.get("linkedin_connected", False))
    verified_count = sum([v_email, v_phone, v_li])
    if   verified_count >= 2: trust_pts = 1
    elif verified_count == 1: trust_pts = 0
    else:                     trust_pts = -1   # no verification at all

    trust_demand = interview_pts + offer_pts + saved_pts + views_pts + completeness_pts + trust_pts  # -3 to +5

    # ── Group 4: Skill Validation (0 to +5) ─────────────────────────
    # 15. Skill assessment scores (validated, not self-reported)
    assessments = signals.get("skill_assessment_scores", {}) or {}
    assessment_pts = 0
    if assessments:
        relevant = [
            float(v) for k, v in assessments.items()
            if TIER1_REGEX.search(re.sub(r"[-_/]", " ", k.lower()))
            or TIER2_REGEX.search(re.sub(r"[-_/]", " ", k.lower()))
        ]
        if relevant:
            avg = sum(relevant) / len(relevant)
            if   avg >= 80: assessment_pts = 2
            elif avg >= 60: assessment_pts = 1

    # 16. GitHub activity (positive use -- P9 penalty covers the negative side)
    github = int(signals.get("github_activity_score", 0) or 0)
    github_pts = 1 if github >= 50 else 0

    # 17. Endorsements received (peer-validated skills)
    endorsements = int(signals.get("endorsements_received", 0) or 0)
    endorsement_pts = 1 if endorsements >= 20 else 0

    # 18. Search appearance (how discoverable are they to recruiters?)
    search_app = int(signals.get("search_appearance_30d", 0) or 0)
    search_pts = 1 if search_app >= 20 else 0

    validation = assessment_pts + github_pts + endorsement_pts + search_pts  # 0 to +5

    # ── Total: clamp to -3..+8 ───────────────────────────────────────
    total = availability + engagement + trust_demand + validation
    total = max(-3, min(8, total))

    return HiringReadiness(
        score=total,
        notice_bonus=notice_bonus,
        otw_pts=otw_pts,
        work_mode_pts=work_mode_pts,
        relocation=relocation,
        activity_score=activity_score,
        recruiter_response=recruiter_response,
        response_time_pts=response_time_pts,
        applications_pts=applications_pts,
        interview_pts=interview_pts,
        offer_pts=offer_pts,
        saved_pts=saved_pts,
        views_pts=views_pts,
        completeness_pts=completeness_pts,
        trust_pts=trust_pts,
        assessment_pts=assessment_pts,
        github_pts=github_pts,
        endorsement_pts=endorsement_pts,
        search_pts=search_pts,
        open_to_work=otw,
    )



# ─────────────────────────────────────────────────────────────────────────────
def analyze_domain_tenure(c: Candidate) -> DomainTenure:
    """
    DomainTenureAnalyzer v6.2 -- 0-10 pts.

    The JD explicitly requires "3+ years specifically in retrieval, search,
    or recommendation systems." This analyzer counts months directly spent
    in those domains (job title + description), regardless of total YoE.

    Scoring:
      domain_years >= 4  -> 10   (above JD requirement)
      domain_years >= 3  ->  8   (meets JD minimum)
      domain_years >= 2  ->  6
      domain_years >= 1  ->  4
      domain_years >= 0.5->  2
      domain_years == 0  ->  0   (no domain history = significant gap)
    """
    domain_months = 0
    for job in c.career:
        t   = (job.get("title", "") or "").lower()
        d   = (job.get("description", "") or "").lower()[:600]
        td  = t + " " + d
        dur = max(1, int(job.get("duration_months", 1) or 1))
        if (SEARCH_RE.search(td)
                or RECOMMENDATION_RE.search(td)
                or any(kw in td for kw in _SPEC_RETRIEVAL_KW)):
            domain_months += dur

    domain_years = domain_months / 12.0

    # MASSIVE EMPHASIS ON DOMAIN TENURE
    if   domain_years >= 5:   score = 25
    elif domain_years >= 4:   score = 20
    elif domain_years >= 3:   score = 15
    elif domain_years >= 2:   score = 10
    elif domain_years >= 1:   score = 5
    elif domain_years >= 0.5: score = 2
    else:                     score = 0

    return DomainTenure(score=score, domain_months=domain_months, domain_years=domain_years)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — FEATURE VECTOR MERGER
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(c: Candidate) -> FeatureVector:
    """Run all analyzers → merge into FeatureVector."""
    career     = analyze_career(c)
    company    = analyze_company(c)
    skills     = analyze_skills(c)
    ev         = analyze_evidence(c)
    ownership  = analyze_ownership(c)
    impact     = analyze_impact(c)
    production = analyze_production(c)
    jd_intent  = analyze_jd_intent(c)
    evaluation = analyze_evaluation(c)
    trajectory    = analyze_trajectory(c)
    hiring        = analyze_hiring_readiness(c)
    domain_tenure = analyze_domain_tenure(c)

    # Merge all evidence for reasoning (sorted by priority, highest first)
    all_evidence = (
        career.evidence + company.evidence + skills.evidence +
        ev.evidence + ownership.evidence + impact.evidence +
        production.evidence + jd_intent.evidence + evaluation.evidence
    )
    all_evidence.sort(key=lambda e: e.priority, reverse=True)

    return FeatureVector(
        career=career, company=company, skills=skills,
        evidence=ev, ownership=ownership, impact=impact,
        production=production, jd_intent=jd_intent,
        evaluation=evaluation, trajectory=trajectory,
        hiring=hiring, domain_tenure=domain_tenure,
        all_evidence=all_evidence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — DEDICATED PENALTY STAGE  (Step 10: separated from positive analyzers)
# ─────────────────────────────────────────────────────────────────────────────

def compute_penalties(c: Candidate, fv: FeatureVector) -> int:
    """
    Dedicated penalty stage — subtracted from final score.

    Penalties are NOT mixed into positive analyzers.
    Each penalty represents an explicit JD disqualifier.
    """
    penalty = 0
    title = c.current_title.lower()

    # P1: Non-tech title (20 pts) — marketing/HR/finance with AI keywords
    if any(title == t or title.startswith(t + " ") or title.startswith(t + ",") for t in NON_TECH_TITLES):
        penalty += 100

    # P2: Junior title (8 pts) — JD needs 5-9 yr seniority
    if any(p in title for p in ("junior", "jr.", "intern", "trainee")) and "senior" not in title:
        penalty += 8

    # P3: Services-only career (6 pts) — JD explicitly warns about consulting
    if fv.company.consult_fraction > 0.90:
        penalty += 100
    elif fv.career.service_years > 0 and fv.career.product_years + fv.career.startup_years == 0:
        penalty += 6

    # P_DOMAIN: Wrong domain (CV/Speech/Robotics)
    if fv.skills.disq_fraction > 0.40:
        penalty += 100
    elif fv.skills.disq_fraction > 0.25 and fv.jd_intent.score < 10:
        penalty += 100
    elif fv.skills.disq_fraction > 0.25:
        penalty += 50
        
    # Broadened CV Title check
    cv_match = re.search(r'\b(computer vision|cv|vision|perception|image processing|robotics|speech)\b', title)
    if cv_match:
        if fv.jd_intent.score < 10:
            penalty += 100

    # P4: Research-only (8 pts) — JD: "tried it twice, didn't work"
    if fv.production.research_only:
        penalty += 100

    # P_PROD: No shipped production experience (ignoring generic title boost)
    if fv.production.ship_count == 0:
        penalty += 100
        
    # P_NO_DOMAIN_PROD: Lacking production retrieval/ranking systems
    if not fv.jd_intent.search_hit and not fv.jd_intent.recommendation_hit:
        if fv.production.score < 5:
            penalty += 100

    # P5: Prompt-engineer-only (10 pts) — JD: "If your experience consists of LangChain + OpenAI"
    skill_names = " ".join(s.get("name", "").lower() for s in c.skills)
    has_tier1 = fv.skills.tier1_count > 0
    llm_hype  = any(re.search(rf"\b{kw}\b", skill_names)
                     for kw in ("langchain", "prompt engineering", "openai", "chatgpt"))
    if not has_tier1 and llm_hype:
        penalty += 10

    # P6: Fabricated experience (10 pts) — claimed >> actual
    if c.years_of_experience > 3 and fv.career.actual_yoe > 0:
        if c.years_of_experience > fv.career.actual_yoe * 2.8:
            penalty += 10

    # P7: Job hopper (6 pts) — avg tenure < 18 months in recent jobs
    if len(c.career) >= 3:
        recent_jobs = c.career[:3]
        total_months = sum(max(1, int(j.get("duration_months", 1) or 1)) for j in recent_jobs)
        avg_tenure   = total_months / len(recent_jobs)
        curr_dur     = int(c.career[0].get("duration_months", 1) or 1) if c.career else 0
        if avg_tenure < 18 and curr_dur < 36:
            penalty += 6

    # P8: Research-heavy career (5 pts)
    if c.career:
        research_heavy = sum(
            1 for job in c.career
            if len(set(RESEARCH_HEAVY_RE.findall((job.get("description", "") or "").lower()))) >= 2
        )
        if research_heavy >= len(c.career) * 0.60:
            penalty += 5

    # P9: No GitHub (4 pts) — 5+ YoE, closed-source, not research
    github = int(c.signals.get("github_activity_score", 0) or 0)
    if fv.career.actual_yoe > 5 and github <= 0 and not fv.production.research_only:
        penalty += 4

    # P10: Trajectory penalty (from trajectory analyzer)
    penalty += fv.trajectory.penalty

    return penalty


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — TWO-STAGE SCORING ENGINE  (Step 1 + Step 3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_score(fv: FeatureVector, penalties: int) -> float:
    """
    Two-Stage Scored Pipeline:

    1. Technical Fit = sum of all technical component scores + synergy bonuses
    2. Hiring Readiness = capped behavioral score
    3. final = technical × 0.90 + hiring × 0.10 − penalties

    Notice period can NEVER dominate technical merit.
    Synergy bonuses reward JD-valued combinations.
    """

    # -- Technical components --
    technical = (
        fv.career.score           # 0-25
        + fv.jd_intent.score      # 0-24 (was 20, +4 for recency bonuses)
        + fv.production.score     # 0-18
        + fv.ownership.score      # 0-12
        + fv.impact.score         # 0-10
        + fv.domain_tenure.score  # 0-25  HEAVILY WEIGHTED: months in retrieval/search/recommendation
        + fv.evaluation.score     # 0-15
        + fv.company.score        # 0-6
        + fv.trajectory.score     # 0-6
        + fv.skills.score         # 0-1  (drastically reduced; experience matters more; skills confirm, don't substitute)
    )

    # ── Step 3: Synergy bonuses ───────────────────────────────────────
    synergy = 0

    # Retrieval/Search + Production → JD's dream candidate
    if fv.jd_intent.search_hit and fv.production.score >= 10:
        synergy += 8

    # Recommendation + Ownership (OWNER/LEAD) → independent builder
    if fv.jd_intent.recommendation_hit and fv.ownership.level in ("OWNER", "LEAD"):
        synergy += 5

    # Production + Evaluation → ships AND measures
    if fv.production.score >= 10 and fv.evaluation.has_eval:
        synergy += 4

    # Recommendation + Marketplace → Redrob's exact domain
    if fv.jd_intent.recommendation_hit and fv.jd_intent.marketplace_hit:
        synergy += 6

    # Search + Recommendation → full-stack retrieval engineer
    if fv.jd_intent.search_hit and fv.jd_intent.recommendation_hit:
        synergy += 5

    # Vector DB + Evaluation → modern retrieval with rigor
    if fv.jd_intent.vector_hit and fv.evaluation.has_eval:
        synergy += 3

    # Elite company + search exposure → proven at scale
    if fv.company.elite_company and fv.company.search_exposure:
        synergy += 3

    # === 3-WAY SYNERGY BONUSES (v6.2) ===
    # Search + Production + Evaluation = rare combination JD specifically values
    if (fv.jd_intent.search_hit and fv.production.score >= 10
            and fv.evaluation.has_eval):
        synergy += 6  # exceeds pairwise: the complete "ship + measure" profile

    # Domain Tenure + Search + Production = proven long-term domain contributor
    if (fv.domain_tenure.domain_years >= 2
            and fv.jd_intent.search_hit and fv.production.score >= 8):
        synergy += 5  # JD says "3+ yrs specifically in domain": long-term + shipping

    # Domain Tenure + Recommendation + Ownership = ran the system, not just worked on it
    if (fv.domain_tenure.domain_years >= 1.5
            and fv.jd_intent.recommendation_hit
            and fv.ownership.level in ("OWNER", "LEAD")):
        synergy += 4

        # Founder Mindset Bonus
    if fv.company.founder_mindset:
        synergy += 6  # Substantial reward for 0->1 builders
        
    # Extra boost for Eval + Production (make it huge)
    if fv.production.score >= 10 and fv.evaluation.has_eval:
        synergy += 4  # Total +8 synergy for this pair now!

    technical += synergy

    # ── Hiring Readiness ──────────────────────────────────────────────
    hiring = fv.hiring.score     # -3 to +8

    # -- Final: capped technical + scaled hiring - penalties ----------
    # Cap technical at 95 so hiring readiness is always visible even for
    # the strongest candidates. Hiring: -3..+8 maps to -1.875..+5.0.
    # Maximum possible: 95 + 5 - 0 = 100 (perfect technical + perfect hiring).
    # -- Technical cap: 95.
    # We added domain_tenure (+10) and JD recency (+4) to the technical scale.
    # Raw technical scores can now reach ~150. Multiplier adjusted to 0.75 so only 
    # the top 1% (score > 126) hit the technical ceiling of 95.0. 
    # This ensures hiring readiness remains visible at the top end.
    tech_contrib   = min(95.0, technical * 0.60)
    hiring_contrib = (hiring / 8.0) * 5.0    # -1.875 to +5.0

    final = tech_contrib + hiring_contrib - penalties

    return max(0.0, min(100.0, final))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 7 — EVIDENCE-DRIVEN REASONING  (Step 11)
# ─────────────────────────────────────────────────────────────────────────────

_EVIDENCE_JD_MAP = {
    "retrieval":       "aligned with Redrob's core retrieval/ranking requirement",
    "search":          "aligned with Redrob's search infrastructure needs",
    "recommendation":  "relevant to Redrob's recommendation/matching platform",
    "marketplace":     "directly relevant to Redrob's marketplace product",
    "production":      "demonstrates shipping ML systems (JD's 'shipper > researcher' criterion)",
    "ownership":       "shows independent ownership of technical systems",
    "impact":          "contains measurable outcomes demonstrating real-world impact",
    "evaluation":      "shows evaluation methodology experience (NDCG, A/B testing)",
    "career":          "demonstrates strong career alignment with JD requirements",
    "company":         "includes experience at relevant company type",
    "skill":           "confirms core technical skill match",
}


def generate_reasoning(
    fv: FeatureVector,
    c: Candidate,
    final_score: float,
    penalties: int,
) -> tuple[str, set[str]]:
    """
    Evidence-driven reasoning v6.0.

    1. Pick top 3 evidence items (by priority)
    2. For each, write one sentence connecting it to the JD requirement
    3. Mention the single biggest gap
    """
    # ── Categorise skills for diversity tracking ──────────────────────
    category_set = set()
    for s_obj in c.skills:
        name = s_obj.get("name", "") or ""
        if len(name) < 2:
            continue
        cat = _categorise_skill(name)
        if cat:
            category_set.add(cat)
    if not category_set:
        category_set.add("BALANCED")

    # ── Build evidence-driven reasoning ───────────────────────────────
    parts = []
    yoe     = c.years_of_experience
    title   = c.current_title or "Engineer"
    company = c.current_company
    notice  = fv.hiring.notice_bonus

    # Stable variation seed
    seed = int(hashlib.blake2s(c.id.encode("utf-8"), digest_size=4).hexdigest(), 16)

    # Top evidence items (already sorted by priority in extract_features)
    top_evidence = fv.all_evidence[:5]  # take up to 5 for selection

    # Deduplicate by category — keep highest-priority per category
    seen_cats = set()
    unique_evidence = []
    for ev in top_evidence:
        if ev.category not in seen_cats:
            seen_cats.add(ev.category)
            unique_evidence.append(ev)
    unique_evidence = unique_evidence[:3]

    if unique_evidence:
        # Opening sentence with candidate context
        company_str = f" at {company}" if company else ""
        loc_str     = f" based in {c.location}" if c.location else ""

        if final_score >= 60:
            openers = [
                f"Strong match: {title}{company_str} with {yoe:g} years of experience{loc_str}.",
                f"{title}{company_str} ({yoe:g} years{loc_str}) — strong alignment with JD.",
                f"Top candidate: {yoe:g}-year {title}{company_str}{loc_str}.",
            ]
        elif final_score >= 35:
            openers = [
                f"Moderate match: {title}{company_str} with {yoe:g} years{loc_str}.",
                f"{title} ({yoe:g} years){company_str} — partial alignment.",
                f"Candidate with {yoe:g} years as {title}{company_str}{loc_str}.",
            ]
        else:
            openers = [
                f"Weak match: {title} with {yoe:g} years experience.",
                f"Limited alignment: {yoe:g}-year {title}{company_str}.",
                f"Below threshold: {title}{company_str} ({yoe:g} years).",
            ]
        parts.append(openers[seed % len(openers)])

        # Evidence sentences — each maps to a JD requirement
        for ev in unique_evidence:
            jd_link = _EVIDENCE_JD_MAP.get(ev.category, "relevant to the role")
            # Clean up the sentence — truncate at word boundary
            sentence = ev.sentence.strip()
            if len(sentence) > 180:
                sentence = sentence[:180].rsplit(" ", 1)[0]
            if sentence and not sentence.endswith("."):
                sentence += "."
            if len(sentence) > 10:
                parts.append(f"{sentence[0].upper()}{sentence[1:]} — {jd_link}.")
            else:
                parts.append(f"Evidence of {ev.category} experience — {jd_link}.")
    else:
        # No evidence extracted — use minimal factual statement
        parts.append(f"{title} with {yoe:g} years experience. No strong retrieval/search evidence found in profile.")

    # ── Biggest gap ───────────────────────────────────────────────────
    gaps = []
    if not fv.jd_intent.search_hit and not fv.jd_intent.recommendation_hit:
        gaps.append("no retrieval/search/recommendation evidence in profile")
    if not fv.evaluation.has_eval:
        gaps.append("no evaluation methodology experience (NDCG, A/B testing)")
    if fv.production.score < 5:
        gaps.append("limited production deployment evidence")
    if fv.company.consult_fraction > 0.70:
        gaps.append("primarily consulting/services background")
    if fv.ownership.level in ("SUPPORT", "UNKNOWN"):
        gaps.append("profile language suggests peripheral involvement rather than ownership")
    if yoe < 4:
        gaps.append(f"only {yoe:g} years experience (JD targets 5-9)")
    elif yoe > 12:
        gaps.append(f"{yoe:g} years may be over-senior for founding team IC role")
    notice_days = int(c.signals.get("notice_period_days", 60) or 60)
    if notice_days > 90:
        gaps.append(f"{notice_days}-day notice period")

    if gaps and final_score < 80:
        gap_prefix = ["Gap: ", "However, ", "Concern: ", "Note: "][seed % 4]
        parts.append(gap_prefix + gaps[0] + ".")
        
    # Hard Rejection override
    if final_score == 0.0 and penalties >= 100:
        if "no production" in gaps[0] or fv.production.score == 0:
            parts = ["Hard Rejection: No evidence of shipping to production."]
        elif fv.production.research_only:
            parts = ["Hard Rejection: Pure research background."]
        elif fv.company.consult_fraction > 0.90:
            parts = ["Hard Rejection: Almost entirely consulting background."]
        elif any(title == t or title.startswith(t + " ") or title.startswith(t + ",") for t in NON_TECH_TITLES):
            parts = [f"Hard Rejection: Non-technical role ({title})."]
        elif fv.production.ship_count == 0:
            parts = ["Hard Rejection: No evidence of shipping ML systems to production."]
        elif not fv.jd_intent.search_hit and not fv.jd_intent.recommendation_hit and fv.production.score < 5:
            parts = ["Hard Rejection: Lacks required production retrieval/ranking domain experience."]
        elif re.search(r'\b(computer vision|cv|vision|perception|image processing|robotics|speech)\b', title) or fv.skills.disq_fraction > 0.25:
            parts = ["Hard Rejection: CV/Speech specialist without core retrieval experience."]
        else:
            parts = ["Hard Rejection: Disqualified due to massive penalties."]

    # ── Quick-join indicator ──────────────────────────────────────────
    if notice_days <= 30 and fv.hiring.open_to_work and final_score >= 40:
        parts.append(["Available quickly.", "Ready to interview.", "Immediate joiner."][seed % 3])

    return " ".join(parts), category_set


# ─────────────────────────────────────────────────────────────────────────────
# DEEP SCORER  (Pass 2 — called only on 12K shortlist)
# ─────────────────────────────────────────────────────────────────────────────

def deep_score(raw: dict) -> tuple[float, str, set[str]]:
    """
    Full v6.0 pipeline:
      is_honeypot(raw)           → early exit
      normalize(raw)             → Candidate
      extract_features(c)        → FeatureVector
      compute_penalties(c, fv)   → penalties (dedicated stage)
      compute_score(fv, pen)     → final score (two-stage + synergy)
      generate_reasoning(fv,c,s) → evidence-driven text
    """
    if is_honeypot(raw):
        return 0.0, (
            "Flagged as honeypot: internally inconsistent profile data "
            "(impossible tenure, inflated experience, or expert/zero-duration skills)."
        ), {"HONEYPOT"}

    c  = normalize(raw)
    fv = extract_features(c)
    penalties = compute_penalties(c, fv)
    final = compute_score(fv, penalties)

    reasoning, category = generate_reasoning(fv, c, final, penalties)

    return final, reasoning, category


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE ITERATOR  [B2][B8] — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

def iter_candidates(input_path: str):
    path = Path(input_path)
    if not path.exists():
        print(f"ERROR: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8") as f:
        first_chunk = f.read(16)

    is_array = first_chunk.lstrip().startswith("[")

    with opener(path, "rt", encoding="utf-8") as f:
        if is_array:
            try:
                data = json.load(f)
                yield from (d for d in data if isinstance(d, dict))
            except json.JSONDecodeError as e:
                print(f"ERROR: JSON parse failed: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                except json.JSONDecodeError:
                    if lineno <= 5:
                        print(f"[WARN] Bad JSON at line {lineno}: {line[:80]}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# TWO-PASS RANKING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def rank_candidates(input_path: str, output_path: str) -> None:
    t0 = time.time()
    print(f"[Ranker v6.0]  Input : {input_path}")
    print(f"[Ranker v6.0]  Output: {output_path}")
    print(f"[Ranker v6.0]  TODAY : {TODAY}\n")

    # ── PASS 1 ────────────────────────────────────────────────────────
    print(f"\n[Pass 1] Fast filtering (streaming)...")
    fast_pool: list[tuple[float, int, int, str, dict]] = []
    total = 0
    for candidate in iter_candidates(input_path):
        cid = candidate.get("candidate_id", "")
        if not cid:
            continue
        fs    = fast_score(candidate)
        total += 1
        entry = (fs, -candidate_id_num(cid), -total, cid, candidate)

        if len(fast_pool) < PASS2_POOL_SIZE:
            heapq.heappush(fast_pool, entry)
        elif entry > fast_pool[0]:
            heapq.heapreplace(fast_pool, entry)
        if total % 10_000 == 0:
            elapsed = time.time() - t0
            print(f"  {total:,} scanned  ({elapsed:.1f}s)", flush=True)

    print(f"[Pass 1] Done: {total:,} candidates scanned in {time.time()-t0:.1f}s")
    pool = [(s, cid, c) for s, _, _, cid, c in sorted(fast_pool, key=lambda x: (-x[0], x[3]))]
    print(f"[Pass 1] Top {len(pool)} candidates passed to Pass 2 (dropped {total - len(pool)})")

    # ── PASS 2 ────────────────────────────────────────────────────────
    print(f"\n[Pass 2] Deep scoring top {len(pool)} candidates...")
    deep_results: list[tuple[str, float, str, set[str]]] = []

    for i, (_, cid, c) in enumerate(pool):
        score, reasoning, category = deep_score(c)
        deep_results.append((cid, score, reasoning, category))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pool)} scored  ({time.time()-t0:.1f}s)", flush=True)

    print(f"[Pass 2] Done in {time.time()-t0:.1f}s")

    deep_results.sort(key=lambda x: (-x[1], x[0]))
    top_100 = deep_results[:100]

    hp_count = sum(1 for _, _, _, cat in top_100 if "HONEYPOT" in cat)
    hp_rate  = hp_count / max(len(top_100), 1)
    flag     = "[WARNING] EXCEEDS 10% THRESHOLD" if hp_rate > 0.10 else "[OK]"
    print(f"\n[Audit]  Honeypots in top 100: {hp_count} ({hp_rate:.0%})  {flag}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score, reasoning, cat) in enumerate(top_100, start=1):
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])

    elapsed = time.time() - t0
    print(f"\n[Done]  {out_path}  written in {elapsed:.1f}s")
    budget  = "[OK] WITHIN BUDGET" if elapsed < 300 else "[WARNING] OVER 5-MIN BUDGET"
    print(f"        {budget}")

    print(f"\n{'Rank':>4}  {'Candidate ID':16}  {'Score':>7}  Reasoning (preview 80 chars)")
    print("-" * 115)
    for rank, (cid, score, reasoning, cat) in enumerate(top_100[:10], start=1):
        print(f"{rank:>4}  {cid:16}  {score:>7.2f}  {reasoning[:80]}")
    print(f"\nScore range: {top_100[-1][1]:.2f} – {top_100[0][1]:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Bug Hunters - Redrob Hackathon Ranker v6.0\n"
                    f"Two-pass: 100K stream → fast filter (top {PASS2_POOL_SIZE}) → deep score → CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl, .jsonl.gz, or sample_candidates.json")
    parser.add_argument("--out", required=True,
                        help="Output CSV path (e.g. bug_hunters.csv)")
    args = parser.parse_args()
    rank_candidates(args.candidates, args.out)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# CHANGELOG
# ─────────────────────────────────────────────────────────────────────────────
# v5.0 → v6.0  (Two-Stage Scored Pipeline)
#
# [V6-1]  Split scoring: Technical Fit (90%) + Hiring Readiness (10%)
#         → notice period can NEVER dominate technical merit
# [V6-2]  Notice period capped: -1 to +2 (was 0.0-1.0 with heavy influence)
# [V6-3]  Synergy bonuses: retrieval+production=+8, rec+marketplace=+6, etc.
# [V6-4]  JDIntentAnalyzer: semantic buckets (RECOMMENDATION, SEARCH, MARKETPLACE)
#         replace individual keyword counting
# [V6-5]  CompanyAnalyzer: returns typed profile (company_type, search_exposure,
#         ranking_exposure, startup) not just a float score
# [V6-6]  EvidenceAnalyzer: extracts Evidence(category, sentence) objects with
#         provenance — same objects drive both scoring and reasoning
# [V6-7]  OwnershipAnalyzer: OWNER/LEAD/CONTRIBUTOR/SUPPORT classification
#         instead of single float ratio
# [V6-8]  CareerAnalyzer: adds specialization tracking (RETRIEVAL/ML/BACKEND/DATA),
#         product_years, startup_years, service_years, career_depth
# [V6-9]  ImpactAnalyzer: extracts Impact(metric, improvement) objects
# [V6-10] Dedicated penalty stage: compute_penalties() separated from positive
#         analyzers, integer penalties (0-50)
# [V6-11] Evidence-driven reasoning: top-3 evidence items → JD mapping → gap
# [V6-12] Integer score ranges: Career 0-25, JD Intent 0-20, Production 0-18,
#         Ownership 0-12, Impact 0-10, Eval 0-8, Company 0-6, Trajectory 0-6,
#         Skills 0-5, Hiring -3 to +8
# [V6-13] New EvalFeatures split from JDIntentFeatures (0-8)
# [V6-14] New TrajectoryFeatures split from CareerFeatures (0-6)
# [V6-15] Removed legacy functions: score_skill_fit, score_product_fit,
#         score_behavioral, score_experience_fit, score_location,
#         score_career_trajectory, _career_score
# [V6-16] ConfidenceAnalyzer absorbed into CareerAnalyzer as career_depth
# [V6-17] All v4.1/v5.0 bug fixes [B1]-[B10] preserved
# [V6-18] fast_score, is_honeypot, iter_candidates, normalize unchanged
# [V6-19] All constants, frozensets, regexes preserved exactly
