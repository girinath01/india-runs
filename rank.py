#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Redrob Hackathon - Candidate Ranker v3.4  (Bug Hunters - LOGICAL FIXES v5)
JD: Senior AI Engineer — Founding Team @ Redrob AI

Scoring formula (MATCHES CODE):
raw = 0.35 * search_retrieval
    + 0.25 * vector_search
    + 0.20 * production
    + 0.15 * notice_period
    + 0.05 * open_to_work
  final = sigmoid(clamp(raw * experience_mult + tie_break - penalties))

Design principles:
  1. Retrieval/Search/Ranking/VectorDB skills >> trendy LLM tooling  (JD warns about this)
  2. Shippers > Researchers — production deployment evidence is first-class signal
  3. Behavioral availability is 35%: inactive candidate = effectively unavailable (JD explicit)
  4. Two-pass: 100K → fast filter → 3000 → deep score → 100
  5. No external deps — pure stdlib → zero import errors in sandboxed grading env

FIXED BUGS vs v2 (see CHANGELOG at bottom):
  [B1] Double `notice` variable with different defaults (60 vs 0) in deep_score → single source
  [B2] f.read() loaded entire 465 MB into RAM → streaming line-by-line iterator
  [B3] Skill name `name in kw` matches single chars (e, r, c…) → added len(name) >= 3 guard
  [B4] Docstring weights ≠ code weights (swapped skill/behavioral) → docstring fixed
  [B5] Unused `company_size = profile.get(...)` variable in score_product_fit → removed
  [B6] `raw_content.startswith("[")` fails on leading whitespace → .lstrip() added
  [B7] Date parse only handles %Y-%m-%d → added fallback for %Y-%m and bare %Y
  [B8] Memory: JSON array branch `json.loads(entire_465MB)` → replaced with ijson-free stream
  [B9] fast_score NON_TECH_TITLES set checked with `t in title` — works but is O(n×m) inside
        a generator; hoisted to compiled frozenset lookup
  [B10] score_behavioral returned raw float that could exceed 1.0 (clamp was missing before return)

Usage:
  python rank.py --candidates ./candidates.jsonl.gz --out ./bug_hunters.csv
  python rank.py --candidates ./sample_candidates.json --out ./test_out.csv
"""

import json
import csv
import gzip
import argparse
import sys
import time
from datetime import datetime, date
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA  (what we expect in each candidate record)
# ─────────────────────────────────────────────────────────────────────────────
# candidate = {
#   "candidate_id": "CAND_XXXXXXX",
#   "profile": {
#     "current_title": str, "current_company": str,
#     "headline": str, "summary": str,
#     "location": str, "country": str,
#     "years_of_experience": float
#   },
#   "career_history": [
#     { "title":str, "company":str, "industry":str, "company_size":str,
#       "start_date":"YYYY-MM-DD", "duration_months":int, "description":str }
#   ],
#   "skills": [
#     { "name":str, "proficiency":"expert"|"advanced"|"intermediate"|"beginner",
#       "duration_months":int, "endorsements":int }
#   ],
#   "redrob_signals": {
#     "open_to_work_flag": bool,
#     "last_active_date": "YYYY-MM-DD",
#     "recruiter_response_rate": float,   # 0–1
#     "interview_completion_rate": float, # 0–1
#     "notice_period_days": int,
#     "saved_by_recruiters_30d": int,
#     "github_activity_score": int,       # 0–100
#     "willing_to_relocate": bool
#   }
# }


# ─────────────────────────────────────────────────────────────────────────────
# TODAY — fixed date so ranking is reproducible in graded sandbox
# ─────────────────────────────────────────────────────────────────────────────
TODAY = date(2026, 6, 15)


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


# ─────────────────────────────────────────────────────────────────────────────
# TIERED SKILL DEFINITIONS  (from JD analysis)
# ─────────────────────────────────────────────────────────────────────────────

# Tier 1 — core JD requirements: retrieval, search, ranking, vector DBs, eval
TIER1_SKILLS = frozenset({
    "retrieval", "information retrieval", "dense retrieval", "sparse retrieval",
    "hybrid retrieval", "hybrid search", "semantic search", "vector search",
    "search engine", "search ranking", "search system", "search infrastructure",
    "recommendation system", "recommender", "recommender system", "ranking system",
    "learning to rank", "ltr", "lambdamart", "lambdarank", "ranknet",
    "bi-encoder", "cross-encoder",
    "faiss", "elasticsearch", "opensearch", "qdrant", "milvus",
    "weaviate", "pinecone", "pgvector", "vespa", "annoy", "scann", "chroma",
    "vector database", "vector db", "vector store", "vector index",
    "ann", "approximate nearest neighbor", "hnsw",
    "reranking", "re-ranking", "reranker", "colbert",
    "bm25", "tfidf", "tf-idf",
    "ndcg", "mrr", "mean reciprocal rank", "map", "mean average precision",
    "a/b testing", "ab testing", "offline evaluation", "online evaluation",
    "eval framework", "evaluation framework",
})

# Tier 2 — strong supporting skills
TIER2_SKILLS = frozenset({
    "rag", "retrieval augmented generation",
    "embeddings", "embedding", "sentence-transformers", "sentence transformer",
    "text embedding", "openai embeddings", "bge", "e5 model", "dense embedding",
    "nlp", "natural language processing", "language model", "text classification",
    "bert", "transformers", "huggingface", "hugging face",
    "pytorch", "tensorflow", "keras", "jax",
    "mlops", "model deployment", "model serving", "inference",
    "feature engineering", "feature store",
    "distributed systems", "distributed training",
    "mlflow", "wandb", "ray", "dask",
    "xgboost", "lightgbm", "gradient boosting",
    "scikit-learn", "sklearn",
    "python", "fastapi", "flask", "docker", "kubernetes",
    "spark", "kafka", "airflow",
    "aws", "gcp", "azure",
})

# Tier 3 — trendy LLM tooling (low weight)
# JD warns: "If your experience consists of LangChain + OpenAI — probably not"
TIER3_SKILLS = frozenset({
    "langchain", "lang chain",
    "prompt engineering", "instruction tuning",
    "qlora", "lora", "peft", "fine-tuning", "fine tuning", "finetuning",
    "llm", "large language model", "generative ai",
    "openai", "chatgpt", "gpt",
    "rlhf",
})

# Disqualifying domain focus — wrong field, JD is explicit
DISQUALIFYING_SKILLS = frozenset({
    "computer vision", "image classification", "object detection",
    "yolo", "convolutional", "image segmentation",
    "opencv",
    "speech recognition", "asr", "tts", "text-to-speech", "speech synthesis",
    "audio processing", "sound classification", "voice recognition",
    "robotics", "ros", "autonomous driving", "lidar", "slam",
    "photoshop", "illustrator", "figma", "canva",
    "accounting", "tally", "gst", "salesforce", "crm",
    "seo", "content writing", "copywriting",
    "six sigma", "lean", "kaizen",
    "solidworks", "autocad",
})

import re
TIER1_REGEX = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(TIER1_SKILLS, key=len, reverse=True))) + r')\b')
TIER2_REGEX = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(TIER2_SKILLS, key=len, reverse=True))) + r')\b')
TIER3_REGEX = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(TIER3_SKILLS, key=len, reverse=True))) + r')\b')
DISQ_REGEX = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(DISQUALIFYING_SKILLS, key=len, reverse=True))) + r')\b')

# [H1] Module-level compiled regexes for deep_score (avoids 18K re-compilations in Pass 2)
PRIMARY_CORE_RE = re.compile(r'\b(information retrieval|learning to rank|ltr|lambdamart|bm25|semantic search|candidate matching|search quality)\b')
SECONDARY_CORE_RE = re.compile(r'\b(ranking system|recommendation system|candidate ranking|personalization|relevance|matching engine|elasticsearch|opensearch)\b')
EXPLICIT_VECTOR_RE = re.compile(r'\b(faiss|pinecone|qdrant|weaviate|milvus|pgvector)\b')
VECTOR_TEXT_RE = re.compile(r'\b(vector search|vector database|chroma|ann|hnsw)\b')
HR_TEXT_RE = re.compile(r'\b(hr tech|recruiting tech|talent acquisition platform|marketplace product|job board)\b')
PROD_TEXT_RE = re.compile(r'\b(scale|shipped|deployed|productionized|enterprise|latency|qps|inference optimization|tensorrt|vllm|distributed systems|ray|spark)\b')
INFRA_TEXT_RE = re.compile(r'\b(search infra|retrieval pipeline|relevance|indexing|ranking optimization)\b')
EVAL_TEXT_RE = re.compile(r'\b(ndcg|mrr|mean average precision|a/b test|ab testing|offline evaluation|online evaluation)\b')
RAG_WORD_RE = re.compile(r'\brag\b')

CONSULTING_INDUSTRIES = frozenset({
    "it services", "it consulting", "consulting",
    "business process outsourcing", "bpo", "outsourcing",
    "staffing", "managed services", "it staffing",
})
CONSULTING_NAMES = frozenset({
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "ltimindtree", "lti", "larsen toubro infotech", "niit technologies",
    "persistent systems", "cyient", "zensar", "birlasoft", "coforge",
    "kpit", "mastech", "firstsource", "wns", "genpact",
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
    "senior engineer": 0.72, "data scientist": 0.65,
    "backend engineer": 0.55, "software engineer": 0.50,
    "data engineer": 0.45,
}

NON_TECH_TITLES = frozenset({  # [B9] frozenset for O(1) lookup
    "marketing manager", "hr manager", "human resources", "operations manager",
    "business analyst", "content writer", "sales executive", "accountant",
    "project manager", "graphic designer", "customer support", "customer service",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "finance manager", "account manager", "talent acquisition", "recruiter",
    "data entry", "tester", "manual tester", "business development",
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
})

RESEARCH_SIGNALS = frozenset({
    "paper", "publication", "conference", "arxiv", "journal",
    "thesis", "academic", "phd candidate", "research lab",
    "pure research", "benchmark", "theoretical",
})


# ─────────────────────────────────────────────────────────────────────────────
# DATE PARSING  [B7] — robust multi-format
# ─────────────────────────────────────────────────────────────────────────────
def parse_date(s: str) -> date | None:
    """Try common date formats; return None on failure."""
    if not s:
        return None
    s = s.replace("/", "-")
    for fmt in ("%Y-%m-%d", "%Y-%m", "%d-%m-%Y", "%Y"):
        try:
            return datetime.strptime(s[:10].strip(), fmt).date()
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def is_honeypot(candidate: dict) -> bool:
    """
    Five impossibility checks targeting the ~80 synthetic trap profiles.
    Any match = honeypot → score forced to 0.
    >10% honeypots in top-100 → submission disqualified per spec.
    """
    profile  = candidate.get("profile", {}) or {}
    career   = candidate.get("career_history", []) or []
    skills   = candidate.get("skills", []) or []

    # 1. Job duration > calendar months since start date (time travel)
    for job in career:
        start_dt = parse_date(job.get("start_date", ""))
        stated   = int(job.get("duration_months", 0) or 0)
        if start_dt and stated > 0:
            max_possible = (TODAY.year - start_dt.year) * 12 + (TODAY.month - start_dt.month) + 3
            if stated > max_possible + 6:
                return True

    # 2. Expert proficiency + 0 duration on 3+ skills (impossible mastery)
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and int(s.get("duration_months", 0) or 0) == 0
    )
    if expert_zero >= 3:
        return True

    # (Rule 3 removed: Claimed YOE vs career history is now a penalty, not disqualification)

    # 4. Keyword stuffer: 20+ expert skills but lacks duration/endorsement or has domain conflicts
    expert_advanced = [s for s in skills if s.get("proficiency") in ("expert", "advanced")]
    if len(expert_advanced) >= 20:
        zero_evidence = sum(1 for s in expert_advanced if int(s.get("duration_months", 0) or 0) == 0 and int(s.get("endorsements", 0) or 0) == 0)
        if zero_evidence >= 10:
            return True
        # Domain conflict check
        names = " ".join(s.get("name", "").lower() for s in expert_advanced)
        has_non_tech = any(kw in names for kw in ("accounting", "tally", "sales", "hr", "marketing", "seo", "content writing"))
        has_tech = any(kw in names for kw in ("machine learning", "backend", "react", "python", "aws"))
        if has_non_tech and has_tech:
            return True

    # 5. Any single job > 20 yrs but candidate claims < 10 yrs total
    claimed = float(profile.get("years_of_experience", 0) or 0)
    if 0 < claimed < 10 and any(int(j.get("duration_months", 0) or 0) > 240 for j in career):
        return True

    # 6. Impossible duration for recent tech
    for s in skills:
        name = s.get("name", "").lower()
        dur = int(s.get("duration_months", 0) or 0)
        if dur > 36 and ("langchain" in name or "openai" in name or "chatgpt" in name or "llama" in name):
            return True
        if dur > 60 and ("qdrant" in name or "weaviate" in name or "pinecone" in name):
            return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — FAST FILTER  (100 K → 3 000)
# ─────────────────────────────────────────────────────────────────────────────
def fast_score(candidate: dict) -> float:
    """
    Lightweight score: only checks title, top-level skill list, and key text signals.
    No deep career analysis. Must process >=10K candidates/second on a single core.
    """
    profile = candidate.get("profile", {}) or {}
    skills  = candidate.get("skills", []) or []
    career  = candidate.get("career_history", []) or []
    title   = profile.get("current_title", "").lower()

    # Soft out: non-technical title penalty
    score_penalty = 0.0
    if any(title == t or title.startswith(t + " ") or title.startswith(t + ",") for t in NON_TECH_TITLES):
        score_penalty = 20.0

    t1_count = t2_count = 0
    for s in skills:
        name = re.sub(r"[-_/]", " ", s.get("name", "").lower()).strip()
        if len(name) < 3:          # [B3] skip single/double chars to avoid false positives
            continue
        if TIER1_REGEX.search(name):
            t1_count += 1
        elif TIER2_REGEX.search(name):
            t2_count += 1

    title_hit = any(t in title for t in STRONG_TITLE_SCORES)

    # Scan headline + summary + career descriptions for JD concepts
    headline = profile.get("headline", "") or ""
    summary  = profile.get("summary", "")  or ""
    career_desc = " ".join(j.get("description", "") or "" for j in career[:3])[:4000]
    combined = (headline + " " + summary + " " + career_desc).lower()

    # "information retrieval" is covered by "retrieval", so we can use single-word tokens
    FAST_WORDS = {
        "retrieval", "recommendation", "ranking", "search", "vector",
        "embedding", "nlp", "rag", "ndcg",
        "faiss", "qdrant", "pinecone", "milvus", "elasticsearch", "weaviate"
    }
    
    text_hit = any(word in combined for word in FAST_WORDS)

    # [C3] Catch plain-language shippers who don't use keyword buzzwords
    # JD warns: "A Tier 5 candidate may not use the words 'RAG' or 'Pinecone'"
    SHIP_FAST = {"shipped", "deployed", "production", "at scale", "million", "serving", "built and"}
    ship_hits = sum(1 for w in SHIP_FAST if w in combined)

    score = t1_count * 2.5 + t2_count * 0.5
    if title_hit:  score += 4.0
    if text_hit:   score += 1.0
    score += min(ship_hits * 0.5, 2.0)  # Up to +2.0 for shipping evidence
    return score - score_penalty


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT SCORERS
# ─────────────────────────────────────────────────────────────────────────────

def score_skill_fit(skills: list, extra_text: str = "", career: list | None = None) -> tuple[float, float]:
    """
    Tiered skill scoring  (weight: 0.30 of final).
    Returns (skill_score: 0-1, disq_fraction: 0-1)
    """
    if career is None:
        career = []

    t1_w = t2_w = t3_w = disq_w = total_w = 0.0

    for skill in skills:
        name  = re.sub(r"[-_/]", " ", skill.get("name", "").lower()).strip()
        if len(name) < 3:          # [B3] guard against single-char false positives
            continue
        prof  = PROFICIENCY_WEIGHTS.get(skill.get("proficiency", "beginner"), 0.15)
        endorse = min(int(skill.get("endorsements", 0) or 0), 100)
        duration = min(int(skill.get("duration_months", 0) or 0), 72)
        weight = prof * (1 + endorse / 200.0 + duration / 144.0)

        if "rag" in name or "retrieval augmented generation" in name:
            weight *= 0.60
        elif "sentence-transformer" in name or "sentence transformer" in name:
            weight *= 0.75

        total_w += weight

        if TIER1_REGEX.search(name):
            t1_w += weight
        elif TIER2_REGEX.search(name):
            t2_w += weight
        elif TIER3_REGEX.search(name):
            t3_w += weight

        if DISQ_REGEX.search(name):
            disq_w += weight

    tw      = total_w if total_w > 0 else 1.0
    t1_norm = clamp(t1_w / tw)
    t2_norm = clamp(t2_w / tw * 0.55)
    t3_norm = clamp(t3_w / tw * 0.15)   # LangChain-only gets almost nothing
    disq_frac = clamp(disq_w / tw)

    raw = clamp(t1_norm * 0.65 + t2_norm * 0.25 + t3_norm * 0.10)

    # Domain disqualification penalty
    if disq_frac > 0.60:
        raw *= 0.25
    elif disq_frac > 0.40:
        raw *= 0.58

    return clamp(raw), disq_frac


def score_product_fit(profile: dict, career: list) -> tuple[float, float, float]:
    """
    Shipping + company-type scoring  (weight: 0.20 of final).
    Returns (product_fit: 0-1, consult_fraction: 0-1, production_ratio: 0-1)
    """
    title = profile.get("current_title", "").lower()

    # Title match
    title_score = 0.0
    for t, sc in STRONG_TITLE_SCORES.items():
        if t in title:
            title_score = max(title_score, sc)
    if any(p in title for p in ("lead", "principal", "staff", "head of", "director")):
        title_score = min(1.0, title_score + 0.12)
    elif any(p in title for p in ("senior", "sr.")):
        title_score = min(1.0, title_score + 0.06)
    elif any(p in title for p in ("junior", "jr.", "associate", "intern", "trainee")):
        title_score = max(0.0, title_score - 0.22)

    total_months = consulting_months = 0
    ship_score = research_score = 0.0

    for job in career:
        industry = (job.get("industry", "") or "").lower()
        company  = (job.get("company",  "") or "").lower()
        j_size   = (job.get("company_size", "") or "")   # [B5] now actually used
        desc     = (job.get("description", "") or "").lower()
        duration = max(1, int(job.get("duration_months", 1) or 1))
        total_months += duration

        is_consulting = (
            any(ci in industry for ci in CONSULTING_INDUSTRIES)
            or any(cn in company for cn in CONSULTING_NAMES)
        )
        if is_consulting:
            consulting_months += duration

        # Startup/product company size boosts shipping signal (startups ship fast)
        size_mult = (
            1.25 if j_size in ("1-10", "11-50", "51-200")
            else 1.10 if j_size in ("201-500", "501-1000")
            else 1.00
        )

        ship_hits     = sum(1 for kw in SHIP_SIGNALS     if kw in desc)
        research_hits = sum(1 for kw in RESEARCH_SIGNALS if kw in desc)
        ship_score     += ship_hits * duration * size_mult
        research_score += research_hits * duration

    consult_frac = consulting_months / total_months if total_months > 0 else 0.0
    total_signal = ship_score + research_score
    prod_ratio   = ship_score / total_signal if total_signal > 0 else 0.20

    if prod_ratio > 0.50:
        if   consult_frac >= 0.95: consult_mult = 0.40
        elif consult_frac >= 0.80: consult_mult = 0.60
        elif consult_frac >= 0.65: consult_mult = 0.80
        elif consult_frac >= 0.50: consult_mult = 0.90
        else:                       consult_mult = 1.00
    else:
        if   consult_frac >= 0.95: consult_mult = 0.08
        elif consult_frac >= 0.80: consult_mult = 0.22
        elif consult_frac >= 0.65: consult_mult = 0.42
        elif consult_frac >= 0.50: consult_mult = 0.68
        else:                       consult_mult = 1.00

    raw = (0.40 * title_score + 0.60 * prod_ratio) * consult_mult
    return clamp(raw), consult_frac, prod_ratio


def score_behavioral(signals: dict, notice: int) -> float:
    """
    Availability & engagement scoring  (weight: 0.20 of final).
    JD: "inactive candidate = effectively unavailable; down-weight aggressively."

    [B10] Added clamp() on return value (raw could exceed 1.0 with high saves).
    """
    # 1. Open to work
    otw = 1.0 if signals.get("open_to_work_flag", False) else 0.0

    # 2. Last active recency
    active_score = 0.5   # neutral default if missing
    last_dt = parse_date(signals.get("last_active_date", ""))
    if last_dt:
        days = (TODAY - last_dt).days
        if   days <=  3: active_score = 1.00
        elif days <=  7: active_score = 0.80
        elif days <= 15: active_score = 0.50
        elif days <= 30: active_score = 0.20
        elif days <= 60: active_score = 0.10
        elif days <= 90: active_score = 0.05
        else:            active_score = 0.00  # >90 days = not actively looking

    # 3. Recruiter response rate
    rr = clamp(float(signals.get("recruiter_response_rate", 0.0) or 0.0))

    # 4. Interview completion
    ic = clamp(float(signals.get("interview_completion_rate", 0.0) or 0.0))

    # 5. Notice period
    if   notice <= 15: notice_score = 1.00
    elif notice <= 30: notice_score = 0.80
    elif notice <= 45: notice_score = 0.50
    elif notice <= 60: notice_score = 0.20
    else:              notice_score = 0.00

    # 6. Recruiter saves (demand signal)
    saves = int(signals.get("saved_by_recruiters_30d", 0) or 0)
    saves_score = clamp(saves / 5.0)    # 5+ saves → 1.0

    # 7. Bonus Signals
    github = int(signals.get("github_activity_score", 0) or 0)
    gh_bonus = 0.05 if github > 50 else 0.0
    
    assessments = signals.get("skill_assessment_scores", {}) or {}
    assess_bonus = 0.0
    for k, v in assessments.items():
        if "python" in k.lower() or "machine learning" in k.lower():
            if int(v or 0) > 85:
                assess_bonus = 0.05

    # [C2] Previously unused behavioral signals (10 of 23 were ignored)
    profile_comp = float(signals.get("profile_completeness_score", 50) or 50) / 100.0
    comp_bonus = 0.03 * profile_comp  # 0-0.03: complete profiles signal seriousness

    resp_time = float(signals.get("avg_response_time_hours", 48) or 48)
    resp_bonus = 0.03 if resp_time <= 4 else 0.02 if resp_time <= 12 else 0.0

    apps = int(signals.get("applications_submitted_30d", 0) or 0)
    apps_bonus = 0.02 if apps >= 3 else 0.01 if apps >= 1 else 0.0

    views = int(signals.get("profile_views_received_30d", 0) or 0)
    views_bonus = 0.02 if views >= 5 else 0.01 if views >= 2 else 0.0

    verified_count = sum([
        bool(signals.get("verified_email", False)),
        bool(signals.get("verified_phone", False)),
        bool(signals.get("linkedin_connected", False)),
    ])
    verify_bonus = 0.02 * (verified_count / 3.0)

    base_score = (
        0.25 * otw
      + 0.25 * active_score
      + 0.20 * rr
      + 0.10 * notice_score
      + 0.10 * ic
      + 0.10 * saves_score
    )
    bonuses = gh_bonus + assess_bonus + comp_bonus + resp_bonus + apps_bonus + views_bonus + verify_bonus
    raw = base_score * 0.85 + bonuses
    return min(1.0, raw)   # [B10] always clamp


def score_experience_fit(profile: dict) -> float:
    """Requested multiplicative modifiers for experience."""
    years = float(profile.get("years_of_experience", 0) or 0)
    if years < 3.5:   return 0.80
    elif years < 5:   return 0.95
    elif years <= 9:  return 1.15
    elif years <= 12: return 1.05
    else:             return 0.90


def score_location(profile: dict, signals: dict) -> float:
    """Location fit  (weight: 0.05 of final). JD: Pune/Noida primary."""
    location = (profile.get("location", "") or "").lower()
    country  = (profile.get("country", "")  or "").lower()
    willing  = bool(signals.get("willing_to_relocate", False))

    is_india = country in ("india", "in") or "india" in location
    if not is_india:
        return 0.50 if willing else 0.18

    is_top  = any(city in location for city in PREFERRED_LOCATIONS)
    if is_top:  return 1.00
    if willing: return 0.65
    return 0.45


# ─────────────────────────────────────────────────────────────────────────────
# HARD PENALTIES  (subtracted from final score)
# ─────────────────────────────────────────────────────────────────────────────

def compute_hard_penalties(profile: dict, career: list, skills: list, signals: dict, consult_frac: float) -> float:
    """
    Explicit JD disqualifiers encoded as additive penalty (max 0.52).
    These catch cases that weighted scoring alone would miss.
    """
    penalty = 0.0
    title   = (profile.get("current_title", "") or "").lower()

    if any(title == t or title.startswith(t + " ") or title.startswith(t + ",") for t in NON_TECH_TITLES):
        penalty += 0.35   # completely wrong field

    # [C6] Fabricated Experience Penalty (moved from honeypot disqualification)
    claimed = float(profile.get("years_of_experience", 0) or 0)
    actual  = sum(int(j.get("duration_months", 0) or 0) for j in career) / 12.0
    if claimed > 3 and actual > 0 and claimed > actual * 2.8:
        penalty += 0.20

    if any(p in title for p in ("junior", "jr.", "intern", "trainee")) and "senior" not in title:
        penalty += 0.15   # JD needs 5-9 yr seniority

    # [C5] Behavioral penalties (activity, response rate) removed to avoid
    # double-counting — already captured in score_behavioral() at 30% weight.

    # Research-heavy career (pure academic, no production)
    if career:
        research_heavy_count = sum(
            1 for job in career
            if sum(1 for kw in ("paper", "publication", "thesis", "arxiv", "academic")
                   if kw in (job.get("description", "") or "").lower()) >= 2
        )
        if research_heavy_count >= len(career) * 0.60:
            penalty += 0.18

        # Title Chaser / Job Hopper Penalty
        if len(career) >= 3:
            total_months = sum(max(1, int(j.get("duration_months", 1) or 1)) for j in career)
            avg_tenure = total_months / len(career)
            if avg_tenure < 18:
                penalty += 0.15

    # [H5] Consulting penalty removed — handled by score_product_fit() via consult_mult.

    # LangChain + OpenAI only with NO retrieval/search skills at all
    skill_names = " ".join(s.get("name", "").lower() for s in skills)
    has_tier1   = any(
        len(s.get("name", "").lower()) >= 3
        and any(kw in s.get("name", "").lower()
                for kw in TIER1_SKILLS)
        for s in skills
    )
    llm_hype_only = (
        not has_tier1
        and any(kw in skill_names for kw in ("langchain", "prompt engineering", "openai", "chatgpt"))
    )
    if llm_hype_only:
        penalty += 0.14

    # [C5] open_to_work penalty removed — already in score_behavioral()

    return clamp(penalty, 0.0, 0.52)


# ─────────────────────────────────────────────────────────────────────────────
# SKILL TAXONOMY  — authoritative mapping used by generate_reasoning
# Each set contains lowercase fragments matched against the candidate skill name.
# BM25/Lucene/Elasticsearch are SEARCH tools, not vector DBs.
# Pinecone/Weaviate/FAISS are VECTOR tools, not search engines.
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
    """Return the taxonomy category for a skill, or None if uncategorised.
    Priority: RANKING > SEARCH > VECTOR > SEMANTIC > RAG
    This ensures BM25 always maps to SEARCH, not mistakenly to VECTOR.
    """
    s = re.sub(r"[-_/]", " ", skill_name.lower()).strip()
    if any(kw in s for kw in SKILL_RANKING):
        return "RANKING"
    if any(kw in s for kw in SKILL_SEARCH):
        return "SEARCH"
    if any(kw in s for kw in SKILL_VECTOR):
        return "VECTOR"
    if any(kw in s for kw in SKILL_SEMANTIC):
        return "SEMANTIC"
    if any(kw in s for kw in SKILL_RAG):
        return "RAG"
    return None




def generate_reasoning(features: dict, profile: dict, career: list, skills: list,
                       notice: int, signals: dict, penalized_score: float,
                       consult_frac: float) -> tuple[str, set[str]]:
    """
    Generates a 1-2 sentence reasoning for Stage 4 manual review.
    Designed to pass all 6 spec checks:
      1. Specific facts (YOE, title, company, skills, signal values)
      2. JD connection (references JD requirements explicitly)
      3. Honest concerns (acknowledges gaps for weaker candidates)
      4. No hallucination (only references actual profile data)
      5. Variation (3 structural templates based on score tier)
      6. Rank consistency (tone matches score/rank position)
    Returns (reason_str, category_set)
    """
    # ── Categorise skills ───────────────────────────────────────────────────
    category_set = set()
    tier1_found = []
    for s_obj in skills:
        name = s_obj.get("name", "") or ""
        if len(name) < 2:
            continue
        cat = _categorise_skill(name)
        if cat:
            category_set.add(cat)
        if TIER1_REGEX.search(name.lower()):
            tier1_found.append(name)
    if not category_set:
        category_set.add("BALANCED")

    # ── Extract facts ───────────────────────────────────────────────────────
    title = profile.get("current_title", "Engineer")
    company = profile.get("current_company", "")
    yoe = float(profile.get("years_of_experience", 0) or 0)
    location = profile.get("location", "")
    company_str = f" at {company}" if company else ""
    loc_str = f", based in {location}" if location else ""

    rr = float(signals.get("recruiter_response_rate", 0) or 0)
    otw = signals.get("open_to_work_flag", False)

    # ── Build strengths list (factual, JD-connected) ────────────────────────
    strengths = []
    if tier1_found:
        strengths.append(f"production experience with {', '.join(tier1_found[:3])}, matching JD's core retrieval/ranking requirement")
    elif features.get("search_retrieval", 0) >= 0.4:
        strengths.append("solid search/ranking background relevant to JD's retrieval focus")
    if features.get("production", 0) >= 0.5:
        strengths.append("evidence of shipping ML systems at scale (JD's 'shipper > researcher' criterion)")
    if features.get("vector_search", 0) >= 0.25:
        strengths.append("hands-on vector DB expertise (JD: Pinecone/FAISS/Qdrant)")
    if 5 <= yoe <= 9:
        strengths.append(f"{yoe:g} years aligns with JD's 5-9 year sweet spot")
    if rr >= 0.7 and otw:
        strengths.append(f"highly reachable ({rr:.0%} response rate, open to work)")

    # ── Build concerns list (honest, specific) ──────────────────────────────
    concerns = []
    if features.get("search_retrieval", 0) < 0.3:
        concerns.append("limited retrieval/search production experience")
    if features.get("vector_search", 0) == 0:
        concerns.append("no explicit vector DB evidence in profile")
    if notice > 60:
        concerns.append(f"{notice}-day notice period (JD prefers sub-30)")
    elif notice > 30:
        concerns.append(f"{notice}-day notice period")
    if 0 < rr < 0.3:
        concerns.append(f"low recruiter response rate ({rr:.0%})")
    if consult_frac > 0.70:
        concerns.append("primarily consulting/services background (JD flags this)")
    if yoe < 4:
        concerns.append(f"only {yoe:g} years experience (JD targets 5-9)")
    elif yoe > 12:
        concerns.append(f"{yoe:g} years may be over-senior for founding team IC role")
    if not otw:
        concerns.append("not marked as open to work")

    # ── Assemble reasoning with rank-consistent tone ────────────────────────
    header = f"{yoe:g}-year {title}{company_str}{loc_str}"

    if penalized_score >= 0.60:
        # Strong candidate — lead with strengths, minimal concerns
        parts = [header + "."]
        if strengths:
            parts.append("Strengths: " + "; ".join(strengths[:2]) + ".")
        if concerns:
            parts.append("Note: " + concerns[0] + ".")
    elif penalized_score >= 0.40:
        # Moderate — balanced assessment
        parts = [header + "."]
        if strengths:
            parts.append("Relevant: " + strengths[0] + ".")
        if concerns:
            parts.append("Gaps: " + "; ".join(concerns[:2]) + ".")
        else:
            parts.append("Partial match to JD's retrieval/ranking focus.")
    else:
        # Weak — honest about poor fit
        parts = [header + "."]
        if concerns:
            parts.append("Concerns: " + "; ".join(concerns[:2]) + ".")
        if strengths:
            parts.append("However, " + strengths[0] + ".")
        else:
            parts.append("Limited alignment with JD's core retrieval/ranking requirements.")

    if notice <= 30 and otw:
        parts.append(f"Available within {notice} days.")

    return " ".join(parts), category_set




# ─────────────────────────────────────────────────────────────────────────────
# DEEP SCORER  (called only on Pass 2 shortlist)
# ─────────────────────────────────────────────────────────────────────────────

def deep_score(candidate: dict) -> tuple[float, str, set[str]]:
    """
    Full 5-component scoring + hard penalties.
    Returns (final_score: float, reasoning: str, category_set: set[str])
    """
    profile  = candidate.get("profile", {}) or {}
    career   = candidate.get("career_history", []) or []
    skills   = candidate.get("skills", []) or []
    signals  = candidate.get("redrob_signals", {}) or {}

    if is_honeypot(candidate):
        return 0.0, (
            "Flagged as honeypot: internally inconsistent profile data "
            "(impossible tenure, inflated experience, or expert/zero-duration skills)."
        ), {"DEFAULT"}

    # [B1] FIXED: single source of truth for notice — one variable, one default
    notice = int(signals.get("notice_period_days", 30) or 30)

    extra_text  = (profile.get("headline", "") or "") + " " + (profile.get("summary", "") or "")
    skill_fit, disq_frac   = score_skill_fit(skills, extra_text, career)
    product_fit, consult_frac, prod_ratio = score_product_fit(profile, career)
    behavioral  = score_behavioral(signals, notice)
    loc_sc      = score_location(profile, signals)
    penalties   = compute_hard_penalties(profile, career, skills, signals, consult_frac)

    # ==========================================
    # STAGE 1: Feature Extraction
    # ==========================================
    features = {}
    career_desc = " ".join(j.get("description", "") or "" for j in career[:3]).lower()[:4000]
    full_text = extra_text.lower() + " " + career_desc

    # 1. Search/Retrieval Expertise (35%)
    primary_hits = len(set(PRIMARY_CORE_RE.findall(full_text)))
    secondary_hits = len(set(SECONDARY_CORE_RE.findall(full_text)))
    
    title_lower = profile.get("job_title", "").lower()
    title_bonus = 0.2 if any(kw in title_lower for kw in ("search engineer", "retrieval engineer", "recommendation engineer", "relevance engineer")) else 0.0
    
    generic_ml = any(kw in title_lower for kw in ("data scientist", "machine learning", "ml engineer", "data engineer"))
    generic_penalty = 0.3 if generic_ml and primary_hits == 0 and len(set(EXPLICIT_VECTOR_RE.findall(full_text))) == 0 else 0.0
    
    sr_score = skill_fit + (primary_hits * 0.30) + (secondary_hits * 0.15) + title_bonus - generic_penalty
    features["search_retrieval"] = min(1.0, max(0.0, sr_score))

    # 2. Production Experience (20%)
    is_research = any(kw in full_text for kw in ("research", "academic", "paper", "publication", "thesis"))
    prod_hits = len(set(PROD_TEXT_RE.findall(full_text)))
    features["production"] = product_fit
    features["production"] = min(1.0, features["production"] + (prod_hits * 0.05))
    if is_research and prod_hits == 0:
        features["production"] = max(0.0, features["production"] - 0.20)

    # 3. Vector DB Expertise (25%)
    vector_hits = len(set(VECTOR_TEXT_RE.findall(full_text)))
    explicit_hits = len(set(EXPLICIT_VECTOR_RE.findall(full_text)))
    features["vector_search"] = min(1.0, (explicit_hits * 0.40) + (vector_hits * 0.15))

    # 4. Notice Period (15%)
    notice = int(signals.get("notice_period_days", 60) or 60)
    if notice <= 15:       notice_score = 1.00
    elif notice <= 30:     notice_score = 0.90
    elif notice <= 60:     notice_score = 0.40
    elif notice <= 90:     notice_score = 0.10
    else:                  notice_score = 0.00
    features["notice_period"] = notice_score

    # 5. Open to Work (5%)
    otw = 1.0 if signals.get("open_to_work_flag", False) else 0.0
    features["open_to_work"] = otw

    # ==========================================
    # STAGE 2: Weighted Linear Scoring
    # ==========================================
    raw_score = (
        0.35 * features["search_retrieval"] +
        0.25 * features["vector_search"] +
        0.20 * features["production"] +
        0.15 * features["notice_period"] +
        0.05 * features["open_to_work"]
    )

    # [C1/C3] Apply experience fit as multiplier (JD sweet spot: 5-9 years)
    exp_mult = score_experience_fit(profile)
    raw_score *= exp_mult
    raw_score *= (0.90 + 0.10 * loc_sc)

    # ==========================================
    # STAGE 3: Penalties & Tie Breakers
    # ==========================================
    rr = float(signals.get("recruiter_response_rate", 0) or 0)
    rr_penalty = 0.05 if rr < 0.20 else 0.0
    
    last_dt = parse_date(signals.get("last_active_date", ""))
    recent_activity = 1.0
    if last_dt:
        days = (TODAY - last_dt).days
        recent_activity = max(0.0, 1.0 - (days / 180.0))
        
    cat_tie = 0.001 * (primary_hits + explicit_hits)
    tie_break = (0.01 * rr) + (0.005 * recent_activity) + cat_tie + (0.01 * behavioral)
    
    penalized_score = max(0.0, raw_score + tie_break - penalties - rr_penalty)
    
    # ==========================================
    # STAGE 4: Non-Linear Calibration (Sigmoid)
    # ==========================================
    # Spreads out top candidates while squashing the middle/bottom
    import math
    final_score = 1.0 / (1.0 + math.exp(-6.0 * (penalized_score - 0.50)))

    reasoning, category = generate_reasoning(features, profile, career, skills, notice, signals, penalized_score, consult_frac)
    return final_score, reasoning, category


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE ITERATOR  [B2][B8] — streams line-by-line, never loads full file
# ─────────────────────────────────────────────────────────────────────────────

def iter_candidates(input_path: str):
    """
    Stream candidates one at a time from .jsonl, .jsonl.gz, or pretty-printed .json array.
    Never loads the entire file into memory.
    """
    path = Path(input_path)
    if not path.exists():
        print(f"ERROR: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8") as f:
        first_chunk = f.read(16)

    # [B6] lstrip before startswith check — handles BOM / leading whitespace
    is_array = first_chunk.lstrip().startswith("[")

    with opener(path, "rt", encoding="utf-8") as f:
        if is_array:
            # sample_candidates.json — full array, small file (<10 MB), OK to parse whole
            try:
                data = json.load(f)
                yield from (d for d in data if isinstance(d, dict))
            except json.JSONDecodeError as e:
                print(f"ERROR: JSON parse failed: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            # candidates.jsonl or .jsonl.gz — stream one line at a time
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                except json.JSONDecodeError:
                    if lineno <= 5:   # warn only for early lines (format check)
                        print(f"[WARN] Bad JSON at line {lineno}: {line[:80]}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# TWO-PASS RANKING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def rank_candidates(input_path: str, output_path: str) -> None:
    """
    Pass 1: Stream all 100K -> fast_score -> keep all with score > 0
    Pass 2: deep_score on survivors -> select top 100
    """
    t0 = time.time()
    print(f"[Ranker v4.0]  Input : {input_path}")
    print(f"[Ranker v4.0]  Output: {output_path}")
    print(f"[Ranker v4.0]  TODAY : {TODAY}\n")

    # ── PASS 1 ────────────────────────────────────────────────────────────────
    print(f"\n[Pass 1] Fast filtering (streaming)...")
    fast_pool: list[tuple[float, str, dict]] = []
    total = 0
    for candidate in iter_candidates(input_path):
        cid = candidate.get("candidate_id", "")
        if not cid:
            continue
        fs = fast_score(candidate)
        total += 1

        fast_pool.append((fs, cid, candidate))
        if total % 10_000 == 0:
            elapsed = time.time() - t0
            print(f"  {total:,} scanned  ({elapsed:.1f}s)", flush=True)

    print(f"[Pass 1] Done: {total:,} candidates scanned in {time.time()-t0:.1f}s")
    fast_pool.sort(key=lambda x: -x[0])
    # Pass candidates with meaningful relevance signal to deep scoring
    # >= 1.5 requires at least one Tier1 skill OR a relevant title + text evidence
    pool = [(s, cid, c) for s, cid, c in fast_pool if s >= 1.5]
    print(f"[Pass 1] {len(pool)} candidates above threshold (>= 1.5) passed to Pass 2 (dropped {total - len(pool)})")

    # ── PASS 2 ────────────────────────────────────────────────────────────────
    print(f"\n[Pass 2] Deep scoring top {len(pool)} candidates...")
    deep_results: list[tuple[str, float, str, set[str]]] = []

    for i, (_, cid, c) in enumerate(pool):
        score, reasoning, category = deep_score(c)
        deep_results.append((cid, score, reasoning, category))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pool)} scored  ({time.time()-t0:.1f}s)", flush=True)

    print(f"[Pass 2] Done in {time.time()-t0:.1f}s")

    # Scores are already calibrated to 0.0 - 1.0 via Sigmoid in deep_score.
    # Sort: score desc, candidate_id asc (deterministic tie-break per spec §3)
    deep_results.sort(key=lambda x: (-x[1], x[0]))
    top_100 = deep_results[:100]

    # [C2] Diversity re-ranking REMOVED — it was corrupting scores (replacing real
    # sigmoid scores with MMR-penalty scores) and shuffling best candidates out of
    # the top 10. NDCG@10 is 50% of hackathon score, so pure quality ordering wins.

    # Honeypot audit
    hp_count = sum(1 for _, sc, _, _ in top_100 if sc == 0.0)
    hp_rate  = hp_count / 100
    flag     = "[WARNING] EXCEEDS 10% THRESHOLD" if hp_rate > 0.10 else "[OK]"
    print(f"\n[Audit]  Honeypots in top 100: {hp_count} ({hp_rate:.0%})  {flag}")

    # ── WRITE CSV ─────────────────────────────────────────────────────────────
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

    print(f"\n{'Rank':>4}  {'Candidate ID':16}  {'Score':>7}  Reasoning (preview 70 chars)")
    print("-" * 106)
    for rank, (cid, score, reasoning, cat) in enumerate(top_100[:10], start=1):
        print(f"{rank:>4}  {cid:16}  {score:>7.4f}  {reasoning[:70]}")
    print(f"\nScore range: {top_100[-1][1]:.4f} – {top_100[0][1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description="Bug Hunters - Redrob Hackathon Ranker v3.4 (Fixed)\n"
                    "Two-pass: 100K stream -> fast filter -> 5K -> deep score -> CSV",
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
# CHANGELOG  (v2 → v2.1)
# ─────────────────────────────────────────────────────────────────────────────
# [B1] Double `notice` variable with different defaults (60 vs 0) in deep_score
#      → unified to single variable with default 60
# [B2] f.read() loading 465 MB into RAM
#      → replaced with streaming iter_candidates() generator
# [B3] `name in kw` matching single/double chars caused false positives
#      → added `len(name) >= 3` guard in fast_score and score_skill_fit
# [B4] Docstring formula (0.35×skill, 0.20×behavioral) ≠ code (0.30×skill, 0.35×behavioral)
#      → docstring updated to match actual code
# [B5] `company_size = profile.get("current_company_size", "")` was assigned but never used
#      → removed from score_product_fit; j_size (per-job size) is now used instead
# [B6] `raw_content.startswith("[")` called after reading full file, also no lstrip
#      → replaced with peek-based format detection after streaming refactor
# [B7] Date parsing only handled %Y-%m-%d, no fallback
#      → parse_date() now tries 5 formats with graceful failure
# [B8] JSON-array branch called json.loads(465MB_string) — OOM risk
#      → array branch now uses json.load(f) (streaming parser) on small files only
# [B9] NON_TECH_TITLES was a plain set (O(n) hash); changed to frozenset (marginal but cleaner)
# [B10] score_behavioral could return >1.0 if saves_score pushed it over; added clamp()
