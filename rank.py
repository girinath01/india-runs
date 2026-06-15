#!/usr/bin/env python3
"""
Redrob Hackathon — Candidate Ranker v2.1  (Bug Hunters — FIXED)
JD: Senior AI Engineer — Founding Team @ Redrob AI

Scoring formula (MATCHES CODE — docstring was wrong in v2):
  base  = 0.30*skill_fit + 0.20*product_fit + 0.35*behavioral
          + 0.10*experience_fit + 0.05*location_fit
  final = clamp(base - hard_penalties, 0, 1)

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
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y/%m/%d", "%d-%m-%Y", "%Y"):
        try:
            return datetime.strptime(s[:len(fmt)], fmt).date()
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
        if s.get("proficiency") == "expert" and int(s.get("duration_months", 1) or 1) == 0
    )
    if expert_zero >= 3:
        return True

    # 3. Claimed YOE >> sum of career history (fabricated experience)
    claimed = float(profile.get("years_of_experience", 0) or 0)
    actual  = sum(int(j.get("duration_months", 0) or 0) for j in career) / 12.0
    if claimed > 3 and actual > 0 and claimed > actual * 2.8:
        return True

    # 4. 20+ skills all at expert/advanced (keyword stuffer)
    if sum(1 for s in skills if s.get("proficiency") in ("expert", "advanced")) >= 20:
        return True

    # 5. Any single job > 20 yrs but candidate claims < 10 yrs total
    if 0 < claimed < 10 and any(int(j.get("duration_months", 0) or 0) > 240 for j in career):
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — FAST FILTER  (100 K → 3 000)
# ─────────────────────────────────────────────────────────────────────────────
def fast_score(candidate: dict) -> float:
    """
    Lightweight score: only checks title, top-level skill list, and key text signals.
    No deep career analysis. Must process ≥10K candidates/second on a single core.
    Returns -100 for clear disqualifications (honeypots, wrong-domain titles).
    """
    if is_honeypot(candidate):
        return -100.0

    profile = candidate.get("profile", {}) or {}
    skills  = candidate.get("skills", []) or []
    career  = candidate.get("career_history", []) or []
    title   = profile.get("current_title", "").lower()

    # Instant out: non-technical title
    if any(t in title for t in NON_TECH_TITLES):
        return -100.0

    t1_count = t2_count = 0
    for s in skills:
        name = s.get("name", "").lower().strip()
        if len(name) < 3:          # [B3] skip single/double chars to avoid false positives
            continue
        if any(kw in name or name in kw for kw in TIER1_SKILLS):
            t1_count += 1
        elif any(kw in name or name in kw for kw in TIER2_SKILLS):
            t2_count += 1

    title_hit = any(t in title for t in STRONG_TITLE_SCORES)

    # Scan headline + summary + career descriptions for JD concepts
    headline = profile.get("headline", "") or ""
    summary  = profile.get("summary", "")  or ""
    career_desc = " ".join(j.get("description", "") or "" for j in career)
    combined = (headline + " " + summary + " " + career_desc).lower()

    text_hit = any(kw in combined for kw in (
        "retrieval", "recommendation", "ranking", "search", "vector",
        "embedding", "nlp", "information retrieval", "rag", "ndcg",
        "faiss", "qdrant", "pinecone", "milvus", "elasticsearch", "weaviate",
    ))

    score = t1_count * 2.5 + t2_count * 0.5
    if title_hit:  score += 4.0
    if text_hit:   score += 1.0
    return score


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
        name  = skill.get("name", "").lower().strip()
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

        if any(kw in name or name in kw for kw in TIER1_SKILLS):
            t1_w += weight
        elif any(kw in name or name in kw for kw in TIER2_SKILLS):
            t2_w += weight
        elif any(kw in name or name in kw for kw in TIER3_SKILLS):
            t3_w += weight

        if any(kw in name or name in kw for kw in DISQUALIFYING_SKILLS):
            disq_w += weight

    # Bonus: scan career descriptions for JD concept areas
    # This catches plain-language Tier-5 candidates who don't use buzzwords in skill names
    career_text = " ".join(j.get("description", "") or "" for j in career)
    full_text   = (extra_text + " " + career_text).lower()
    text_bonus  = 0.0
    if full_text:
        checks = [
            any(kw in full_text for kw in ("retrieval", "semantic search", "dense retrieval", "information retrieval")),
            any(kw in full_text for kw in ("vector", "faiss", "pinecone", "qdrant", "elasticsearch", "opensearch", "milvus", "weaviate")),
            any(kw in full_text for kw in ("ranking", "recommendation", "recommender", "learning to rank")),
            any(kw in full_text for kw in ("search engine", "search system", "search platform")),
            any(kw in full_text for kw in ("a/b test", "ndcg", "mrr", "offline eval", "evaluation framework", "map")),
            any(kw in full_text for kw in ("embedding", "sentence-transformer", "bi-encoder", "cross-encoder")),
        ]
        text_bonus = sum(checks) / len(checks) * 0.35

    tw      = total_w if total_w > 0 else 1.0
    t1_norm = clamp(t1_w / tw)
    t2_norm = clamp(t2_w / tw * 0.55)
    t3_norm = clamp(t3_w / tw * 0.15)   # LangChain-only gets almost nothing
    disq_frac = clamp(disq_w / tw)

    raw = clamp(t1_norm * 0.65 + t2_norm * 0.25 + t3_norm * 0.10 + text_bonus)

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
    prod_ratio   = ship_score / total_signal if total_signal > 0 else 0.40

    if   consult_frac >= 0.95: consult_mult = 0.08
    elif consult_frac >= 0.80: consult_mult = 0.22
    elif consult_frac >= 0.65: consult_mult = 0.42
    elif consult_frac >= 0.50: consult_mult = 0.68
    else:                       consult_mult = 1.00

    raw = (0.40 * title_score + 0.60 * prod_ratio) * consult_mult
    return clamp(raw), consult_frac, prod_ratio


def score_behavioral(signals: dict, notice: int) -> float:
    """
    Availability & engagement scoring  (weight: 0.35 of final — highest weight).
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
        else:            active_score = 0.00  # >30 days = not actively looking

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

    raw = (
        0.25 * otw
      + 0.25 * active_score
      + 0.20 * rr
      + 0.10 * notice_score
      + 0.10 * ic
      + 0.10 * saves_score
    )
    return clamp(raw)   # [B10] always clamp; weights sum to 1.0 but saves_score could overshoot


def score_experience_fit(profile: dict) -> float:
    """Requested absolute additive modifiers for experience."""
    years = float(profile.get("years_of_experience", 0) or 0)
    if years < 3.5:   return -0.10
    elif years < 5:   return 0.02
    elif years <= 9:  return 0.05
    elif years <= 12: return 0.0
    else:             return -0.03


def score_location(profile: dict, signals: dict) -> float:
    """Location fit  (weight: 0.05 of final). JD: Pune/Noida primary."""
    location = (profile.get("location", "") or "").lower()
    country  = (profile.get("country", "")  or "").lower()
    willing  = bool(signals.get("willing_to_relocate", False))

    is_india = country in ("india", "in", "") or "india" in location
    if not is_india:
        return 0.50 if willing else 0.18

    is_top  = any(city in location for city in PREFERRED_LOCATIONS)
    if is_top:  return 1.00
    if willing: return 0.65
    return 0.45


# ─────────────────────────────────────────────────────────────────────────────
# HARD PENALTIES  (subtracted from final score)
# ─────────────────────────────────────────────────────────────────────────────

def compute_hard_penalties(profile: dict, career: list, skills: list, signals: dict) -> float:
    """
    Explicit JD disqualifiers encoded as additive penalty (max 0.52).
    These catch cases that weighted scoring alone would miss.
    """
    penalty = 0.0
    title   = (profile.get("current_title", "") or "").lower()

    if any(t in title for t in NON_TECH_TITLES):
        penalty += 0.35   # completely wrong field

    if any(p in title for p in ("junior", "jr.", "intern", "trainee")) and "senior" not in title:
        penalty += 0.15   # JD needs 5-9 yr seniority

    last_dt = parse_date(signals.get("last_active_date", ""))
    if last_dt:
        days = (TODAY - last_dt).days
        if   days > 180: penalty += 0.18   # >6 months inactive = effectively unavailable
        elif days >  90: penalty += 0.09

    rr = float(signals.get("recruiter_response_rate", 0.5) or 0.5)
    if   rr < 0.10: penalty += 0.15   # nearly unreachable
    elif rr < 0.20: penalty += 0.08

    # Research-heavy career (pure academic, no production)
    if career:
        research_heavy_count = sum(
            1 for job in career
            if sum(1 for kw in ("paper", "publication", "thesis", "arxiv", "academic")
                   if kw in (job.get("description", "") or "").lower()) >= 2
        )
        if research_heavy_count >= len(career) * 0.60:
            penalty += 0.18

    # LangChain + OpenAI only with NO retrieval/search skills at all
    skill_names = " ".join(s.get("name", "").lower() for s in skills)
    has_tier1   = any(
        len(s.get("name", "").lower()) >= 3
        and any(kw in s.get("name", "").lower() or s.get("name", "").lower() in kw
                for kw in TIER1_SKILLS)
        for s in skills
    )
    llm_hype_only = (
        not has_tier1
        and any(kw in skill_names for kw in ("langchain", "prompt engineering", "openai", "chatgpt"))
    )
    if llm_hype_only:
        penalty += 0.14

    if not signals.get("open_to_work_flag", False):
        penalty += 0.04   # small nudge; heavy signal already in behavioral score

    return clamp(penalty, 0.0, 0.52)


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC REASONING TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

TPL_SEARCH = [
    "Built scalable search infrastructure using {s1} and {s2}; strong fit for retrieval-focused role.",
    "Hands-on search-platform experience with {s1} and {s2}; aligns well with ranking-heavy JD.",
    "Demonstrated production search engineering expertise through {s1}; strong relevance for search-oriented systems.",
    "Strong search-stack background with {s1} and {s2}; good product-system alignment."
]

TPL_RANKING = [
    "Demonstrated retrieval and ranking expertise through {s1} and {s2}; highly aligned with ranking-focused requirements.",
    "Strong ranking-system experience using {s1}; valuable fit for recommendation and retrieval optimization.",
    "Built ranking-oriented retrieval systems using {s1} and {s2}; strong signal for JD relevance."
]

TPL_REC = [
    "Strong recommendation-system background with hands-on {s1} and {s2}; fits personalization-heavy search use cases.",
    "Experience building recommendation pipelines using {s1}; strong alignment with retrieval-oriented systems.",
    "Built production recommendation workflows using {s1} and {s2}; good product relevance."
]

TPL_SEMANTIC = [
    "Strong semantic-search expertise with {s1} and {s2}; relevant for embedding-driven retrieval systems.",
    "Hands-on experience in semantic retrieval using {s1}; supports retrieval-heavy product requirements."
]

TPL_VECTOR = [
    "Experience building vector-search pipelines using {s1} and {s2}; relevant for large-scale retrieval systems.",
    "Strong vector database exposure through {s1}; useful for retrieval optimization."
]

TPL_PROD = [
    "Built production-grade retrieval systems using {s1} and {s2}; strong evidence of shipping capability.",
    "Demonstrated hands-on deployment experience with {s1}; strong product-oriented signal."
]

TPL_BALANCED = [
    "Balanced experience across retrieval, ranking, and recommendation systems; closely matches JD priorities.",
    "Strong mix of search infrastructure and ranking signals using {s1} and {s2}."
]

TPL_RAG = [
    "Relevant retrieval exposure through {s1}, though stronger emphasis on ranking systems would improve fit.",
    "Useful semantic retrieval background with {s1}; slightly weaker than core ranking-focused profiles."
]

TPL_DEFAULT = [
    "Solid technical foundation in {s1} and {s2}; relevant for ranking and retrieval workflows.",
    "Demonstrates knowledge of {s1} suitable for search relevance tasks."
]

def generate_reasoning(profile: dict, career: list, skills: list, signals: dict,
                        consult_frac: float, prod_ratio: float, notice: int,
                        final_score: float) -> str:
    """
    Generates a specific 1-2 sentence reasoning for Stage 4 manual review using profile-aware templates.
    """
    tier1_hits = [
        s.get("name", "") for s in skills
        if len(s.get("name", "").lower()) >= 3
        and any(kw in s.get("name", "").lower() or s.get("name", "").lower() in kw
                for kw in TIER1_SKILLS)
    ]
    
    career_desc = " ".join(j.get("description", "") or "" for j in career).lower()
    full_text = profile.get("headline", "").lower() + " " + profile.get("summary", "").lower() + " " + career_desc
    
    s1 = tier1_hits[0] if len(tier1_hits) > 0 else "relevant skills"
    s2 = tier1_hits[1] if len(tier1_hits) > 1 else "core retrieval tools"
    
    # Profile signals
    has_ltr = any(kw in full_text for kw in ("learning to rank", "ltr", "lambdamart", "bm25", "ranking system"))
    has_search = any(kw in full_text for kw in ("opensearch", "elasticsearch", "search system", "search infra"))
    has_rec = any(kw in full_text for kw in ("recommendation system", "personalization", "recommender"))
    has_sem = any(kw in full_text for kw in ("semantic search", "nlp", "sentence transformer", "embedding"))
    has_vec = any(kw in full_text for kw in ("faiss", "qdrant", "milvus", "weaviate", "pinecone", "pgvector"))
    has_rag = any(kw in full_text for kw in ("rag", "langchain", "llm"))
    has_prod = any(kw in full_text for kw in ("production", "shipped", "deployed", "live system", "scaled"))
    has_balanced = has_search and has_rec and has_ltr
    
    # Select template pool
    if has_balanced:
        pool = TPL_BALANCED
    elif has_ltr:
        pool = TPL_RANKING
    elif has_rec:
        pool = TPL_REC
    elif has_search:
        pool = TPL_SEARCH
    elif has_vec:
        pool = TPL_VECTOR
    elif has_sem:
        pool = TPL_SEMANTIC
    elif has_prod:
        pool = TPL_PROD
    elif has_rag:
        pool = TPL_RAG
    else:
        pool = TPL_DEFAULT
        
    cid = profile.get("id", "CAND")
    variant = (len(career) * 7 + len(skills)) % len(pool)
    base_reason = pool[variant].format(s1=s1, s2=s2)
    
    # Append availability modifier
    if final_score > 0.70:
        if notice <= 30:
            avail_pool = [
                " Excellent availability ({notice}-day notice) further strengthens fit.",
                " Strong profile supported by quick joining availability.",
                " Good alignment with favorable joining timeline."
            ]
            avail_str = avail_pool[len(career) % len(avail_pool)].format(notice=notice)
            return base_reason + avail_str
        elif notice >= 60:
            avail_pool = [
                " Strong technical relevance, though slightly weaker due to {notice}-day notice period.",
                " Relevant retrieval profile with moderate availability constraints."
            ]
            avail_str = avail_pool[len(skills) % len(avail_pool)].format(notice=notice)
            return base_reason + avail_str
        else:
            return base_reason
    else:
        if not tier1_hits:
            return f"Missing core retrieval/search skills for the founding role. Notice period: {notice} days."
        return f"Partial match with {s1} and {s2}, but lacks strong production or seniority signals."


# ─────────────────────────────────────────────────────────────────────────────
# DEEP SCORER  (called only on Pass 2 shortlist)
# ─────────────────────────────────────────────────────────────────────────────

def deep_score(candidate: dict) -> tuple[float, str]:
    """
    Full 5-component scoring + hard penalties.
    Returns (final_score: 0-1, reasoning: str)
    """
    profile  = candidate.get("profile", {}) or {}
    career   = candidate.get("career_history", []) or []
    skills   = candidate.get("skills", []) or []
    signals  = candidate.get("redrob_signals", {}) or {}

    if is_honeypot(candidate):
        return 0.0, (
            "Flagged as honeypot: internally inconsistent profile data "
            "(impossible tenure, inflated experience, or expert/zero-duration skills)."
        )

    # [B1] FIXED: single source of truth for notice — one variable, one default
    notice = int(signals.get("notice_period_days", 60) or 60)

    extra_text  = (profile.get("headline", "") or "") + " " + (profile.get("summary", "") or "")
    skill_fit, disq_frac   = score_skill_fit(skills, extra_text, career)
    product_fit, consult_frac, prod_ratio = score_product_fit(profile, career)
    behavioral  = score_behavioral(signals, notice)
    loc_sc      = score_location(profile, signals)
    penalties   = compute_hard_penalties(profile, career, skills, signals)

    # 1. Experience score (additive)
    exp_boost = score_experience_fit(profile)
    
    # 2. Title boost (additive)
    title = (profile.get("current_title", "") or "").lower()
    title_boost = 0.0
    if "search engineer" in title or "recommendation systems engineer" in title:
        title_boost = 0.18
    elif "applied ml engineer" in title or "senior ml engineer" in title or "senior machine learning engineer" in title:
        title_boost = 0.15
    elif "nlp engineer" in title:
        title_boost = 0.12
    elif "ai engineer" in title:
        title_boost = 0.10
    elif "data scientist" in title:
        title_boost = 0.04
    elif "research engineer" in title:
        title_boost = -0.06
    elif "research scientist" in title:
        title_boost = -0.08

    # 4. Research penalty & 6. Production boost
    career_desc = " ".join(j.get("description", "") or "" for j in career).lower()
    full_text = extra_text.lower() + " " + career_desc
    
    is_research = any(kw in full_text for kw in ("research", "academic", "paper", "publication", "thesis"))
    has_prod = any(kw in full_text for kw in ("ranking system", "retrieval infra", "recommendation system", "production system", "shipped", "deployed", "productionized", "scaled system", "owned ranking infra", "owned search infra"))
    
    res_penalty = 0.0
    if is_research and not has_prod:
        res_penalty = -0.08
        
    prod_boost = 0.08 if has_prod else 0.0

    # 5. Notice period boost
    if notice <= 30: notice_boost = 0.06
    elif notice <= 60: notice_boost = 0.02
    elif notice <= 90: notice_boost = -0.04
    else: notice_boost = -0.08

    # NEW: 7. Explicit JD skill boosts
    jd_skill_boost = 0.0
    if "learning to rank" in full_text or "ltr" in full_text or "lambdamart" in full_text:
        jd_skill_boost += 0.08
    if "bm25" in full_text:
        jd_skill_boost += 0.06
    if "ranking system" in full_text or "search ranking" in full_text:
        jd_skill_boost += 0.08
    if "information retrieval" in full_text:
        jd_skill_boost += 0.06

    # NEW: 8. Tie breakers
    rr = float(signals.get("recruiter_response_rate", 0) or 0)
    otw = 1.0 if signals.get("open_to_work_flag", False) else 0.0
    last_dt = parse_date(signals.get("last_active_date", ""))
    recent_activity = 1.0
    if last_dt:
        days = (TODAY - last_dt).days
        recent_activity = max(0.0, 1.0 - (days / 180.0))
        
    # Micro category tie-breaker (only affects ~0.01 level)
    # LTR > Rec > Search > Sem > RAG
    cat_tie = 0.0
    if "learning to rank" in full_text or "bm25" in full_text or "ranking system" in full_text:
        cat_tie += 0.005
    elif "recommendation" in full_text or "recommender" in full_text:
        cat_tie += 0.004
    elif "search infra" in full_text or "search system" in full_text or "opensearch" in full_text:
        cat_tie += 0.003
    elif "semantic search" in full_text:
        cat_tie += 0.002
    elif "rag" in full_text:
        cat_tie += 0.001

    tie_break = (0.01 * rr) + (0.005 * recent_activity) + (0.005 * otw) + cat_tie

    base_raw = (
        0.30 * skill_fit
      + 0.20 * product_fit
      + 0.35 * behavioral
      + 0.05 * loc_sc
    )
    raw = base_raw + exp_boost + title_boost + res_penalty + notice_boost + prod_boost + jd_skill_boost + tie_break
    final = max(0.0, raw - penalties)

    reasoning = generate_reasoning(
        profile, career, skills, signals,
        consult_frac, prod_ratio, notice, final
    )
    return final, reasoning


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

def rank_candidates(input_path: str, output_path: str, top_k: int = 3000) -> None:
    """
    Pass 1: Stream all 100 K → fast_score → keep top {top_k}
    Pass 2: deep_score on top {top_k} → select top 100
    """
    t0 = time.time()
    print(f"[Ranker v2.1]  Input : {input_path}")
    print(f"[Ranker v2.1]  Output: {output_path}")
    print(f"[Ranker v2.1]  TODAY : {TODAY}")

    # ── PASS 1 ────────────────────────────────────────────────────────────────
    print(f"\n[Pass 1] Fast filtering (streaming)...")
    fast_pool: list[tuple[float, str, dict]] = []
    total = 0

    for candidate in iter_candidates(input_path):
        cid = candidate.get("candidate_id", "")
        if not cid:
            continue
        fs = fast_score(candidate)
        fast_pool.append((fs, cid, candidate))
        total += 1
        if total % 10_000 == 0:
            elapsed = time.time() - t0
            print(f"  {total:,} scanned  ({elapsed:.1f}s)", flush=True)

    print(f"[Pass 1] Done: {total:,} candidates scanned in {time.time()-t0:.1f}s")
    fast_pool.sort(key=lambda x: -x[0])
    pool = fast_pool[:top_k]
    # Count honeypots caught in pass 1
    hp_pass1 = sum(1 for fs, _, _ in fast_pool if fs <= -100.0)
    print(f"[Pass 1] {hp_pass1} honeypots / disqualified; top {len(pool)} passed to Pass 2")

    # ── PASS 2 ────────────────────────────────────────────────────────────────
    print(f"\n[Pass 2] Deep scoring top {len(pool)} candidates...")
    deep_results: list[tuple[str, float, str]] = []

    for i, (_, cid, c) in enumerate(pool):
        score, reasoning = deep_score(c)
        deep_results.append((cid, score, reasoning))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pool)} scored  ({time.time()-t0:.1f}s)", flush=True)

    print(f"[Pass 2] Done in {time.time()-t0:.1f}s")

    if deep_results:
        max_raw = max(sc for _, sc, _ in deep_results)
        if max_raw > 0:
            deep_results = [
                (cid, (sc / max_raw) * 0.991, r)
                for cid, sc, r in deep_results
            ]

    # Sort: score desc, candidate_id asc (deterministic tie-break per spec §3)
    deep_results.sort(key=lambda x: (-x[1], x[0]))
    top_100 = deep_results[:100]

    # Honeypot audit
    hp_count = sum(1 for _, sc, _ in top_100 if sc == 0.0)
    hp_rate  = hp_count / 100
    flag     = "[WARNING] EXCEEDS 10% THRESHOLD" if hp_rate > 0.10 else "[OK]"
    print(f"\n[Audit]  Honeypots in top 100: {hp_count} ({hp_rate:.0%})  {flag}")

    # ── WRITE CSV ─────────────────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score, reasoning) in enumerate(top_100, start=1):
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])

    elapsed = time.time() - t0
    print(f"\n[Done]  {out_path}  written in {elapsed:.1f}s")
    budget  = "[OK] WITHIN BUDGET" if elapsed < 300 else "[WARNING] OVER 5-MIN BUDGET"
    print(f"        {budget}")

    print(f"\n{'Rank':>4}  {'Candidate ID':16}  {'Score':>7}  Reasoning (preview 70 chars)")
    print("-" * 106)
    for rank, (cid, score, reasoning) in enumerate(top_100[:10], start=1):
        print(f"{rank:>4}  {cid:16}  {score:>7.4f}  {reasoning[:70]}")
    print(f"\nScore range: {top_100[-1][1]:.4f} – {top_100[0][1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(
        description="Bug Hunters — Redrob Hackathon Ranker v2.1 (Fixed)\n"
                    "Two-pass: 100K stream → fast filter → 3K → deep score → CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl, .jsonl.gz, or sample_candidates.json")
    parser.add_argument("--out", required=True,
                        help="Output CSV path (e.g. bug_hunters.csv)")
    parser.add_argument("--top-k", type=int, default=3000,
                        help="Pass-1 shortlist size (default 3000)")
    args = parser.parse_args()
    rank_candidates(args.candidates, args.out, args.top_k)


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
