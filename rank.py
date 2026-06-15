#!/usr/bin/env python3
"""
Redrob Hackathon — Candidate Ranker (Bug Hunters)
JD: Senior AI Engineer — Founding Team @ Redrob AI

WIP — adding signal definitions first, scorers next.
"""

import json
import csv
import argparse
import sys
from datetime import datetime, date
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# JD SIGNAL DEFINITIONS
# Read the JD very carefully before touching these.
# The key trap: keyword match is NOT enough. A "Marketing Manager"
# with perfect AI keywords is NOT a fit. A Recommendation Systems Engineer
# without "RAG"/"Pinecone" in their skills IS a fit.
# ─────────────────────────────────────────────────────────────────────────────

# Core required skills from the JD (embeddings, vector DBs, evaluation, NLP)
REQUIRED_SKILLS = {
    # Embeddings & Retrieval — JD explicitly needs PRODUCTION experience here
    "sentence-transformers", "sentence transformer", "embeddings", "embedding",
    "openai embeddings", "text-embedding", "bge", "e5 model", "ada",
    "cohere embed", "dense retrieval", "sparse retrieval",
    "hybrid retrieval", "hybrid search",
    "bi-encoder", "cross-encoder", "semantic search", "semantic similarity",
    "retrieval", "information retrieval", "dense passage retrieval", "dpr",
    # Vector Databases — JD explicitly needs PRODUCTION experience here
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "chroma", "pgvector", "vespa", "annoy", "scann",
    "vector database", "vector db", "vector store", "vector index",
    "vector search", "ann", "approximate nearest neighbor", "hnsw",
    # RAG & Search
    "rag", "retrieval augmented generation", "bm25", "tfidf", "tf-idf",
    "reranking", "re-ranking", "reranker", "colbert",
    # Ranking & Recommendation (recommendation system counts per JD)
    "learning to rank", "ltr", "lambdamart", "lambdarank", "ranknet",
    "recommendation system", "recommender", "ranking system", "search ranking",
    "ndcg", "mrr", "mean reciprocal rank", "map", "mean average precision",
    "a/b testing", "ab testing", "offline evaluation", "online evaluation",
    "eval framework", "evaluation framework",
    # NLP & LLMs
    "nlp", "natural language processing", "language model", "text classification",
    "bert", "transformers", "huggingface", "hugging face",
    "llm", "large language model", "generative ai", "gpt",
    "fine-tuning", "fine tuning", "finetuning", "lora", "qlora", "peft",
    "prompt engineering", "instruction tuning",
    # ML Engineering
    "machine learning", "deep learning", "neural network",
    "pytorch", "tensorflow", "keras", "jax",
    "xgboost", "lightgbm", "gradient boosting",
    "scikit-learn", "sklearn",
    "mlops", "model deployment", "model serving", "inference",
    "feature engineering", "feature store",
    # Engineering fundamentals for product role
    "python", "fastapi", "flask", "api", "microservices", "docker", "kubernetes",
    "spark", "kafka", "airflow",
    "aws", "gcp", "azure", "cloud",
    "sql", "postgresql", "mongodb", "redis",
    "distributed systems", "distributed training",
    # Specific tools mentioned in JD
    "weights & biases", "wandb", "mlflow", "ray", "dask",
    "triton", "tensorrt", "onnx",
}

# Nice-to-have (lower weight contribution)
NICE_TO_HAVE_SKILLS = {
    "rlhf", "reinforcement learning from human feedback",
    "data parallelism", "model parallelism", "quantization", "distillation",
    "hr tech", "talent intelligence", "marketplace", "two-sided marketplace",
    "open source", "github contributions",
}

# CV / Speech / Robotics — disqualifying if DOMINANT in skill profile
# JD explicitly calls these out as "wrong domain"
DISQUALIFYING_SKILL_FOCUS = {
    "computer vision", "image classification", "object detection",
    "yolo", "convolutional", "image segmentation", "instance segmentation",
    "opencv", "open cv", "object recognition",
    "speech recognition", "asr", "tts", "text-to-speech", "speech synthesis",
    "audio processing", "sound classification", "voice recognition",
    "robotics", "ros", "autonomous driving", "lidar", "slam", "point cloud",
    # Non-ML roles
    "photoshop", "illustrator", "figma", "adobe", "canva",
    "six sigma", "lean", "kaizen",
    "solidworks", "autocad", "ansys", "creo",
    "accounting", "tally", "gst", "taxation",
    "seo", "content writing", "copywriting",
    "salesforce", "crm",
}

# Consulting companies by name (fallback if industry field unavailable)
CONSULTING_COMPANY_NAMES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "ltimindtree", "lti", "larsen toubro infotech", "niit technologies",
    "persistent systems", "cyient", "zensar", "sonata software",
    "kpit", "mastech", "igate", "firstsource", "wns", "genpact",
    "birlasoft", "coforge", "happiest minds",
}

# Consulting by INDUSTRY FIELD — more reliable than company name matching
CONSULTING_INDUSTRIES = {
    "it services", "it consulting", "consulting",
    "business process outsourcing", "bpo", "outsourcing",
    "staffing", "staffing and recruiting", "managed services", "it staffing",
}

# Target technical titles for this JD
TARGET_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "nlp engineer", "search engineer", "information retrieval engineer",
    "data scientist", "applied scientist", "research engineer",
    "senior engineer", "principal engineer", "staff engineer",
    "deep learning engineer", "recommendation engineer",
    "ranking engineer", "relevance engineer", "search scientist",
    "software engineer", "backend engineer", "platform engineer",
    "applied ml", "applied ai", "ai researcher", "ml researcher",
    "data engineer", "mlops engineer",
}

# Clearly non-technical — strong negative signal
NON_TECH_TITLES = {
    "marketing manager", "hr manager", "human resources", "operations manager",
    "business analyst", "content writer", "sales executive", "accountant",
    "project manager", "graphic designer", "customer support", "customer service",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "finance manager", "account manager", "talent acquisition", "recruiter",
    "junior analyst", "data entry", "qa engineer", "tester",
    "business development",
}

# JD preferred locations (Pune/Noida first, then other Tier-1 India cities)
PREFERRED_INDIA_LOCATIONS = {
    "pune",
    "noida", "delhi", "delhi ncr", "ncr", "new delhi",
    "gurgaon", "gurugram", "faridabad", "greater noida",
    "hyderabad", "mumbai", "bangalore", "bengaluru",
}

# Proficiency → numeric weight mapping
PROFICIENCY_WEIGHTS = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.45,
    "beginner": 0.15,
}

TODAY = date(2026, 6, 15)  # Evaluation reference date


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def normalize(value, min_v, max_v):
    if max_v <= min_v:
        return 0.0
    return clamp((value - min_v) / (max_v - min_v))


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT SCORERS
# ─────────────────────────────────────────────────────────────────────────────

def score_skills(skills: list, extra_text: str = "") -> tuple:
    """
    Returns (ai_skill_score [0-1], disq_skill_fraction [0-1])

    ai_skill_score: proficiency + endorsement + duration weighted match vs JD
    disq_skill_fraction: fraction of weight coming from CV/Speech/Robotics skills

    Also scans extra_text (profile summary + headline) for JD keywords —
    catches plain-language Tier 5 candidates who describe experience in prose
    without using keyword-heavy skill lists.
    """
    if not skills and not extra_text:
        return 0.0, 0.0

    total_weight = 0.0
    matched_weight = 0.0
    disq_weight = 0.0

    for skill in skills:
        name_raw = skill.get("name", "")
        name = name_raw.lower().strip()
        proficiency = skill.get("proficiency", "beginner")
        endorsements = min(skill.get("endorsements", 0), 100)
        duration = min(skill.get("duration_months", 0), 72)

        prof_w = PROFICIENCY_WEIGHTS.get(proficiency, 0.15)
        endorse_bonus = (endorsements / 100.0) * 0.20   # up to +0.20
        duration_bonus = (duration / 72.0) * 0.20        # up to +0.20

        skill_weight = prof_w * (1.0 + endorse_bonus + duration_bonus)
        total_weight += skill_weight

        # JD keyword match (substring in both directions to catch partials)
        is_match = any(kw in name or name in kw for kw in REQUIRED_SKILLS)
        if is_match:
            matched_weight += skill_weight

        # Disqualifying domain check
        is_disq = any(dq in name or name in dq for dq in DISQUALIFYING_SKILL_FOCUS)
        if is_disq:
            disq_weight += skill_weight

    # Also scan free text (summary / headline) for JD concept areas
    # Weighted lower than skills list — free text is noisier
    if extra_text:
        text_l = extra_text.lower()
        JD_CONCEPT_CHECKS = [
            any(kw in text_l for kw in ["embedding", "retrieval", "semantic search", "dense"]),
            any(kw in text_l for kw in ["vector", "pinecone", "weaviate", "faiss", "qdrant", "milvus"]),
            any(kw in text_l for kw in ["ranking", "recommendation", "recommender", "ltr", "rerank"]),
            any(kw in text_l for kw in ["nlp", "natural language", "language model", "llm", "bert"]),
            any(kw in text_l for kw in ["rag", "retrieval augmented"]),
            any(kw in text_l for kw in ["a/b test", "ab test", "ndcg", "mrr", "offline eval"]),
            any(kw in text_l for kw in ["python", "pytorch", "tensorflow", "deep learning"]),
        ]
        text_score = sum(JD_CONCEPT_CHECKS) / len(JD_CONCEPT_CHECKS)
        # Blend text score in at 30% weight relative to skills list
        text_synthetic_weight = total_weight * 0.3 if total_weight > 0 else 5.0
        total_weight += text_synthetic_weight
        matched_weight += text_score * text_synthetic_weight

    if total_weight == 0:
        return 0.0, 0.0

    skill_score = clamp(matched_weight / total_weight)
    disq_frac = clamp(disq_weight / (total_weight * 0.7 + 1e-9))  # normalize against skills-only weight
    return skill_score, disq_frac


def score_title_fit(profile: dict, career_history: list) -> tuple:
    """
    Returns (title_score [0-1], is_non_tech bool, consulting_fraction [0-1])

    Consulting detection uses the `industry` field in career_history —
    this is more reliable than company name matching since many consulting
    companies operate under subsidiary names that aren't in our list.
    """
    current_title = profile.get("current_title", "").lower()
    current_industry = profile.get("current_industry", "").lower()

    # Score current title against target technical roles
    title_score = 0.0
    for target in TARGET_TITLES:
        if target in current_title:
            title_score = 1.0
            break

    is_non_tech = any(t in current_title for t in NON_TECH_TITLES)
    if is_non_tech:
        title_score = max(0.0, title_score * 0.1)

    # Walk career history: score historical technical titles + consulting fraction
    hist_tech_score = 0.0
    total_months = 0
    consulting_months = 0

    for job in career_history:
        job_title = job.get("title", "").lower()
        company = job.get("company", "").lower()
        industry = job.get("industry", "").lower()
        duration = max(0, job.get("duration_months", 0))
        total_months += duration

        # Technical title check in history
        for target in TARGET_TITLES:
            if target in job_title:
                hist_tech_score = max(hist_tech_score, 0.85)
                break

        # Consulting detection: prefer industry field, fall back to company name
        is_consulting = (
            any(ci in industry for ci in CONSULTING_INDUSTRIES)
            or any(cn in company for cn in CONSULTING_COMPANY_NAMES)
        )
        if is_consulting:
            consulting_months += duration

    # Also check current company industry (profile-level field)
    if any(ci in current_industry for ci in CONSULTING_INDUSTRIES):
        # If current role is consulting, add its implied months
        if career_history:
            current_job = next((j for j in career_history if j.get("is_current")), None)
            if current_job:
                pass  # already counted above
            else:
                # Approximate 12 months if we can't find current job
                consulting_months += 12
                total_months += 12

    consulting_fraction = consulting_months / total_months if total_months > 0 else 0.0

    # Blend current title (60%) + historical technical score (40%)
    final_score = clamp(0.6 * title_score + 0.4 * hist_tech_score)
    return final_score, is_non_tech, consulting_fraction


def score_experience_years(profile: dict) -> float:
    """
    Score years of experience. JD sweet spot: 5-9 years.
    The JD says '5-9 is a range, not a requirement' but the ideal
    imagined candidate is 6-8 years at product companies.
    """
    years = profile.get("years_of_experience", 0)

    if years < 2:
        return 0.05
    elif years < 3:
        return 0.25
    elif years < 4:
        return 0.45
    elif years < 5:
        return 0.68
    elif years <= 9:
        # Peak zone: scale up to 7yr then gently down
        if years <= 7:
            return 0.88 + (years - 5) * 0.035   # 0.88 → 0.95
        else:
            return 0.95 - (years - 7) * 0.025   # 0.95 → 0.90
    elif years <= 12:
        return 0.80 - (years - 9) * 0.04         # 0.80 → 0.68
    else:
        return max(0.45, 0.68 - (years - 12) * 0.04)


def score_production_vs_research(career_history: list) -> float:
    """
    Assess production deployment experience vs. pure research.
    JD is explicit: 'pure research without production → we will not move forward.'

    Scan career descriptions for production signals (weighted by job duration)
    vs. research-only signals.
    """
    PROD_SIGNALS = [
        "production", "deployed", "shipped", "launched", "live system",
        "real users", "at scale", "billion", "million requests", "million users",
        "latency", "throughput", "serving", "inference server", "inference api",
        "a/b test", "experiment", "rollout", "canary",
        "api", "microservice", "pipeline", "platform", "system design",
        "ranking", "retrieval", "recommendation", "search engine",
        "embedding service", "vector", "index",
        "users", "customers", "traffic", "qps", "rps",
        "fine-tun", "rag", "llm", "model serving",
        "product company", "startup", "product",
    ]
    RESEARCH_ONLY_SIGNALS = [
        "paper", "publication", "conference paper", "journal", "arxiv",
        "phd candidate", "academic", "thesis", "dissertation",
        "pure research", "benchmark only", "research lab", "research-only",
        "no production", "theoretical",
    ]

    prod_score = 0.0
    res_score = 0.0

    for job in career_history:
        desc = job.get("description", "").lower()
        duration = max(1, job.get("duration_months", 1))

        prod_hits = sum(1 for kw in PROD_SIGNALS if kw in desc)
        res_hits = sum(1 for kw in RESEARCH_ONLY_SIGNALS if kw in desc)

        prod_score += prod_hits * duration
        res_score += res_hits * duration

    total = prod_score + res_score
    if total == 0:
        return 0.45  # no description signal — neutral but slightly below midpoint

    return clamp(prod_score / total)


def score_behavioral(signals: dict) -> float:
    """
    Score behavioral engagement signals from the Redrob platform.
    10 sub-signals covering availability, engagement, and reliability.

    Key insight from the JD: 'a perfect-on-paper candidate who hasn't
    logged in for 6 months and has a 5% response rate is, for hiring
    purposes, not actually available.'
    """
    sub_scores = []

    # 1. Open to work (binary — most important availability signal)
    sub_scores.append(1.0 if signals.get("open_to_work_flag", False) else 0.20)

    # 2. Last active date (recency)
    last_active_str = signals.get("last_active_date", "")
    if last_active_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_ago = (TODAY - last_active).days
            if days_ago <= 7:
                sub_scores.append(1.00)
            elif days_ago <= 30:
                sub_scores.append(0.90)
            elif days_ago <= 90:
                sub_scores.append(0.70)
            elif days_ago <= 180:
                sub_scores.append(0.40)
            elif days_ago <= 365:
                sub_scores.append(0.20)
            else:
                sub_scores.append(0.05)
        except ValueError:
            sub_scores.append(0.50)
    else:
        sub_scores.append(0.50)

    # 3. Recruiter response rate (engagement reliability)
    rr = signals.get("recruiter_response_rate", 0.5)
    sub_scores.append(clamp(float(rr)))

    # 4. Interview completion rate (no-ghosters)
    icr = signals.get("interview_completion_rate", 0.5)
    sub_scores.append(clamp(float(icr)))

    # 5. GitHub activity (important for Senior AI Engineer role)
    gh = signals.get("github_activity_score", -1)
    if gh == -1:
        sub_scores.append(0.20)   # No GitHub linked — mild negative for AI Engineer role
    else:
        sub_scores.append(normalize(float(gh), 0, 100))

    # 6. Notice period (JD explicitly wants sub-30 days, can buy out up to 30d)
    notice = signals.get("notice_period_days", 60)
    if notice <= 0:
        sub_scores.append(1.00)
    elif notice <= 15:
        sub_scores.append(0.98)
    elif notice <= 30:
        sub_scores.append(0.90)
    elif notice <= 60:
        sub_scores.append(0.65)
    elif notice <= 90:
        sub_scores.append(0.40)
    elif notice <= 120:
        sub_scores.append(0.20)
    else:
        sub_scores.append(0.08)

    # 7. Profile completeness
    pc = signals.get("profile_completeness_score", 50)
    sub_scores.append(normalize(float(pc), 20, 100))

    # 8. Saved by recruiters in last 30 days (social proof — others find them relevant)
    saved = signals.get("saved_by_recruiters_30d", 0)
    sub_scores.append(min(1.0, saved / 8.0))

    # 9. Skill assessment scores (taking assessments = shows initiative and transparency)
    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        avg_assess = sum(assessments.values()) / len(assessments)
        sub_scores.append(normalize(avg_assess, 30, 90))
    else:
        sub_scores.append(0.25)   # No assessments — mild negative

    # 10. Average response time (fast responders are easier to hire)
    avg_rt = signals.get("avg_response_time_hours", -1)
    if avg_rt < 0:
        sub_scores.append(0.50)   # unknown
    elif avg_rt <= 4:
        sub_scores.append(1.00)
    elif avg_rt <= 24:
        sub_scores.append(0.90)
    elif avg_rt <= 72:
        sub_scores.append(0.65)
    elif avg_rt <= 168:
        sub_scores.append(0.40)
    else:
        sub_scores.append(0.15)

    return sum(sub_scores) / len(sub_scores)


def score_location(profile: dict, signals: dict) -> float:
    """Score location fit for Pune/Noida-focused JD."""
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)

    is_india = country in ("india", "in") or country == ""

    if not is_india:
        return 0.45 if willing_to_relocate else 0.20

    for city in PREFERRED_INDIA_LOCATIONS:
        if city in location:
            if any(p in location for p in ("pune", "noida", "delhi", "gurgaon", "gurugram", "ncr")):
                return 1.0
            return 0.85

    return 0.65 if willing_to_relocate else 0.50


def score_education(education: list) -> float:
    """Score education tier and field relevance."""
    if not education:
        return 0.35

    TIER_SCORES = {"tier_1": 1.0, "tier_2": 0.80, "tier_3": 0.60, "tier_4": 0.40, "unknown": 0.50}
    RELEVANT_FIELDS = {
        "computer science", "cs", "information technology",
        "machine learning", "artificial intelligence", "data science",
        "mathematics", "statistics", "electrical", "electronics",
        "software engineering", "information systems", "engineering",
        "computational", "operations research",
    }

    best = 0.0
    for edu in education:
        tier = edu.get("tier", "unknown")
        field = edu.get("field_of_study", "").lower()
        t_score = TIER_SCORES.get(tier, 0.50)
        field_bonus = 0.15 if any(f in field for f in RELEVANT_FIELDS) else 0.0
        best = max(best, min(1.0, t_score + field_bonus))

    return best


def detect_honeypot(candidate: dict) -> bool:
    """
    Detect profiles with internally impossible data.
    ~80 honeypots in the dataset. >10% in top 100 → submission disqualified.

    Four impossibility checks:
    1. Job duration exceeds time since start date
    2. Expert proficiency + 0 duration on 3+ skills
    3. Claimed years of experience > 2.8x career history total
    4. 20+ skills all marked expert/advanced (keyword stuffer)
    """
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    for job in career_history:
        start_str = job.get("start_date", "")
        stated_duration = job.get("duration_months", 0)
        if start_str and stated_duration > 0:
            try:
                start = datetime.strptime(start_str, "%Y-%m-%d").date()
                max_possible = (TODAY.year - start.year) * 12 + (TODAY.month - start.month) + 3
                if stated_duration > max_possible + 6:
                    return True
            except ValueError:
                pass

    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    )
    if expert_zero >= 3:
        return True

    claimed_yrs = profile.get("years_of_experience", 0)
    actual_months = sum(j.get("duration_months", 0) for j in career_history)
    actual_yrs = actual_months / 12.0
    if claimed_yrs > 3 and actual_yrs > 0 and claimed_yrs > actual_yrs * 2.8:
        return True

    high_prof_count = sum(1 for s in skills if s.get("proficiency") in ("expert", "advanced"))
    if high_prof_count >= 20:
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# MASTER SCORING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_score(candidate: dict) -> tuple:
    """
    Compute the final weighted score and generate a specific reasoning string.

    Component weights (sum = 1.0):
      Skills match         0.28
      Title/career fit     0.22
      Production exp       0.20
      Years of experience  0.12
      Behavioral signals   0.10
      Location fit         0.06
      Education            0.02

    Disqualifier multipliers applied after weighted sum.
    Returns: (score: float [0-1], reasoning: str)
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    signals = candidate.get("redrob_signals", {})

    # ── Honeypot guard ───────────────────────────────────────────────────────
    if detect_honeypot(candidate):
        return 0.0, "Profile flagged as honeypot: internally inconsistent data (impossible tenure or inflated experience)."

    # ── Build extra text for skills scorer (summary + headline) ─────────────
    summary = profile.get("summary", "")
    headline = profile.get("headline", "")
    extra_text = f"{headline} {summary}"

    # ── Component scores ─────────────────────────────────────────────────────
    ai_skill_score, disq_frac = score_skills(skills, extra_text)
    title_score, is_non_tech, consult_frac = score_title_fit(profile, career)
    prod_score = score_production_vs_research(career)
    exp_score = score_experience_years(profile)
    behavioral_score = score_behavioral(signals)
    location_score = score_location(profile, signals)
    edu_score = score_education(education)

    # ── Disqualifier penalty multipliers ────────────────────────────────────
    penalty = 1.0

    # Consulting-only career — JD explicitly disqualifies this
    if consult_frac >= 0.95:
        penalty *= 0.12    # Entire career at IT services
    elif consult_frac >= 0.80:
        penalty *= 0.28
    elif consult_frac >= 0.65:
        penalty *= 0.50
    elif consult_frac >= 0.50:
        penalty *= 0.72

    # Non-technical title throughout career
    if is_non_tech and title_score < 0.15:
        penalty *= 0.18

    # CV/Speech/Robotics primary focus — wrong domain
    if disq_frac > 0.65:
        penalty *= 0.22
    elif disq_frac > 0.45:
        penalty *= 0.52

    # Pure researcher with no production signal
    if prod_score < 0.10 and ai_skill_score < 0.15:
        penalty *= 0.38

    # ── Weighted sum ─────────────────────────────────────────────────────────
    raw = (
        0.28 * ai_skill_score
        + 0.22 * title_score
        + 0.20 * prod_score
        + 0.12 * exp_score
        + 0.10 * behavioral_score
        + 0.06 * location_score
        + 0.02 * edu_score
    )

    final_score = clamp(raw * penalty)

    # ── Reasoning (specific facts, JD-connected, honest about concerns) ──────
    years = profile.get("years_of_experience", 0)
    curr_title = profile.get("current_title", "N/A")
    curr_company = profile.get("current_company", "")
    location = profile.get("location", "N/A")
    country = profile.get("country", "")
    notice = signals.get("notice_period_days", 0)
    rr = signals.get("recruiter_response_rate", 0)
    otw = signals.get("open_to_work_flag", False)
    gh = signals.get("github_activity_score", -1)
    last_active_str = signals.get("last_active_date", "")

    # Pick the top relevant skill names from skills list
    rel_skills = [
        s.get("name", "") for s in skills
        if any(kw in s.get("name", "").lower() or s.get("name", "").lower() in kw
               for kw in REQUIRED_SKILLS)
    ]
    top_skills_str = ", ".join(rel_skills[:3]) if rel_skills else "general engineering"

    # Location string
    loc_str = location
    if country and country.lower() not in ("india", "in", ""):
        loc_str = f"{location}, {country}"

    # Collect strengths
    strengths = []
    if ai_skill_score >= 0.50:
        strengths.append("strong AI/ML skill match")
    if prod_score >= 0.65:
        strengths.append("clear production deployment evidence")
    if title_score >= 0.60:
        strengths.append("relevant technical title")
    if gh >= 40:
        strengths.append(f"active GitHub (score {gh:.0f})")
    if otw:
        strengths.append("actively open to work")
    if notice <= 30:
        strengths.append(f"{notice}d notice")

    # Collect concerns (be honest — Stage 4 checks for this)
    concerns = []
    if consult_frac > 0.65:
        concerns.append(f"consulting-heavy career ({consult_frac:.0%} in IT services)")
    if is_non_tech:
        concerns.append("non-technical current title")
    if disq_frac > 0.40:
        concerns.append("CV/Speech/Robotics focus vs NLP/IR required")
    if notice > 60:
        concerns.append(f"{notice}d notice period")
    if not otw:
        concerns.append("not marked open-to-work")
    if last_active_str:
        try:
            la = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days_ago = (TODAY - la).days
            if days_ago > 90:
                concerns.append(f"inactive {days_ago}d")
        except ValueError:
            pass
    if rr < 0.25:
        concerns.append(f"low recruiter response rate ({rr:.0%})")

    # Assemble reasoning (specific, factual, not templated)
    company_str = f" at {curr_company}" if curr_company else ""
    parts = [f"{curr_title}{company_str}, {years:.1f}yr exp, {loc_str}."]
    parts.append(f"Key skills: {top_skills_str}.")
    if strengths:
        parts.append(f"Strengths: {'; '.join(strengths[:2])}.")
    if concerns:
        parts.append(f"Concerns: {'; '.join(concerns[:2])}.")
    parts.append(f"Response rate {rr:.0%}, notice {notice}d.")

    reasoning = " ".join(parts)
    if len(reasoning) > 400:
        reasoning = reasoning[:397] + "..."

    return final_score, reasoning


# ─────────────────────────────────────────────────────────────────────────────
# RANKING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def rank_candidates(input_path: str, output_path: str):
    """Stream candidates from JSONL, score all, output top-100 CSV."""
    path = Path(input_path)
    if not path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[Bug Hunters Ranker] Input : {input_path}")
    print(f"[Bug Hunters Ranker] Output: {output_path}")
    print("[Bug Hunters Ranker] Scoring candidates...")

    import gzip
    if path.suffix == ".gz":
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = lambda: open(path, "r", encoding="utf-8")

    all_scores = []
    count = 0

    with opener() as f:
        raw = f.read()

    raw = raw.strip()
    if raw.startswith("["):
        # JSON array (e.g. sample_candidates.json)
        try:
            candidates_list = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"ERROR: Could not parse JSON array: {e}", file=sys.stderr)
            sys.exit(1)
        for c in candidates_list:
            cid = c.get("candidate_id", "")
            if not cid:
                continue
            score, reasoning = compute_score(c)
            all_scores.append((cid, score, reasoning))
            count += 1
    else:
        # JSONL — process line by line (memory efficient for 100K candidates)
        with opener() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = c.get("candidate_id", "")
                if not cid:
                    continue
                score, reasoning = compute_score(c)
                all_scores.append((cid, score, reasoning))
                count += 1
                if count % 10_000 == 0:
                    print(f"  ...scored {count:,} candidates", flush=True)

    print(f"[Bug Hunters Ranker] Scored {count:,} total.")

    # Sort: descending score, ascending candidate_id for tie-breaks (spec requirement)
    all_scores.sort(key=lambda x: (-x[1], x[0]))
    top_100 = all_scores[:100]

    # Write CSV
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score, reasoning) in enumerate(top_100, start=1):
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])

    print(f"\n[Bug Hunters Ranker] Written: {out_path}")
    print("\nTop 10:")
    print(f"{'Rank':>4}  {'Candidate':14}  {'Score':>7}  Reasoning (preview)")
    print("-" * 90)
    for rank, (cid, score, reasoning) in enumerate(top_100[:10], start=1):
        print(f"{rank:>4}  {cid:14}  {score:>7.4f}  {reasoning[:60]}...")
    if top_100:
        print(f"\nScore range: [{top_100[-1][1]:.4f} – {top_100[0][1]:.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Bug Hunters — Redrob Hackathon Candidate Ranker\n"
                    "Produces top-100 submission CSV for the Senior AI Engineer JD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--candidates", required=True,
                        help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True,
                        help="Output path for submission CSV")
    args = parser.parse_args()
    rank_candidates(args.candidates, args.out)


if __name__ == "__main__":
    main()

