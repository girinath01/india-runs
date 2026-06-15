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


if __name__ == "__main__":
    print("Experience, production, and behavioral scorers loaded OK.")
