import re
from .constants import TODAY, NON_TECH_TITLES, TIER1_REGEX, TIER2_REGEX, STRONG_TITLE_SCORES, ELITE_SEARCH_COMPANIES, FAST_WORDS_RE, SHIP_FAST_RE, PRIMARY_CORE_RE
from .utils import calculate_actual_yoe, parse_date

def is_honeypot(candidate: dict) -> bool:
    profile     = candidate.get("profile", {}) or {}
    career      = candidate.get("career_history", []) or []
    skills      = candidate.get("skills", []) or []
    claimed_yoe = float(profile.get("years_of_experience", 0) or 0)
    actual_yoe  = calculate_actual_yoe(career)
    
    suspicion_score = 0

    # 1. Chronological Impossibility (Instant fail +4)
    for job in career:
        start_dt = parse_date(job.get("start_date", ""))
        stated   = int(job.get("duration_months", 0) or 0)
        if start_dt and stated > 0:
            max_possible = (TODAY.year - start_dt.year) * 12 + (TODAY.month - start_dt.month) + 3
            if stated > max_possible + 6:
                suspicion_score += 4
                break

    # 2. Career Math Discrepancies
    if actual_yoe > claimed_yoe + 3.0 and actual_yoe > 10.0:
        suspicion_score += 1
    if 0 < claimed_yoe < 10 and any(int(j.get("duration_months", 0) or 0) > 240 for j in career):
        suspicion_score += 2

    # 3. Skill Spamming / Domain Conflict
    expert_advanced = [s for s in skills if str(s.get("proficiency", "")).lower() in ("expert", "advanced")]
    if len(expert_advanced) >= 20:
        zero_evidence = sum(
            1 for s in expert_advanced
            if float(s.get("duration_months", 0) or 0) <= 0.0 and int(s.get("endorsements", 0) or 0) == 0
        )
        if zero_evidence >= 10:
            suspicion_score += 2
        elif zero_evidence >= 20:
            suspicion_score += 3
            
        names = " ".join(s.get("name", "").lower() for s in expert_advanced)
        has_non_tech = any(re.search(rf"\b{kw}\b", names) for kw in ("accounting", "tally", "sales", "hr", "marketing", "seo", "content writing"))
        has_tech     = any(re.search(rf"\b{kw}\b", names) for kw in ("machine learning", "backend", "react", "python", "aws"))
        if has_non_tech and has_tech:
            suspicion_score += 1

    # 4. Technology Time-Travel & 5. Skill Exceeds Career
    max_excess_penalty = 0
    total_career_months = actual_yoe * 12
    
    for s in skills:
        name = s.get("name", "").lower()
        dur  = int(s.get("duration_months", 0) or 0)
        
        # Tech Time-Travel
        if "langchain" in name or "openai" in name or "chatgpt" in name or "llama" in name:
            if dur > 120: suspicion_score += 4
            elif dur > 48: suspicion_score += 2
        elif "qdrant" in name or "weaviate" in name or "pinecone" in name:
            if dur > 120: suspicion_score += 4
            elif dur > 84: suspicion_score += 2
            
        # Skill Exceeds Career
        if total_career_months > 0:
            excess = dur - total_career_months
            if excess > 60:
                max_excess_penalty = max(max_excess_penalty, 2)
            elif excess > 24:
                max_excess_penalty = max(max_excess_penalty, 1)

    suspicion_score += max_excess_penalty

    return suspicion_score >= 4

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
