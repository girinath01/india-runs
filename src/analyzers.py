import math
import re
from datetime import datetime
from .constants import *
from .constants import _SPEC_ML_KW, _SPEC_RETRIEVAL_KW, _SPEC_BACKEND_KW, _SPEC_DATA_KW
from .schemas import *
from .utils import calculate_actual_yoe, _split_sentences, _get_career_text, parse_date
from .semantic_engine import compute_semantic_scores, detect_negation, detect_weak_context, is_model_available
from .ner_engine import extract_companies, has_senior_title

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
                
                curr_is_senior = has_senior_title(curr_t)
                prev_is_senior = has_senior_title(prev_t)
                
                if curr_is_senior and not prev_is_senior:
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
            
        # Use NER to extract implicit companies mentioned in descriptions (e.g. B2B clients, partners)
        orgs = extract_companies(desc)
        for org in orgs:
            if any(ec in org for ec in ELITE_SEARCH_COMPANIES) and not elite_hit:
                elite_hit = True
                evidence.append(Evidence("company", f"Collaborated with/built for {org.title()} (elite search company)", priority=6))
            if any(pc in org for pc in PRODUCT_TECH_COMPANIES):
                evidence.append(Evidence("company", f"Collaborated with/built for {org.title()} (product tech company)", priority=4))

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
    if   notice <= 30: notice_bonus = 2   # Sub-30 loved, can buy out 30
    elif notice <= 60: notice_bonus = -1  # 30+ in scope but bar is higher
    else:              notice_bonus = -2

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
    
    is_preferred = any(city in location for city in ("pune", "noida", "greater noida"))
    is_welcome   = any(city in location for city in ("hyderabad", "mumbai", "delhi", "gurgaon", "gurugram", "ncr", "new delhi", "faridabad"))
    
    if is_preferred:        relocation = 2
    elif is_welcome:        relocation = 1
    elif is_india:          relocation = 0
    elif willing:           relocation = -1
    else:                   relocation = -2

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


def analyze_semantic(c: Candidate) -> SemanticFeatures:
    """
    SemanticAnalyzer v1.0 — embedding-based understanding.

    Uses sentence-transformers to compute cosine similarity between candidate
    career text and JD target phrases. This catches synonyms, detects negation,
    and scores contextual depth that pure regex cannot.

    Returns:
        SemanticFeatures with per-domain similarity scores and a combined score (0–15).
    """
    if not is_model_available():
        return SemanticFeatures(model_available=False)

    full_text = _get_career_text(c, max_jobs=5)
    if len(full_text.strip()) < 30:
        return SemanticFeatures(model_available=True)

    # Compute semantic similarity for ALL domains at once
    scores = compute_semantic_scores(full_text)
    if not scores:
        return SemanticFeatures(model_available=True)

    # Count negation and weak context in career text
    sentences = _split_sentences(full_text.lower())
    neg_count = sum(1 for s in sentences if detect_negation(s))
    weak_count = sum(1 for s in sentences if detect_weak_context(s))

    # Extract the per-domain similarities
    search_sim = scores.get("search", 0.0)
    rec_sim = scores.get("recommendation", 0.0)
    prod_sim = scores.get("production", 0.0)
    own_sim = scores.get("ownership", 0.0)
    impact_sim = scores.get("impact", 0.0)
    mkt_sim = scores.get("marketplace", 0.0)
    eval_sim = scores.get("evaluation", 0.0)
    disq_sim = scores.get("disqualifying", 0.0)

    # Compute combined score: weighted by JD importance (0–15 range)
    # Weights reflect what the JD values most:
    #   search/recommendation: highest (core role)
    #   production/ownership: high (shipper > researcher)
    #   marketplace/evaluation: moderate (bonus alignment)
    #   disqualifying: negative (wrong domain)
    combined = (
        search_sim * 4.0          # 0–4.0
        + rec_sim * 3.5           # 0–3.5
        + prod_sim * 2.5          # 0–2.5
        + own_sim * 1.5           # 0–1.5
        + impact_sim * 1.0        # 0–1.0
        + mkt_sim * 1.5           # 0–1.5
        + eval_sim * 1.0          # 0–1.0
        - disq_sim * 3.0          # penalty for CV/speech/robotics alignment
    )
    combined = max(0.0, min(15.0, combined))

    # Build evidence from semantic findings
    evidence = []
    if search_sim > 0.45:
        evidence.append(Evidence("retrieval",
            f"Semantic analysis confirms strong search/retrieval alignment (similarity: {search_sim:.2f})",
            priority=9))
    if rec_sim > 0.45:
        evidence.append(Evidence("recommendation",
            f"Semantic analysis confirms recommendation system experience (similarity: {rec_sim:.2f})",
            priority=9))
    if prod_sim > 0.45:
        evidence.append(Evidence("production",
            f"Semantic analysis confirms production deployment experience (similarity: {prod_sim:.2f})",
            priority=8))
    if disq_sim > 0.50:
        evidence.append(Evidence("skill",
            f"Semantic analysis flags strong CV/speech/robotics alignment (similarity: {disq_sim:.2f})",
            priority=3))

    return SemanticFeatures(
        search_sim=search_sim,
        recommendation_sim=rec_sim,
        production_sim=prod_sim,
        ownership_sim=own_sim,
        impact_sim=impact_sim,
        marketplace_sim=mkt_sim,
        evaluation_sim=eval_sim,
        disqualifying_sim=disq_sim,
        combined_score=combined,
        negation_count=neg_count,
        weak_context_count=weak_count,
        model_available=True,
        evidence=evidence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — FEATURE VECTOR MERGER
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(c: Candidate, skip_semantic: bool = False) -> FeatureVector:
    """Run all analyzers → merge into FeatureVector.
    If skip_semantic is True, returns an empty SemanticFeatures object to save time.
    """
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
    
    if skip_semantic:
        semantic = SemanticFeatures(model_available=False)
    else:
        semantic = analyze_semantic(c)

    # Merge all evidence for reasoning (sorted by priority, highest first)
    all_evidence = (
        career.evidence + company.evidence + skills.evidence +
        ev.evidence + ownership.evidence + impact.evidence +
        production.evidence + jd_intent.evidence + evaluation.evidence +
        semantic.evidence
    )
    all_evidence.sort(key=lambda e: e.priority, reverse=True)

    return FeatureVector(
        career=career, company=company, skills=skills,
        evidence=ev, ownership=ownership, impact=impact,
        production=production, jd_intent=jd_intent,
        evaluation=evaluation, trajectory=trajectory,
        hiring=hiring, domain_tenure=domain_tenure,
        semantic=semantic,
        all_evidence=all_evidence,
    )
