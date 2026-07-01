import re
import hashlib
from .constants import NON_TECH_TITLES, RESEARCH_HEAVY_RE
from .schemas import Candidate, FeatureVector
from .utils import _categorise_skill

def compute_penalties(c: Candidate, fv: FeatureVector) -> float:
    """
    Dedicated penalty stage — reduces the final score via a multiplier.

    Penalties are NOT mixed into positive analyzers.
    Each penalty represents an explicit JD disqualifier.
    """
    multiplier = 1.0
    title = c.current_title.lower()

    # P1: Non-tech title — marketing/HR/finance with AI keywords
    if any(title == t or title.startswith(t + " ") or title.startswith(t + ",") for t in NON_TECH_TITLES):
        multiplier *= 0.40

    # P2: Junior title — JD needs 5-9 yr seniority
    if any(p in title for p in ("junior", "jr.", "intern", "trainee")) and "senior" not in title:
        multiplier *= 0.92

    # P3: Services-only career — JD explicitly warns about consulting
    if fv.company.consult_fraction > 0.90:
        multiplier *= 0.40
    elif fv.career.service_years > 0 and fv.career.product_years + fv.career.startup_years == 0:
        multiplier *= 0.94

    # P_DOMAIN: Wrong domain (CV/Speech/Robotics)
    if fv.skills.disq_fraction > 0.40:
        multiplier *= 0.40
    elif fv.skills.disq_fraction > 0.25 and fv.jd_intent.score < 10:
        multiplier *= 0.40
    elif fv.skills.disq_fraction > 0.25:
        multiplier *= 0.50
        
    # Broadened CV Title check
    cv_match = re.search(r'\b(computer vision|cv|vision|perception|image processing|robotics|speech)\b', title)
    if cv_match:
        if fv.jd_intent.score < 10:
            multiplier *= 0.40

    # P4: Research-only — JD: "tried it twice, didn't work"
    if fv.production.research_only:
        multiplier *= 0.40

    # P_PROD: No shipped production experience (ignoring generic title boost)
    if fv.production.ship_count == 0:
        multiplier *= 0.40
        
    # P_NO_DOMAIN_PROD: Lacking production retrieval/ranking systems
    if not fv.jd_intent.search_hit and not fv.jd_intent.recommendation_hit:
        if fv.production.score < 5:
            multiplier *= 0.40

    # P5: Prompt-engineer-only — JD: "If your experience consists of LangChain + OpenAI"
    skill_names = " ".join(s.get("name", "").lower() for s in c.skills)
    has_tier1 = fv.skills.tier1_count > 0
    llm_hype  = any(re.search(rf"\b{kw}\b", skill_names)
                     for kw in ("langchain", "prompt engineering", "openai", "chatgpt"))
    if not has_tier1 and llm_hype:
        multiplier *= 0.90

    # P6: Fabricated experience — claimed >> actual
    if c.years_of_experience > 3 and fv.career.actual_yoe > 0:
        if c.years_of_experience > fv.career.actual_yoe * 2.8:
            multiplier *= 0.90

    # P7: Job hopper — avg tenure < 18 months in recent jobs
    if len(c.career) >= 3:
        recent_jobs = c.career[:3]
        total_months = sum(max(1, int(j.get("duration_months", 1) or 1)) for j in recent_jobs)
        avg_tenure   = total_months / len(recent_jobs)
        curr_dur     = int(c.career[0].get("duration_months", 1) or 1) if c.career else 0
        if avg_tenure < 18 and curr_dur < 36:
            multiplier *= 0.94

    # P8: Research-heavy career
    if c.career:
        research_heavy = sum(
            1 for job in c.career
            if len(set(RESEARCH_HEAVY_RE.findall((job.get("description", "") or "").lower()))) >= 2
        )
        if research_heavy >= len(c.career) * 0.60:
            multiplier *= 0.95

    # P9: No GitHub — 5+ YoE, closed-source, not research
    github = int(c.signals.get("github_activity_score", 0) or 0)
    if fv.career.actual_yoe > 5 and github <= 0 and not fv.production.research_only:
        multiplier *= 0.96

    # P10: Trajectory penalty (from trajectory analyzer)
    if fv.trajectory.penalty > 0:
        multiplier *= max(0.10, 1.0 - (fv.trajectory.penalty / 100.0))

    return multiplier


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — TWO-STAGE SCORING ENGINE  (Step 1 + Step 3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_score(fv: FeatureVector, penalty_multiplier: float) -> float:
    """
    Two-Stage Scored Pipeline with Semantic Understanding:

    1. Technical Fit = sum of all technical component scores + semantic score + synergy bonuses
    2. Hiring Readiness = capped behavioral score
    3. final = technical × 0.90 + hiring × 0.10 − penalties

    The semantic score adds meaning-based understanding that catches synonyms,
    penalizes negation, and rewards contextual depth beyond keyword matching.
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

    # ── NEW: Semantic Understanding Score ─────────────────────────────
    # The semantic engine provides a combined score (0-15) based on
    # cosine similarity between candidate text and JD target phrases.
    # This catches synonyms regex misses and penalizes negated mentions.
    semantic_score = fv.semantic.combined_score if fv.semantic.model_available else 0.0

    # Cross-validation: when regex and semantic AGREE, boost confidence.
    # When they DISAGREE, apply corrections.
    if fv.semantic.model_available:
        # Case 1: Regex found domain keywords but semantic says it's weak context
        #         → reduce the score (candidate just mentioned words, no real depth)
        if fv.jd_intent.search_hit and fv.semantic.search_sim < 0.25:
            semantic_score -= 2.0  # regex false positive correction
        if fv.jd_intent.recommendation_hit and fv.semantic.recommendation_sim < 0.25:
            semantic_score -= 2.0

        # Case 2: Regex missed keywords but semantic found strong alignment
        #         → the candidate used different words to describe the same work
        if not fv.jd_intent.search_hit and fv.semantic.search_sim > 0.55:
            semantic_score += 3.0  # reward synonym/paraphrase understanding
        if not fv.jd_intent.recommendation_hit and fv.semantic.recommendation_sim > 0.55:
            semantic_score += 3.0

        # Case 3: Negation penalty — candidate mentioned domain terms but negated them
        if fv.semantic.negation_count >= 2:
            semantic_score -= 3.0  # multiple negations = misleading profile
        elif fv.semantic.negation_count >= 1:
            semantic_score -= 1.5

        # Case 4: Disqualifying domain alignment — semantic confirms wrong domain
        if fv.semantic.disqualifying_sim > 0.50:
            semantic_score -= 4.0  # strong CV/speech/robotics focus

        semantic_score = max(0.0, min(15.0, semantic_score))

    technical += semantic_score

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

    # === NEW: Semantic Synergy Bonuses ===
    # When BOTH regex and semantic strongly agree, it's very high confidence
    if fv.semantic.model_available:
        if fv.jd_intent.search_hit and fv.semantic.search_sim > 0.55:
            synergy += 3  # regex + semantic double-confirm search expertise
        if fv.jd_intent.recommendation_hit and fv.semantic.recommendation_sim > 0.55:
            synergy += 3  # regex + semantic double-confirm recommendation expertise
        if fv.production.score >= 10 and fv.semantic.production_sim > 0.50:
            synergy += 2  # strong production confirmed semantically

    technical += synergy

    # ── Hiring Readiness ──────────────────────────────────────────────
    hiring = fv.hiring.score     # -3 to +8

    # -- Final: capped technical + scaled hiring - penalties ----------
    # Cap technical at 95 so hiring readiness is always visible even for
    # the strongest candidates. Hiring: -3..+8 maps to -1.875..+5.0.
    # Maximum possible: 95 + 5 - 0 = 100 (perfect technical + perfect hiring).
    # -- Technical cap: 95.
    # We added domain_tenure (+10), JD recency (+4), and semantic (+15) to
    # the technical scale. Raw technical scores can now reach ~170.
    # Multiplier adjusted to 0.55 so only the top 1% hit the ceiling.
    tech_contrib   = min(95.0, technical * 0.55)
    hiring_contrib = (hiring / 8.0) * 5.0    # -1.875 to +5.0

    final = (tech_contrib + hiring_contrib) * penalty_multiplier

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
    penalty_multiplier: float,
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
        import random
        rng = random.Random(seed)
        
        company_str = f" at {company.title()}" if company else ""
        if final_score >= 60:
            openers = [
                f"With {yoe:g} years of experience, this {title.title()}{company_str} shows strong alignment.",
                f"A standout {title.title()}{company_str} bringing {yoe:g} years of relevant experience.",
                f"This {yoe:g}-year veteran {title.title()}{company_str} presents a highly compelling profile.",
                f"An exceptional {title.title()}{company_str} whose {yoe:g} years of background strongly fit the JD."
            ]
        elif final_score >= 35:
            openers = [
                f"A capable {title.title()}{company_str} with {yoe:g} years of experience.",
                f"This profile features a {title.title()}{company_str} possessing {yoe:g} years of background.",
                f"An experienced {title.title()}{company_str} ({yoe:g} years) with partial JD alignment.",
                f"Bringing {yoe:g} years to the table, this {title.title()}{company_str} meets some key requirements."
            ]
        else:
            openers = [
                f"A {title.title()}{company_str} with {yoe:g} years of experience.",
                f"This candidate is a {title.title()}{company_str} ({yoe:g} years) but lacks core alignment.",
                f"While possessing {yoe:g} years of background, this {title.title()}{company_str} has limited JD fit."
            ]
        
        opener = rng.choice(openers)
        
        # Evidence sentences — each maps to a JD requirement
        evidence_phrases = []
        for ev in unique_evidence:
            jd_link = _EVIDENCE_JD_MAP.get(ev.category, "relevant to the role")
            sentence = ev.sentence.strip().rstrip('.')
            if len(sentence) > 180:
                sentence = sentence[:180].rsplit(" ", 1)[0]
            evidence_phrases.append(f"{sentence.lower()} (which {jd_link})")
            
        if len(evidence_phrases) == 1:
            connectors = ["This is evidenced by", "We noted", "A key strength is"]
            parts.append(f"{opener} {rng.choice(connectors)} {evidence_phrases[0]}.")
        elif len(evidence_phrases) == 2:
            parts.append(f"{opener} Highlights include {evidence_phrases[0]}, as well as {evidence_phrases[1]}.")
        else:
            parts.append(f"{opener} Key indicators include {evidence_phrases[0]}, {evidence_phrases[1]}, and {evidence_phrases[2]}.")
    else:
        # No evidence extracted — use minimal factual statement
        company_str = f" at {company.title()}" if company else ""
        parts.append(f"A {title.title()}{company_str} with {yoe:g} years of experience. The profile lacks strong retrieval, search, or ranking evidence.")

    # ── Biggest gap ───────────────────────────────────────────────────
    gaps = []
    
    # Check for chronological inconsistencies
    time_travel = False
    for s in c.skills:
        name = s.get("name", "").lower()
        dur = int(s.get("duration_months", 0) or 0)
        if dur > 48 and any(kw in name for kw in ("langchain", "openai", "chatgpt", "llama")): time_travel = True
        if dur > 84 and any(kw in name for kw in ("qdrant", "weaviate", "pinecone")): time_travel = True
        
    total_career_months = fv.career.actual_yoe * 12
    if time_travel:
        gaps.append("profile claims impossible skill durations (exceeds existence of technology)")
    elif total_career_months > 0:
        max_skill_dur = max([int(s.get("duration_months", 0) or 0) for s in c.skills] + [0])
        if max_skill_dur > total_career_months + 36:
            gaps.append("minor date inconsistencies (skill durations significantly exceed career length)")

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

    if gaps and (final_score < 85 or "impossible" in gaps[0] or "inconsistencies" in gaps[0]):
        gap_str = gaps[0] if len(gaps) == 1 else f"{gaps[0]} and {gaps[1]}"
        parts.append(f"However, we note that {gap_str}.")
        
    # Hard Rejection override (now Major Penalty)
    if penalty_multiplier <= 0.6:
        if gaps and ("no production" in gaps[0] or fv.production.score == 0):
            parts.append("Major Penalty: No evidence of shipping to production.")
        elif fv.production.research_only:
            parts.append("Major Penalty: Pure research background.")
        elif fv.company.consult_fraction > 0.90:
            parts.append("Major Penalty: Almost entirely consulting background.")
        elif any(title == t or title.startswith(t + " ") or title.startswith(t + ",") for t in NON_TECH_TITLES):
            parts.append(f"Major Penalty: Non-technical role ({title}).")
        elif fv.production.ship_count == 0:
            parts.append("Major Penalty: No evidence of shipping ML systems to production.")
        elif not fv.jd_intent.search_hit and not fv.jd_intent.recommendation_hit and fv.production.score < 5:
            parts.append("Major Penalty: Lacks required production retrieval/ranking domain experience.")
        elif re.search(r'\b(computer vision|cv|vision|perception|image processing|robotics|speech)\b', title) or fv.skills.disq_fraction > 0.25:
            parts.append("Major Penalty: CV/Speech specialist without core retrieval experience.")
        else:
            parts.append("Major Penalty: Disqualified due to massive penalties.")

    # ── Quick-join indicator ──────────────────────────────────────────
    if notice_days <= 30 and fv.hiring.open_to_work and final_score >= 40:
        parts.append(["Available quickly.", "Ready to interview.", "Immediate joiner."][seed % 3])

    return " ".join(parts), category_set


# ─────────────────────────────────────────────────────────────────────────────
# DEEP SCORER  (Pass 2 — called only on 12K shortlist)
# ─────────────────────────────────────────────────────────────────────────────
