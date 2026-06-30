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
        gap_prefix = ["Gap: ", "However, ", "Concern: ", "Note: "][seed % 4]
        parts.append(gap_prefix + gaps[0] + ".")
        
    # Hard Rejection override (now Major Penalty)
    if penalty_multiplier <= 0.6:
        if gaps and ("no production" in gaps[0] or fv.production.score == 0):
            parts = ["Major Penalty: No evidence of shipping to production."]
        elif fv.production.research_only:
            parts = ["Major Penalty: Pure research background."]
        elif fv.company.consult_fraction > 0.90:
            parts = ["Major Penalty: Almost entirely consulting background."]
        elif any(title == t or title.startswith(t + " ") or title.startswith(t + ",") for t in NON_TECH_TITLES):
            parts = [f"Major Penalty: Non-technical role ({title})."]
        elif fv.production.ship_count == 0:
            parts = ["Major Penalty: No evidence of shipping ML systems to production."]
        elif not fv.jd_intent.search_hit and not fv.jd_intent.recommendation_hit and fv.production.score < 5:
            parts = ["Major Penalty: Lacks required production retrieval/ranking domain experience."]
        elif re.search(r'\b(computer vision|cv|vision|perception|image processing|robotics|speech)\b', title) or fv.skills.disq_fraction > 0.25:
            parts = ["Major Penalty: CV/Speech specialist without core retrieval experience."]
        else:
            parts = ["Major Penalty: Disqualified due to massive penalties."]

    # ── Quick-join indicator ──────────────────────────────────────────
    if notice_days <= 30 and fv.hiring.open_to_work and final_score >= 40:
        parts.append(["Available quickly.", "Ready to interview.", "Immediate joiner."][seed % 3])

    return " ".join(parts), category_set


# ─────────────────────────────────────────────────────────────────────────────
# DEEP SCORER  (Pass 2 — called only on 12K shortlist)
# ─────────────────────────────────────────────────────────────────────────────
