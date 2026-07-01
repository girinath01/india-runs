from dataclasses import dataclass, field
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
class SemanticFeatures:
    """Output of SemanticAnalyzer. Cosine similarity scores per JD domain (0.0–1.0)."""
    search_sim: float = 0.0           # similarity to search/retrieval target
    recommendation_sim: float = 0.0   # similarity to recommendation target
    production_sim: float = 0.0       # similarity to production deployment target
    ownership_sim: float = 0.0        # similarity to ownership/leadership target
    impact_sim: float = 0.0           # similarity to measurable impact target
    marketplace_sim: float = 0.0      # similarity to marketplace/matching target
    evaluation_sim: float = 0.0       # similarity to evaluation methodology target
    disqualifying_sim: float = 0.0    # similarity to disqualifying domains (CV/speech/robotics)
    combined_score: float = 0.0       # weighted combination → 0–15 score range
    negation_count: int = 0           # sentences with negated domain mentions
    weak_context_count: int = 0       # sentences with only weak/surface context
    model_available: bool = False     # whether the semantic model was loaded
    evidence: list = field(default_factory=list)


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
    semantic:      SemanticFeatures = field(default_factory=SemanticFeatures)
    all_evidence: list = field(default_factory=list)  # merged evidence for reasoning
