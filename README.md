# Redrob Hackathon — Bug Hunters

Ranking system for the **Intelligent Candidate Discovery & Ranking Challenge**
by Redrob AI. Produces a top-100 submission CSV for the Senior AI Engineer JD.

## Reproduce Command

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./bug_hunters.csv
python validate_submission.py bug_hunters.csv
```

Accepts `.jsonl` (uncompressed) and `.jsonl.gz` (gzipped) input.  
**Runtime: ~40 seconds** for 100,000 candidates on CPU (two-pass pipeline).  
**Memory: ~2–3 GB** (streams JSONL, never loads full dataset at once).

---

## Scoring Formula

```
FinalScore = 0.35×SkillFit + 0.25×ProductFit + 0.20×Behavioral
           + 0.10×ExperienceFit + 0.10×LocationNoticeFit
           − HardPenalties
```

### Component Details

| Component | Weight | Key Logic |
|---|---|---|
| **Skill Fit** | 35% | Tiered keyword scoring (see below). Also scans profile summary + headline to catch plain-language candidates who describe experience in prose. |
| **Product Fit** | 25% | Title quality (Search/Ranking/Applied ML > generic DS/Research) + consulting career penalty via `industry` field + production shipping evidence from career descriptions. |
| **Behavioral Signals** | 20% | 12 platform sub-signals: open-to-work, response rate, activity recency, interview completion, GitHub activity, recruiter saves, profile completeness, assessment scores, response time, active applications, verified contacts, LinkedIn. |
| **Experience Fit** | 10% | 6–8yr sweet spot curve + **seniority modifier from title** (Lead/Staff +12%, Junior −22%). |
| **Location + Notice** | 10% | India + Pune/Noida = 1.0; other Tier-1 India = 0.85; overseas = 0.18–0.50. Notice: ≤30d = 0.95, ≤60d = 0.55, ≥120d = 0.04. |

### Tiered Skill Scoring (the core differentiator)

The JD explicitly warns: *"If your experience consists of LangChain calling OpenAI — probably not."*

| Tier | Contribution | Skills |
|---|---|---|
| **Tier 1** | 65% of skill score | FAISS, Qdrant, Pinecone, Elasticsearch, OpenSearch, Weaviate, Milvus, RAG, retrieval, semantic search, vector search, ranking, recommendation, LTR, NDCG, A/B testing |
| **Tier 2** | 25% of skill score | NLP, BERT, PyTorch, TensorFlow, MLOps, MLflow, Spark, Docker, Kubernetes, AWS/GCP/Azure |
| **Tier 3** | 10% of skill score | LangChain, Prompt Engineering, QLoRA, LoRA, OpenAI, ChatGPT — trendy LLM tooling without retrieval/search context |
| **Disqualifying** | Penalty applied | Computer Vision, Speech Recognition, Robotics, SLAM — wrong domain |

### Hard Penalties

Subtracted directly from the final score (caps at −0.52):

| Condition | Penalty |
|---|---|
| Non-technical title (Marketing Manager, HR, etc.) | −0.35 |
| Research-only career (papers, thesis, no production evidence) | −0.18 |
| Inactive > 180 days on platform | −0.18 |
| Junior title | −0.15 |
| Recruiter response rate < 10% | −0.15 |
| LLM-only skills with zero Tier-1 retrieval/search skills | −0.14 |
| Inactive 90–180 days | −0.09 |
| Response rate 10–20% | −0.08 |
| Not open to work | −0.04 |

### Honeypot Detection

Four impossibility checks catch the ~80 synthetic trap profiles:

1. `duration_months` exceeds months since `start_date` (chronologically impossible)
2. ≥3 skills marked `expert` with `duration_months = 0`
3. `years_of_experience` > 2.8× sum of `career_history` durations
4. ≥20 skills all at `expert` or `advanced` (keyword stuffer pattern)

---

## Two-Pass Pipeline

```
100,000 candidates
     ↓
  Pass 1: Fast filter (~1s per 10K)
  Check: has any Tier-1 skill OR high-signal title
     ↓
  Top 3,000 candidates
     ↓
  Pass 2: Full 5-component deep scoring
     ↓
  Top 100 → sorted by score (tie-break: ascending candidate_id)
     ↓
  bug_hunters.csv
```

This keeps runtime under 60 seconds on CPU, well within the 5-minute limit.

---

## Key Design Decisions

**Why tiered skills instead of flat keyword matching?**  
The JD says retrieval/search/ranking systems are the core requirement, and explicitly warns that LLM-tooling-only candidates are not a fit. Tier 1 skills get 6.5× the weight of Tier 3.

**Why a separate ProductFit component?**  
Two candidates with identical skills can have very different fit: one shipped a recommendation engine at Swiggy (product company), the other maintained integration tests at TCS (IT services). This needed its own signal.

**Why use the `industry` field for consulting detection?**  
Many consulting companies operate under subsidiary names not in any fixed list. The `industry` field ("IT Services", "Consulting", etc.) is a schema-level field — more reliable than name-matching.

**Why scan profile summary and headline?**  
The JD describes a "Tier 5" candidate who built a retrieval system at a product company but doesn't use buzzwords in their skill list. Scanning free-text catches these candidates that skill-only systems miss.

**Why are behavioral signals weighted at 20%?**  
From the JD: *"A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available."*

---

## File Structure

```
redrob_ranker/
├── rank.py                   # Main ranker — single command to reproduce
├── validate_submission.py    # Official format validator
├── requirements.txt          # Minimal deps (tqdm, pyyaml)
├── submission_metadata.yaml  # Team metadata and methodology
└── README.md                 # This file
```
