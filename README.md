# Redrob Hackathon — Bug Hunters

Ranking system for the **Intelligent Candidate Discovery & Ranking Challenge**
by Redrob AI. Produces a top-100 submission CSV for the Senior AI Engineer JD.

## Reproduce Command

```bash
python rank.py --candidates ./candidates.jsonl --out ./bug_hunters.csv
```

Accepts both `.jsonl` (uncompressed) and `.jsonl.gz` (gzipped) input.  
Expected runtime: **~90 seconds** for 100,000 candidates on CPU.  
Memory: **~2–3 GB** (streams JSONL line-by-line, never loads full file).

## Setup

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./bug_hunters.csv
python validate_submission.py bug_hunters.csv
```

---

## Scoring Architecture

A **7-component weighted hybrid scorer** with disqualifier multipliers.  
Pure Python — no GPU, no network, no ML frameworks at inference time.

### Component Weights

| Component | Weight | Description |
|---|---|---|
| AI/ML Skills Match | 26% | Proficiency × endorsement × duration weighted overlap against ~120 JD-specific keywords (embeddings, vector DBs, NLP, evaluation). **Also scans profile summary + headline** to catch plain-language candidates who describe experience in prose rather than keyword lists. |
| Role & Title Fit | 22% | Current title + career history alignment with AI Engineer target roles. Consulting detection uses the `industry` field (more reliable than company name matching) — consulting careers get a 0.12–0.72× penalty. |
| Production Experience | 20% | NLP on `career_history[].description`: production signals (deployed, shipped, at scale, real users, latency) vs research-only signals (paper, thesis, academic lab), weighted by job duration. JD explicitly disqualifies pure researchers. |
| Years of Experience | 12% | Curve peaking at 6–7yr (JD's stated ideal: "6–8 years at product companies"). |
| Behavioral Signals | 10% | 12 platform sub-signals: open-to-work flag, activity recency, recruiter response rate, interview completion rate, GitHub activity, notice period, profile completeness, saved by recruiters, skill assessment scores, response time, active applications count, verified contacts. |
| Location Fit | 6% | India → Pune/Noida = 1.0; other Tier-1 cities = 0.85; outside India = 0.20–0.45. |
| Education | 4% | Institution tier (tier_1–4) + STEM field of study bonus. |

### Disqualifier Multipliers

Applied as a single penalty factor after the weighted sum:

| Condition | Multiplier |
|---|---|
| Honeypot (impossible profile data) | **0.00×** |
| Entire career in IT services/consulting | **0.12×** |
| 80–95% career in consulting | **0.28×** |
| 65–80% career in consulting | **0.50×** |
| 50–65% career in consulting | **0.72×** |
| Non-technical title, no technical history | **0.18×** |
| CV/Speech/Robotics primary focus (>65%) | **0.22×** |
| CV/Speech/Robotics moderate focus (45–65%) | **0.52×** |
| Pure researcher (no production signal) | **0.38×** |

### Honeypot Detection

Four impossibility checks identify the ~80 honeypot profiles:
1. Stated job `duration_months` exceeds months since `start_date` (chronologically impossible)
2. ≥3 skills marked `expert` proficiency with `duration_months = 0`
3. `years_of_experience` > 2.8× sum of `career_history` durations
4. ≥20 skills all marked `expert` or `advanced` (keyword stuffer)

---

## Key Design Decisions

### Why not use sentence-transformers / LLMs for ranking?
The spec enforces ≤5 minutes, CPU-only, no network. Even a small transformer
over 100K candidates takes 30+ minutes on CPU. This ranker runs in ~90 seconds
using pure keyword + rule-based scoring.

### Why use the `industry` field over company name matching for consulting?
Many consulting companies operate under subsidiary names not in any fixed list.
The `industry` field ("IT Services", "IT Consulting", etc.) is a direct schema
field — more reliable and complete.

### Why scan profile summary and headline?
The JD explicitly says: *"A Tier 5 candidate may not use the words 'RAG' or
'Pinecone' in their profile, but if their career history shows they built a
recommendation system at a product company, they're a fit."*
Scanning free-text summary catches these plain-language Tier 5 candidates that
keyword-only systems miss.

### Why are behavioral signals weighted at 10%?
From the JD: *"A perfect-on-paper candidate who hasn't logged in for 6 months
and has a 5% recruiter response rate is, for hiring purposes, not actually
available."* Availability matters more than a pure skill-match score.

---

## File Structure

```
redrob_ranker/
├── rank.py                   # Main ranker — single command to produce submission
├── validate_submission.py    # Official format validator (from hackathon bundle)
├── requirements.txt          # Minimal dependencies (tqdm, pyyaml)
├── submission_metadata.yaml  # Team metadata
└── README.md                 # This file
```
