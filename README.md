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
**Runtime: ~20-40 seconds** for 100,000 candidates on CPU (two-pass pipeline).  
**Memory: ~2–3 GB** (streams JSONL, never loads full dataset at once).

---

## Scoring Formula (v4.0)

```
FinalScore = 0.35×SearchRetrieval + 0.25×VectorSearch + 0.20×Production
           + 0.15×NoticePeriod + 0.05×OpenToWork
```

*(Note: A Sigmoid calibration function is applied to spread scores cleanly before tie-breaking).*

### Component Details

| Component | Weight | Key Logic |
|---|---|---|
| **Search/Retrieval** | 35% | Primary hits (BM25, LTR, Information Retrieval). Penalizes generic ML profiles lacking search experience. Rewards Search/Retrieval titles. |
| **Vector DBs** | 25% | Massive boosts for explicit Vector DBs (FAISS, Pinecone, Qdrant, Weaviate, Milvus, pgvector). |
| **Production Experience** | 20% | Heavy focus on deployed systems, large-scale inference. Penalizes pure academic/researchers. |
| **Notice Period** | 15% | <=15 days gets 100% of score. 16-30 days gets 90%. >90 days gets 0%. |
| **Open To Work** | 5% | Binary metric. Missing OTW removes 5% of final score. |

### Tie-Breakers & Modifiers
- **Experience Sweet Spot**: 5–9 years of experience applies a 1.15x multiplier. Junior profiles (<3.5 yrs) suffer a 0.80x multiplier.
- **Location Alignment**: Candidates in preferred locations (Pune/Noida) retain 100% of their score. Unwilling or out-of-country candidates suffer an 8.2% multiplier deduction.
- **Recruiter Response**: Candidates with <20% response rate receive a flat -0.05 deduction at the tie-breaker stage.
- **Honeypot Shield**: The system soft-penalizes honeypots in Pass 1 to allow them through for proper auditing, and cleanly zeroes them in Pass 2.

---

## Two-Pass Pipeline

```
100,000 candidates
     ↓
  Pass 1: Fast filter (~1s per 10K)
  Check: has any Tier-1 skill OR high-signal title
     ↓
  Top 5,000 candidates
     ↓
  Pass 2: Full 5-component deep scoring
     ↓
  Top 100 → sorted by score (tie-break: ascending candidate_id)
     ↓
  bug_hunters.csv
```

This keeps runtime under 60 seconds on CPU, well within the 5-minute limit.

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
