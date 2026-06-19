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
FinalScore = 0.45×SearchRetrieval + 0.25×VectorSearch + 0.20×Production + 0.10×BehavioralBase
```

*(Note: A Sigmoid calibration function is applied to spread scores cleanly before tie-breaking).*

### Component Details

| Component | Weight | Key Logic |
|---|---|---|
| **Search/Retrieval** | 45% | Primary hits (BM25, LTR, Information Retrieval). Penalizes generic ML profiles lacking search experience. Rewards Search/Retrieval titles. |
| **Vector DBs** | 25% | Massive boosts for explicit Vector DBs (FAISS, Pinecone, Qdrant, Weaviate, Milvus, pgvector). |
| **Production Experience** | 20% | Heavy focus on deployed systems, large-scale inference. Penalizes pure academic/researchers. |
| **Behavioral Base** | 10% | Last active recency, recruiter response rate, github activity, connections, etc. |

### Tie-Breakers & Modifiers
- **Massive Hard Penalties**: Candidates who are NOT open to work (-0.40), have extreme notice periods (-0.35), are Job Hoppers (-0.35), or have purely Consulting careers (-0.40) suffer massive linear penalties *before* Sigmoid scaling, ensuring they are mathematically blocked from the top 100.
- **Location Alignment**: Candidates in preferred locations (Pune/Noida/Delhi NCR) retain 100% of their score. Unwilling candidates suffer deductions. Global talent willing to relocate is scored fairly without hard biases.
- **Honeypot Shield**: The system soft-penalizes honeypots in Pass 1 to allow them through for proper auditing, and cleanly zeroes them in Pass 2.

---

## Two-Pass Pipeline

```
100,000 candidates
     ↓
  Pass 1: Fast filter (~1s per 10K)
  Check: positive fast_score (drops obvious non-tech / irrelevant profiles)
     ↓
  Top 500 candidates
     ↓
  Pass 2: Full multi-signal deep scoring
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
├── submission_metadata.yaml  # Team metadata and methodology
├── README.md                 # This file
├── tools/                    # Validation, auditing, and compare scripts
└── tests/                    # Unit tests for scoring logic
```
