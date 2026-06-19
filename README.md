# Redrob Hackathon — Bug Hunters (v4.1)

Highly optimized candidate ranking system for the **Intelligent Candidate Discovery & Ranking Challenge** by Redrob AI. Designed to isolate elite Senior Relevance/Search Engineers and produce a top-100 submission CSV tailored for maximum NDCG@10, Recall, and Precision.

## Reproduce Command

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./bug_hunters.csv
```

Accepts `.jsonl` (uncompressed), `.jsonl.gz` (gzipped), and `.json` arrays.  
**Runtime: ~20 seconds** for 100,000 candidates on CPU.  
**Memory: ~50 MB** (streams JSONL using lazy evaluation; never loads full dataset).

---

## Architecture & Logic Flow

The system processes candidates using an ultra-fast, two-pass architecture.

### Pass 1: Fast Filter (100K → Top 5,000)
A highly optimized, regex-driven heuristic pass that drops 95% of candidates.
- Evaluates top-level titles against `STRONG_TITLE_SCORES`.
- Performs string-normalized matching across `TIER1_SKILLS` and `TIER2_SKILLS`.
- Checks combined headline, summary, and recent job descriptions for core search/vector terminology.
- **Elite Company Anchor:** Automatically grants a massive fast-score bypass to candidates with experience at FAANG or elite search companies (e.g., Google, Meta, Pinterest) who might use natural language instead of keyword-stuffed buzzwords.

### Pass 2: Deep Scoring (5,000 → Top 100)
A heavy temporal and semantic extraction pass. Applies the final scoring formula, handles edge cases, and calculates deterministic tie-breakers.

---

## Scoring Formula (v4.1)

```text
RawScore = 0.30×SearchRetrieval + 0.20×VectorSearch + 0.15×Production 
         + 0.15×Behavioral + 0.15×NoticePeriod + 0.05×OpenToWork

FinalScore = max(0.0, (RawScore × ExperienceFitMultiplier) + TieBreaker - HardPenalties)
```

*(Note: Sigmoid calibration was explicitly removed in v4.1 to prevent floating-point collisions at the extreme top end, strictly preserving mathematically optimal candidate ordering for NDCG@10 evaluation).*

### Component Logic

| Component | Logic |
|---|---|
| **Search/Retrieval (30%)** | Temporal extraction of IR skills (BM25, LTR, NDCG). Penalizes generic ML profiles lacking search experience. Co-occurrence bonuses for Hybrid Search (IR + Vectors in the same job). |
| **Vector DBs (20%)** | High-weight extraction for explicit Vector DBs (FAISS, Pinecone, Qdrant, Weaviate, Milvus). |
| **Production Experience (15%)** | Focuses on absolute volume of deployed systems (`ship_hits`). Intersects shipping terminology with technical keywords to reject false positives (e.g., "shipped marketing emails"). |
| **Behavioral Base (15%)** | Recruiter response rates, interview completion, offer acceptance, and github activity. |
| **Availability (20%)** | Notice period evaluation and explicit Open-To-Work signals. |

### Advanced Modifiers & Defenses

1. **Date-Based Exponential Decay:** Older jobs naturally decay in weight. We calculate exact temporal chronology using parsed `end_date` vs `TODAY` to accurately discount stale tech stacks.
2. **Honeypot Shield:** Traps 5 variations of synthetic candidates (e.g., impossible time-travel tenures, conflicting domain overlaps). Includes a **Seniority Whitelist** (>10 YOE) to prevent accidentally disqualifying legitimate senior engineers whose resumes were parsed without skill duration timestamps.
3. **Hard Penalties:** Flat deductions for Title Chasers (<18 mo tenure), pure academics without production code, LLM hype-riders (LangChain without IR fundamentals), and candidates definitively not open to work.

---

## File Structure

```text
redrob_ranker/
├── rank.py                   # Main pipeline — logic, scraping, scoring, and output
├── requirements.txt          # Minimal dependencies (tqdm)
├── submission_metadata.yaml  # Team methodologies and hyperparameters
└── README.md                 # System documentation
```
