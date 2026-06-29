# Redrob Hackathon - Bug Hunters Ranker

Offline CPU-only candidate ranker for the Redrob AI Senior AI Engineer
(Founding Team) challenge. The pipeline ranks the top 100 candidates from a
large JSONL/JSONL.GZ candidate pool and writes a submission CSV with:

```text
candidate_id,rank,score,reasoning
```

## Reproduce

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./bug_hunters.csv
python validate_submission.py ./bug_hunters.csv
```

`rank.py` uses only the Python standard library. It performs no network calls,
uses no hosted LLM APIs, and does not require a GPU.

Supported input formats:

- `.jsonl`
- `.jsonl.gz`
- `.json` arrays for small local samples

Use the registered team/participant id for the final filename if it differs
from `bug_hunters.csv`.

## Method

The ranker uses a two-pass architecture designed for the 5-minute CPU limit:

1. Pass 1 streams every candidate, computes a fast heuristic score, and keeps
   only the top 12,000 profiles in memory.
2. Pass 2 performs deeper scoring on that shortlist and writes the top 100.

The final score emphasizes the reference-file hierarchy:

- Search, retrieval, ranking, and recommendation experience (treated equally)
- Domain Tenure (Years spent explicitly in search/recommendation domains)
- Production shipping evidence over pure research
- Search Quality Metrics (NDCG, MAP, offline/online A/B testing)
- "Founding Mindset" (0->1 builder experience)
- Notice period, open-to-work, location, and relocation signals

Hard rejections (-100 penalties) strictly filter out JD red flags:
- Pure academic/research profiles without evidence of deployed systems
- Computer Vision or Speech specialists lacking core retrieval intent
- Candidates with generic titles who have zero production shipping evidence
- Non-technical roles (Sales, HR, PMs, Scrum Masters)
- Profiles with >90% of their career in consulting firms
- LangChain/OpenAI-only profiles without retrieval fundamentals
- Fabricated experience and extreme title-chasing job hoppers
## Files

```text
rank.py                   Main offline ranking pipeline
validate_submission.py    CSV format validator for the official submission rules
requirements.txt          No third-party runtime dependencies
submission_metadata.yaml  Team and reproduction metadata
```

## Notes

- Output is UTF-8 CSV.
- The generated CSV always uses ranks 1-100 when at least 100 candidates are
  available.
- Scores are sorted monotonically non-increasing by rank.
- Reasoning strings are generated from candidate profile facts and include
  rank-consistent strengths and concerns.
