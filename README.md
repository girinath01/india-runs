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
   only the top 5,000 profiles in memory.
2. Pass 2 performs deeper scoring on that shortlist and writes the top 100.

The final score emphasizes the reference-file hierarchy:

- Search, retrieval, ranking, and evaluation experience
- Vector database and ANN experience
- Production shipping evidence over pure research
- Behavioral availability and responsiveness
- Notice period, open-to-work, location, and relocation signals

Hard penalties cover JD red flags such as non-technical roles, pure academic
profiles without production evidence, LangChain/OpenAI-only profiles without
retrieval fundamentals, consulting-heavy backgrounds, long notice periods, and
title-chasing job histories.

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
