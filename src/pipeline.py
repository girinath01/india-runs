import csv
import gzip
import heapq
import json
import sys
import time
from pathlib import Path
from .schemas import Candidate
from .constants import TODAY
from .utils import candidate_id_num
from .fast_filter import fast_score, is_honeypot
from .analyzers import extract_features
from .scoring import compute_penalties, compute_score, generate_reasoning
import concurrent.futures

PASS2_POOL_SIZE = 8000

def normalize(raw: dict) -> Candidate:
    profile = raw.get("profile", {}) or {}
    return Candidate(
        id=raw.get("candidate_id", ""),
        headline=profile.get("headline", "") or "",
        summary=profile.get("summary", "") or "",
        current_title=profile.get("current_title", "") or "",
        current_company=profile.get("current_company", "") or "",
        location=profile.get("location", "") or "",
        country=profile.get("country", "") or "",
        years_of_experience=float(profile.get("years_of_experience", 0) or 0),
        career=raw.get("career_history", []) or [],
        skills=raw.get("skills", []) or [],
        signals=raw.get("redrob_signals", {}) or {},
    )

def deep_score(raw: dict, skip_semantic: bool = False) -> tuple[float, str, set[str]]:
    """
    Full v6.0 pipeline:
      is_honeypot(raw)           → early exit
      normalize(raw)             → Candidate
      extract_features(c)        → FeatureVector
      compute_penalties(c, fv)   → penalties (dedicated stage)
      compute_score(fv, pen)     → final score (two-stage + synergy)
      generate_reasoning(fv,c,s) → evidence-driven text
    """
    if is_honeypot(raw):
        return 0.0, (
            "Flagged as honeypot: internally inconsistent profile data "
            "(impossible tenure, inflated experience, or expert/zero-duration skills)."
        ), {"HONEYPOT"}

    c  = normalize(raw)
    fv = extract_features(c, skip_semantic=skip_semantic)
    penalties_multiplier = compute_penalties(c, fv)
    final = compute_score(fv, penalties_multiplier)

    reasoning, category = generate_reasoning(fv, c, final, penalties_multiplier)

    return final, reasoning, category


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE ITERATOR  [B2][B8] — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

def iter_candidates(input_path: str):
    path = Path(input_path)
    if not path.exists():
        print(f"ERROR: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8") as f:
        first_chunk = f.read(16)

    is_array = first_chunk.lstrip().startswith("[")

    with opener(path, "rt", encoding="utf-8") as f:
        if is_array:
            try:
                data = json.load(f)
                yield from (d for d in data if isinstance(d, dict))
            except json.JSONDecodeError as e:
                print(f"ERROR: JSON parse failed: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
                except json.JSONDecodeError:
                    if lineno <= 5:
                        print(f"[WARN] Bad JSON at line {lineno}: {line[:80]}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# TWO-PASS RANKING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def rank_candidates(input_path: str, output_path: str) -> None:
    t0 = time.time()
    print(f"[Ranker v6.0]  Input : {input_path}")
    print(f"[Ranker v6.0]  Output: {output_path}")
    print(f"[Ranker v6.0]  TODAY : {TODAY}\n")

    # ── PASS 1 ────────────────────────────────────────────────────────
    print(f"\n[Pass 1] Fast filtering (streaming)...")
    fast_pool: list[tuple[float, int, int, str, dict]] = []
    total = 0
    for candidate in iter_candidates(input_path):
        cid = candidate.get("candidate_id", "")
        if not cid:
            continue
        fs    = fast_score(candidate)
        total += 1
        entry = (fs, -candidate_id_num(cid), -total, cid, candidate)

        if len(fast_pool) < PASS2_POOL_SIZE:
            heapq.heappush(fast_pool, entry)
        elif entry > fast_pool[0]:
            heapq.heapreplace(fast_pool, entry)
        if total % 10_000 == 0:
            elapsed = time.time() - t0
            print(f"  {total:,} scanned  ({elapsed:.1f}s)", flush=True)

    print(f"[Pass 1] Done: {total:,} candidates scanned in {time.time()-t0:.1f}s")
    pool = [(s, cid, c) for s, _, _, cid, c in sorted(fast_pool, key=lambda x: (-x[0], x[3]))]
    print(f"[Pass 1] Top {len(pool)} candidates passed to Pass 2 (dropped {total - len(pool)})")

    # ── PASS 2 ────────────────────────────────────────────────────────
    # Run fast regex scoring on the 12,000 top candidates (skipping heavy semantic embeddings)
    print(f"\n[Pass 2] Deep regex scoring top {len(pool)} candidates...")
    pass2_results: list[tuple[str, float, dict]] = []

    def process_candidate(item):
        i, (_, cid, c) = item
        score, _, _ = deep_score(c, skip_semantic=True)
        return (i, cid, score, c)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for i, cid, score, c in executor.map(process_candidate, enumerate(pool)):
            pass2_results.append((cid, score, c))
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(pool)} scored  ({time.time()-t0:.1f}s)", flush=True)

    print(f"[Pass 2] Done in {time.time()-t0:.1f}s")
    
    # Take the Top 300 candidates from Pass 2 for Semantic Re-Scoring
    pass2_results.sort(key=lambda x: (-x[1], x[0]))
    top_300_pool = pass2_results[:300]
    print(f"[Pass 2] Top {len(top_300_pool)} candidates passed to Pass 3 (dropped {len(pass2_results) - len(top_300_pool)})")

    # ── PASS 3 ────────────────────────────────────────────────────────
    # Run heavy semantic embedding model on the top 300 candidates
    print(f"\n[Pass 3] Semantic re-scoring top {len(top_300_pool)} candidates...")
    deep_results: list[tuple[str, float, str, set[str]]] = []
    
    for i, (cid, pass2_score, c) in enumerate(top_300_pool):
        score, reasoning, category = deep_score(c, skip_semantic=False)
        deep_results.append((cid, score, reasoning, category))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(top_300_pool)} scored  ({time.time()-t0:.1f}s)", flush=True)

    print(f"[Pass 3] Done in {time.time()-t0:.1f}s")

    deep_results.sort(key=lambda x: (-x[1], x[0]))
    top_100 = deep_results[:100]

    hp_count = sum(1 for _, _, _, cat in top_100 if "HONEYPOT" in cat)
    hp_rate  = hp_count / max(len(top_100), 1)
    flag     = "[WARNING] EXCEEDS 10% THRESHOLD" if hp_rate > 0.10 else "[OK]"
    print(f"\n[Audit]  Honeypots in top 100: {hp_count} ({hp_rate:.0%})  {flag}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score, reasoning, cat) in enumerate(top_100, start=1):
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])

    elapsed = time.time() - t0
    print(f"\n[Done]  {out_path}  written in {elapsed:.1f}s")
    budget  = "[OK] WITHIN BUDGET" if elapsed < 300 else "[WARNING] OVER 5-MIN BUDGET"
    print(f"        {budget}")

    print(f"\n{'Rank':>4}  {'Candidate ID':16}  {'Score':>7}  Reasoning (preview 80 chars)")
    print("-" * 115)
    for rank, (cid, score, reasoning, cat) in enumerate(top_100[:10], start=1):
        print(f"{rank:>4}  {cid:16}  {score:>7.2f}  {reasoning[:80]}")
    print(f"\nScore range: {top_100[-1][1]:.2f} – {top_100[0][1]:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────
