import re
from datetime import datetime, date
from .constants import TODAY, SKILL_RANKING, SKILL_SEARCH, SKILL_VECTOR, SKILL_SEMANTIC, SKILL_RAG
from .schemas import Candidate

def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))


def candidate_id_num(candidate_id: str) -> int:
    match = re.search(r"\d+$", candidate_id or "")
    return int(match.group(0)) if match else 0

def _categorise_skill(skill_name: str) -> str | None:
    s = re.sub(r"[-_/]", " ", skill_name.lower()).strip()
    if any(kw in s for kw in SKILL_RANKING):  return "RANKING"
    if any(kw in s for kw in SKILL_SEARCH):   return "SEARCH"
    if any(kw in s for kw in SKILL_VECTOR):   return "VECTOR"
    if any(kw in s for kw in SKILL_SEMANTIC): return "SEMANTIC"
    if any(kw in s for kw in SKILL_RAG):      return "RAG"
    return None

def parse_date(s: str) -> date | None:
    if not s:
        return None
    s = s.replace("/", "-")
    for fmt in ("%Y-%m-%d", "%Y-%m", "%d-%m-%Y", "%Y"):
        try:
            return datetime.strptime(s[:10].strip(), fmt).date()
        except ValueError:
            continue
    return None


def calculate_actual_yoe(career: list) -> float:
    """Calculate non-overlapping years of experience."""
    intervals = []
    for job in career:
        start_str = job.get("start_date", "")
        end_str   = job.get("end_date", "")
        start_dt  = parse_date(start_str)
        end_dt = None
        if end_str:
            end_dt = parse_date(end_str)
        dur = int(job.get("duration_months", 0) or 0)
        if start_dt and not end_dt and dur > 0:
            try:
                y, m = divmod(start_dt.month - 1 + dur, 12)
                end_dt = start_dt.replace(year=start_dt.year + y, month=m + 1)
            except ValueError:
                pass
        if not start_dt and dur > 0:
            intervals.append((-1, dur))
            continue
        if start_dt:
            if not end_dt:
                end_dt = TODAY
            start_m = start_dt.year * 12 + start_dt.month
            end_m   = end_dt.year * 12 + end_dt.month
            if end_m > start_m:
                intervals.append((start_m, end_m))

    valid_intervals = [i for i in intervals if i[0] != -1]
    valid_intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in valid_intervals:
        if not merged:
            merged.append(interval)
        else:
            last = merged[-1]
            if interval[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], interval[1]))
            else:
                merged.append(interval)
    total_months = sum(i[1] - i[0] for i in merged)
    for i in intervals:
        if i[0] == -1:
            total_months += i[1]
    return total_months / 12.0

def _split_sentences(text: str) -> list[str]:
    """Split text into rough sentences for evidence extraction."""
    parts = re.split(r'[.;!\n]+', text)
    return [p.strip() for p in parts if len(p.strip()) >= 15]


def _get_career_text(c: Candidate, max_jobs: int = 4) -> str:
    """Concatenate headline + summary + top career descriptions."""
    parts = [c.headline, c.summary]
    for job in c.career[:max_jobs]:
        parts.append((job.get("title", "") or "") + " " + (job.get("description", "") or ""))
    return " ".join(parts)
