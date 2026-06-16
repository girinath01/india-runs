import json, os
from collections import Counter

base = r'c:\Users\girinath.k\OneDrive\Desktop\new project\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge'

path_sample = os.path.join(base, 'sample_candidates.json')
with open(path_sample, encoding='utf-8') as f:
    data = json.load(f)

titles = [c['profile']['current_title'] for c in data]
countries = [c['profile']['country'] for c in data]
years = [c['profile']['years_of_experience'] for c in data]

print('=== TITLES ===')
for t in titles:
    print(' ', t)

print()
print('=== COUNTRIES ===')
print(Counter(countries).most_common())

print()
print('=== YEARS RANGE ===')
print(f'min={min(years):.1f}, max={max(years):.1f}, avg={sum(years)/len(years):.1f}')

print()
print('=== SAMPLE SIGNALS ===')
for c in data[:5]:
    sig = c['redrob_signals']
    print(f"{c['candidate_id']}: open={sig['open_to_work_flag']}, last_active={sig['last_active_date']}, notice={sig['notice_period_days']}d, gh={sig['github_activity_score']}, rr={sig['recruiter_response_rate']:.0%}")

print()
print('=== CERTIFICATIONS in sample ===')
cert_counts = [len(c.get('certifications', [])) for c in data]
print(f'Avg certs: {sum(cert_counts)/len(cert_counts):.1f}, max: {max(cert_counts)}')

print()
path_jsonl = os.path.join(base, 'candidates.jsonl')
size_mb = os.path.getsize(path_jsonl) / (1024*1024)
print(f'candidates.jsonl size: {size_mb:.1f} MB')
print('Counting lines (candidates)...')
with open(path_jsonl, encoding='utf-8') as f:
    lines = sum(1 for line in f if line.strip())
print(f'Total candidates in JSONL: {lines:,}')

print()
print('=== FIRST CANDIDATE FROM JSONL ===')
with open(path_jsonl, encoding='utf-8') as f:
    first = json.loads(f.readline())
print('Keys:', list(first.keys()))
print('Career history jobs:', len(first['career_history']))
print('Skills count:', len(first['skills']))
print('Industry of current job:', first['profile'].get('current_industry'))
