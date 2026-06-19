import csv

with open('bug_hunters_v4.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

print(f'Total rows: {len(rows)}')
print()
print(f"{'Rank':>4}  {'Score':>7}  {'Candidate':14}  Reasoning preview")
print('-'*115)
for r in rows[:25]:
    print(f"{r['rank']:>4}  {float(r['score']):>7.4f}  {r['candidate_id']:14}  {r['reasoning'][:70]}...")

print()
print('--- Scores summary ---')
scores = [float(r['score']) for r in rows]
print(f'Min score : {min(scores):.4f}')
print(f'Max score : {max(scores):.4f}')
print(f'Avg score : {sum(scores)/len(scores):.4f}')
print(f'Non-increasing: {all(scores[i] >= scores[i+1] for i in range(len(scores)-1))}')
