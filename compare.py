import sys
import json
from rank import deep_score, iter_candidates

def find_candidate(input_path: str, cid: str) -> dict:
    for c in iter_candidates(input_path):
        if c.get("candidate_id") == cid:
            return c
    return None

def compare_candidates(input_path: str, cid_a: str, cid_b: str):
    cand_a = find_candidate(input_path, cid_a)
    cand_b = find_candidate(input_path, cid_b)
    
    if not cand_a or not cand_b:
        print("Candidate not found.")
        return

    score_a, reason_a, _ = deep_score(cand_a)
    score_b, reason_b, _ = deep_score(cand_b)
    
    print("================ Candidate Comparison ================")
    print(f"Candidate A: {cid_a} | Score: {score_a:.4f}")
    print(f"Reasoning: {reason_a}")
    print("------------------------------------------------------")
    print(f"Candidate B: {cid_b} | Score: {score_b:.4f}")
    print(f"Reasoning: {reason_b}")
    print("======================================================")
    
    if score_a > score_b:
        print(f"\nConclusion: {cid_a} ranks higher than {cid_b}.")
        diff = score_a - score_b
        print(f"Margin: +{diff:.4f}")
    elif score_b > score_a:
        print(f"\nConclusion: {cid_b} ranks higher than {cid_a}.")
        diff = score_b - score_a
        print(f"Margin: +{diff:.4f}")
    else:
        print("\nConclusion: Candidates are tied.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python compare.py <input.jsonl> <candidate_id_A> <candidate_id_B>")
        sys.exit(1)
    compare_candidates(sys.argv[1], sys.argv[2], sys.argv[3])
