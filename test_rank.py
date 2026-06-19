import unittest
import datetime
from rank import (
    score_behavioral, 
    score_location, 
    score_product_fit, 
    score_skill_fit, 
    compute_hard_penalties,
    _categorise_skill
)

class TestRankingSystem(unittest.TestCase):
    
    def test_behavioral_scoring(self):
        # Excellent behavioral signals
        signals_good = {
            "open_to_work_flag": True,
            "notice_period_days": 15,
            "recruiter_response_rate": 1.0,
            "last_active_date": "2026-06-18",
            "interview_completion_rate": 1.0,
            "saved_by_recruiters_30d": 5,
            "offer_acceptance_rate": 1.0,
            "search_appearance_30d": 10,
            "connection_count": 500,
            "github_activity_score": 80
        }
        score = score_behavioral(signals_good)
        self.assertGreater(score, 0.8)
        
        # Poor behavioral signals
        signals_bad = {
            "open_to_work_flag": False,
            "notice_period_days": 90,
            "recruiter_response_rate": 0.1
        }
        score2 = score_behavioral(signals_bad)
        self.assertLess(score2, 0.4)

    def test_hard_penalties(self):
        profile = {"current_title": "Data Scientist", "years_of_experience": 5}
        career = [{"title": "Data Scientist", "duration_months": 24}]
        skills = [{"name": "Python"}, {"name": "SQL"}]
        signals = {"open_to_work_flag": True, "notice_period_days": 30}
        
        # Baseline
        penalty = compute_hard_penalties(profile, career, skills, signals, 0.0)
        self.assertEqual(penalty, 0.0)
        
        # Not open to work
        signals["open_to_work_flag"] = False
        penalty = compute_hard_penalties(profile, career, skills, signals, 0.0)
        self.assertGreaterEqual(penalty, 0.40)
        
        # Job hopper
        career_hop = [
            {"title": "Dev", "duration_months": 6},
            {"title": "Dev", "duration_months": 5},
            {"title": "Dev", "duration_months": 4}
        ]
        penalty = compute_hard_penalties(profile, career_hop, skills, signals, 0.0)
        self.assertGreaterEqual(penalty, 0.75) # 0.40 (not OTW) + 0.35 (hopper)

    def test_skill_categorization(self):
        self.assertEqual(_categorise_skill("faiss"), "VECTOR")
        self.assertEqual(_categorise_skill("elasticsearch"), "SEARCH")
        self.assertEqual(_categorise_skill("learning to rank"), "RANKING")

if __name__ == "__main__":
    unittest.main()
