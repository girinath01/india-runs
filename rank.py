#!/usr/bin/env python3
"""
Redrob Hackathon — Candidate Ranker (Bug Hunters)
JD: Senior AI Engineer — Founding Team @ Redrob AI

WIP — adding signal definitions first, scorers next.
"""

import json
import csv
import argparse
import sys
from datetime import datetime, date
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# JD SIGNAL DEFINITIONS
# Read the JD very carefully before touching these.
# The key trap: keyword match is NOT enough. A "Marketing Manager"
# with perfect AI keywords is NOT a fit. A Recommendation Systems Engineer
# without "RAG"/"Pinecone" in their skills IS a fit.
# ─────────────────────────────────────────────────────────────────────────────

# Core required skills from the JD (embeddings, vector DBs, evaluation, NLP)
REQUIRED_SKILLS = {
    # Embeddings & Retrieval — JD explicitly needs PRODUCTION experience here
    "sentence-transformers", "sentence transformer", "embeddings", "embedding",
    "openai embeddings", "text-embedding", "bge", "e5 model", "ada",
    "cohere embed", "dense retrieval", "sparse retrieval",
    "hybrid retrieval", "hybrid search",
    "bi-encoder", "cross-encoder", "semantic search", "semantic similarity",
    "retrieval", "information retrieval", "dense passage retrieval", "dpr",
    # Vector Databases — JD explicitly needs PRODUCTION experience here
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "chroma", "pgvector", "vespa", "annoy", "scann",
    "vector database", "vector db", "vector store", "vector index",
    "vector search", "ann", "approximate nearest neighbor", "hnsw",
    # RAG & Search
    "rag", "retrieval augmented generation", "bm25", "tfidf", "tf-idf",
    "reranking", "re-ranking", "reranker", "colbert",
    # Ranking & Recommendation (recommendation system counts per JD)
    "learning to rank", "ltr", "lambdamart", "lambdarank", "ranknet",
    "recommendation system", "recommender", "ranking system", "search ranking",
    "ndcg", "mrr", "mean reciprocal rank", "map", "mean average precision",
    "a/b testing", "ab testing", "offline evaluation", "online evaluation",
    "eval framework", "evaluation framework",
    # NLP & LLMs
    "nlp", "natural language processing", "language model", "text classification",
    "bert", "transformers", "huggingface", "hugging face",
    "llm", "large language model", "generative ai", "gpt",
    "fine-tuning", "fine tuning", "finetuning", "lora", "qlora", "peft",
    "prompt engineering", "instruction tuning",
    # ML Engineering
    "machine learning", "deep learning", "neural network",
    "pytorch", "tensorflow", "keras", "jax",
    "xgboost", "lightgbm", "gradient boosting",
    "scikit-learn", "sklearn",
    "mlops", "model deployment", "model serving", "inference",
    "feature engineering", "feature store",
    # Engineering fundamentals for product role
    "python", "fastapi", "flask", "api", "microservices", "docker", "kubernetes",
    "spark", "kafka", "airflow",
    "aws", "gcp", "azure", "cloud",
    "sql", "postgresql", "mongodb", "redis",
    "distributed systems", "distributed training",
    # Specific tools mentioned in JD
    "weights & biases", "wandb", "mlflow", "ray", "dask",
    "triton", "tensorrt", "onnx",
}

# Nice-to-have (lower weight contribution)
NICE_TO_HAVE_SKILLS = {
    "rlhf", "reinforcement learning from human feedback",
    "data parallelism", "model parallelism", "quantization", "distillation",
    "hr tech", "talent intelligence", "marketplace", "two-sided marketplace",
    "open source", "github contributions",
}

# CV / Speech / Robotics — disqualifying if DOMINANT in skill profile
# JD explicitly calls these out as "wrong domain"
DISQUALIFYING_SKILL_FOCUS = {
    "computer vision", "image classification", "object detection",
    "yolo", "convolutional", "image segmentation", "instance segmentation",
    "opencv", "open cv", "object recognition",
    "speech recognition", "asr", "tts", "text-to-speech", "speech synthesis",
    "audio processing", "sound classification", "voice recognition",
    "robotics", "ros", "autonomous driving", "lidar", "slam", "point cloud",
    # Non-ML roles
    "photoshop", "illustrator", "figma", "adobe", "canva",
    "six sigma", "lean", "kaizen",
    "solidworks", "autocad", "ansys", "creo",
    "accounting", "tally", "gst", "taxation",
    "seo", "content writing", "copywriting",
    "salesforce", "crm",
}

# Consulting companies by name (fallback if industry field unavailable)
CONSULTING_COMPANY_NAMES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "ltimindtree", "lti", "larsen toubro infotech", "niit technologies",
    "persistent systems", "cyient", "zensar", "sonata software",
    "kpit", "mastech", "igate", "firstsource", "wns", "genpact",
    "birlasoft", "coforge", "happiest minds",
}

# Consulting by INDUSTRY FIELD — more reliable than company name matching
CONSULTING_INDUSTRIES = {
    "it services", "it consulting", "consulting",
    "business process outsourcing", "bpo", "outsourcing",
    "staffing", "staffing and recruiting", "managed services", "it staffing",
}

# Target technical titles for this JD
TARGET_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "nlp engineer", "search engineer", "information retrieval engineer",
    "data scientist", "applied scientist", "research engineer",
    "senior engineer", "principal engineer", "staff engineer",
    "deep learning engineer", "recommendation engineer",
    "ranking engineer", "relevance engineer", "search scientist",
    "software engineer", "backend engineer", "platform engineer",
    "applied ml", "applied ai", "ai researcher", "ml researcher",
    "data engineer", "mlops engineer",
}

# Clearly non-technical — strong negative signal
NON_TECH_TITLES = {
    "marketing manager", "hr manager", "human resources", "operations manager",
    "business analyst", "content writer", "sales executive", "accountant",
    "project manager", "graphic designer", "customer support", "customer service",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "finance manager", "account manager", "talent acquisition", "recruiter",
    "junior analyst", "data entry", "qa engineer", "tester",
    "business development",
}

# JD preferred locations (Pune/Noida first, then other Tier-1 India cities)
PREFERRED_INDIA_LOCATIONS = {
    "pune",
    "noida", "delhi", "delhi ncr", "ncr", "new delhi",
    "gurgaon", "gurugram", "faridabad", "greater noida",
    "hyderabad", "mumbai", "bangalore", "bengaluru",
}

# Proficiency → numeric weight mapping
PROFICIENCY_WEIGHTS = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.45,
    "beginner": 0.15,
}

TODAY = date(2026, 6, 15)  # Evaluation reference date


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def normalize(value, min_v, max_v):
    if max_v <= min_v:
        return 0.0
    return clamp((value - min_v) / (max_v - min_v))


# Scorers will go here next
if __name__ == "__main__":
    print("Signal definitions loaded. Scorers coming next.")
