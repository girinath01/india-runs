import re
from datetime import date

TODAY = date(2026, 6, 15)

TIER1_SKILLS = frozenset({
    "retrieval", "information retrieval", "dense retrieval", "sparse retrieval",
    "hybrid retrieval", "hybrid search", "semantic search", "vector search",
    "search engine", "search ranking", "search system", "search infrastructure",
    "recommendation system", "recommender", "recommender system", "ranking system",
    "learning to rank", "ltr", "lambdamart", "lambdarank", "ranknet",
    "bi encoder", "cross encoder",
    "faiss", "elasticsearch", "opensearch", "qdrant", "milvus",
    "weaviate", "pinecone", "pgvector", "vespa", "annoy", "scann", "chroma",
    "vector database", "vector db", "vector store", "vector index",
    "ann", "approximate nearest neighbor", "hnsw",
    "reranking", "re ranking", "reranker", "colbert",
    "bm25", "tfidf", "tf idf",
    "ndcg", "mrr", "mean reciprocal rank", "map", "mean average precision",
    "a b testing", "ab testing", "offline evaluation", "online evaluation",
    "eval framework", "evaluation framework",
})

TIER2_SKILLS = frozenset({
    "rag", "retrieval augmented generation",
    "embeddings", "embedding", "sentence transformers", "sentence transformer",
    "text embedding", "openai embeddings", "bge", "e5 model", "dense embedding",
    "nlp", "natural language processing", "language model", "text classification",
    "bert", "transformers", "huggingface", "hugging face",
    "pytorch", "tensorflow", "keras", "jax",
    "mlops", "model deployment", "model serving", "inference",
    "feature engineering", "feature store",
    "distributed systems", "distributed training",
    "mlflow", "wandb", "ray", "dask",
    "xgboost", "lightgbm", "gradient boosting",
    "scikit learn", "sklearn",
    "python", "fastapi", "flask", "docker", "kubernetes",
    "spark", "kafka", "airflow",
    "aws", "gcp", "azure",
})

TIER3_SKILLS = frozenset({
    "langchain", "lang chain",
    "prompt engineering", "instruction tuning",
    "qlora", "lora", "peft", "fine tuning", "finetuning",
    "llm", "large language model", "generative ai",
    "openai", "chatgpt", "gpt",
    "rlhf",
})

DISQUALIFYING_SKILLS = frozenset({
    "computer vision", "image classification", "object detection",
    "yolo", "convolutional", "image segmentation",
    "opencv",
    "speech recognition", "asr", "tts", "text to speech", "speech synthesis",
    "audio processing", "sound classification", "voice recognition",
    "robotics", "ros", "autonomous driving", "lidar", "slam",
    "photoshop", "illustrator", "figma", "canva",
    "accounting", "tally", "gst", "salesforce", "crm",
    "seo", "content writing", "copywriting",
    "six sigma", "lean", "kaizen",
    "solidworks", "autocad",
})

TIER1_REGEX = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(TIER1_SKILLS, key=len, reverse=True))) + r')\b')
TIER2_REGEX = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(TIER2_SKILLS, key=len, reverse=True))) + r')\b')
TIER3_REGEX = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(TIER3_SKILLS, key=len, reverse=True))) + r')\b')
DISQ_REGEX  = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(DISQUALIFYING_SKILLS, key=len, reverse=True))) + r')\b')

# Module-level compiled regexes — UNCHANGED
PRIMARY_CORE_RE    = re.compile(r'\b(information retrieval|learning to rank|ltr|lambdamart|bm25|semantic search|candidate matching|search quality)\b')
SECONDARY_CORE_RE  = re.compile(r'\b(ranking system|recommendation system|candidate ranking|personalization|relevance|matching engine|elasticsearch|opensearch)\b')
EXPLICIT_VECTOR_RE = re.compile(r'\b(faiss|pinecone|qdrant|weaviate|milvus|pgvector)\b')
VECTOR_TEXT_RE     = re.compile(r'\b(vector search|vector database|chroma|ann|hnsw)\b')
# HR_TEXT_RE superseded by MARKETPLACE_RE in v6.0 — kept for schema reference only
# HR_TEXT_RE       = re.compile(r'\b(hr tech|recruiting tech|talent acquisition platform|marketplace product|job board)\b')
PROD_TEXT_RE       = re.compile(r'\b(scale|shipped|deployed|productionized|enterprise|latency|qps|inference optimization|tensorrt|vllm|distributed systems|ray|spark)\b')
# INFRA_TEXT_RE superseded by SEARCH_RE in v6.0 — kept for schema reference only
# INFRA_TEXT_RE      = re.compile(r'\b(search infra|retrieval pipeline|relevance|indexing|ranking optimization)\b')
EVAL_TEXT_RE       = re.compile(r'\b(ndcg|mrr|mean average precision|a/b test|ab testing|offline evaluation|online evaluation)\b')
RESEARCH_HEAVY_RE  = re.compile(r'\b(paper|publication|thesis|arxiv|academic)\b')

# ─────────────────────────────────────────────────────────────────────────────
# V6.0 — SEMANTIC BUCKET REGEXES (Step 4: JDIntentAnalyzer)
# ─────────────────────────────────────────────────────────────────────────────

RECOMMENDATION_RE = re.compile(
    r'\b(recommendation|personalization|feed ranking|product ranking|'
    r'collaborative filtering|content ranking|recommender|ranking model|'
    r'recommendation engine|recommendation system|recommender system)\b'
)
SEARCH_RE = re.compile(
    r'\b(semantic search|retrieval|hybrid search|information retrieval|'
    r'dense retrieval|query understanding|search engine|search ranking|'
    r'search quality|search system|search infrastructure|search platform|'
    r'retrieval system|retrieval pipeline)\b'
)
MARKETPLACE_RE = re.compile(
    r'\b(marketplace|two.?sided|supply.?demand|talent marketplace|'
    r'job matching|candidate matching|hiring platform|hr tech|'
    r'recruiting tech|talent acquisition|marketplace product|job board)\b'
)

# ─────────────────────────────────────────────────────────────────────────────
# V6.0 — OWNERSHIP VERB CATEGORIES (Step 7: OWNER/LEAD/CONTRIBUTOR/SUPPORT)
# ─────────────────────────────────────────────────────────────────────────────

OWNER_VERBS_RE = re.compile(
    r'\b(architected|designed|invented|pioneered|founded|created from scratch)\b'
)
LEAD_VERBS_RE = re.compile(
    r'\b(led|owned|drove|spearheaded|established|launched|initiated|'
    r'built and deployed|built and launched)\b'
)
CONTRIBUTOR_VERBS_RE = re.compile(
    r'\b(built|implemented|developed|created|wrote|shipped|engineered|'
    r'designed and implemented|coded|constructed|delivered)\b'
)
SUPPORT_VERBS_RE = re.compile(
    r'\b(worked on|assisted|supported|helped|participated|contributed to|'
    r'was part of|involved in|collaborated on)\b'
)

# v5 ownership regexes preserved for backward compat (used in evidence analyzer)
OWNERSHIP_HIGH_RE = re.compile(
    r'\b(designed|architected|led|owned|created|implemented|launched|drove|'
    r'founded|established|spearheaded|initiated|invented|pioneered|'
    r'built and deployed|built and launched)\b'
)
# OWNERSHIP_LOW_RE superseded by SUPPORT_VERBS_RE in v6.0 — kept for backward compat
# OWNERSHIP_LOW_RE = re.compile(
#     r'\b(worked on|assisted|supported|helped|participated|contributed to|'
#     r'was part of|involved in|collaborated on)\b'
# )

# ImpactAnalyzer — measurable outcomes (Step 9)
IMPACT_METRICS_RE = re.compile(
    r'\b(reduced latency|improved ctr|click.?through|conversion rate|revenue|'
    r'throughput|scaled to \d+[mk]?|million users|billion|10[xX]|'
    r'latency reduction|cost reduction|precision|recall|f1 score|'
    r'a/?b test|experiment|rollout|lifted|increased by|decreased by|'
    r'saved \$|reduced \$|qps improvement|p99|p50|nps improvement)\b'
)

# Impact value extraction — captures numbers/percentages near metric words
IMPACT_VALUE_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*(%|x|[mk]\b|million|billion|users|queries|rps|qps|ms|seconds)',
    re.IGNORECASE
)

# ─────────────────────────────────────────────────────────────────────────────
# COMPANY / TITLE / LOCATION CONSTANTS — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

CONSULTING_INDUSTRIES = frozenset({
    "it services", "it consulting", "consulting",
    "business process outsourcing", "bpo", "outsourcing",
    "staffing", "managed services", "it staffing",
})
CONSULTING_NAMES = frozenset({
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "l&t infotech", "lti", "cognizant technology solutions", "atos",
    "niit technologies", "persistent systems", "cyient", "zensar",
    "birlasoft", "coforge", "kpit", "mastech", "firstsource", "wns", "genpact",
})
CONSULTING_RE = re.compile(r'\b(?:' + '|'.join(map(re.escape, CONSULTING_INDUSTRIES | CONSULTING_NAMES)) + r')\b')

PRODUCT_TECH_COMPANIES = frozenset({
    "swiggy", "zomato", "razorpay", "meesho", "flipkart", "ola", "gojek",
    "grab", "zepto", "uber", "doordash", "instacart",
})
PRODUCT_TECH_COMPANIES_RE = re.compile(r'\b(?:' + '|'.join(map(re.escape, PRODUCT_TECH_COMPANIES)) + r')\b')

ELITE_SEARCH_COMPANIES = frozenset({
    "google", "meta", "facebook", "linkedin", "netflix", "pinterest", "airbnb", "amazon",
})
ELITE_SEARCH_COMPANIES_RE = re.compile(r'\b(?:' + '|'.join(map(re.escape, ELITE_SEARCH_COMPANIES)) + r')\b')

SEARCH_COMPANIES = frozenset({
    "linkedin", "google", "meta", "amazon", "airbnb", "pinterest", "spotify", "netflix",
})
SEARCH_COMPANIES_RE = re.compile(r'\b(?:' + '|'.join(map(re.escape, SEARCH_COMPANIES)) + r')\b')

STRONG_TITLE_SCORES: dict[str, float] = {
    "search engineer": 1.0, "ranking engineer": 1.0,
    "relevance engineer": 1.0, "search scientist": 1.0,
    "recommendation engineer": 1.0, "recommendation systems engineer": 1.0,
    "applied ml engineer": 0.95, "applied ml": 0.95,
    "nlp engineer": 0.90, "ai engineer": 0.88,
    "ml engineer": 0.88, "machine learning engineer": 0.88,
    "applied scientist": 0.82, "research engineer": 0.75,
    "principal engineer": 0.82, "staff engineer": 0.82,
    "senior engineer": 0.72, "data engineer": 0.45,
}

NON_TECH_TITLES = frozenset({
    "marketing manager", "hr manager", "human resources", "operations manager",
    "business analyst", "content writer", "sales executive", "accountant",
    "project manager", "graphic designer", "customer support", "customer service",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "finance manager", "account manager", "talent acquisition", "recruiter",
    "data entry", "tester", "manual tester", "business development",
    "scrum master", "product manager", "sales", "marketing",
})

SHIP_SIGNALS = frozenset({
    "production", "shipped", "launched", "deployed", "live system",
    "real users", "at scale", "million", "billion",
    "latency", "throughput", "serving", "inference", "qps", "rps",
    "a/b test", "experiment", "rollout",
    "ranking system", "retrieval system", "recommendation engine",
    "search engine", "embedding service", "vector index", "reranker",
    "search platform", "recommendation platform",
    "sla", "uptime", "oncall", "latency reduction", "cost reduction",
    "serving system", "online inference", "production traffic",
})
SHIP_SIGNALS_RE = re.compile(r'\b(?:' + '|'.join(map(re.escape, SHIP_SIGNALS)) + r')\b')

RESEARCH_SIGNALS = frozenset({
    "paper", "publication", "conference", "arxiv", "journal",
    "thesis", "academic", "phd candidate", "research lab",
    "pure research", "benchmark", "theoretical",
})
RESEARCH_SIGNALS_RE = re.compile(r'\b(?:' + '|'.join(map(re.escape, RESEARCH_SIGNALS)) + r')\b')

# Production-scale signals — differentiate 100 users from 10M users
SCALE_RE = re.compile(
    r'\\b(?:'
    r'\\d+(?:\\.\\d+)?\\s*(?:million|billion|m\\+?|b\\+?)\\s*(?:users?|requests?|queries?|candidates?|records?|dau|mau)\\b'
    r'|(?:millions?|billions?)\\s+of\\s+(?:users?|requests?|queries?|records?|profiles?)'
    r'|10[kmb]\\+?\\s*(?:users?|requests?|queries?)'
    r'|at\\s+(?:massive|web|production)\\s+scale'
    r'|high[- ](?:traffic|throughput|volume|qps)'
    r'|(?:1|2|5|10|50|100|500)[kmb]\\s+(?:rps|qps|tps)'
    r')',
    re.IGNORECASE
)

FOUNDING_MINDSET_RE = re.compile(
    r'\b(0\s*to\s*1|0->1|greenfield|from scratch|built v1|first engineer|founding|early stage|startup)\b',
    re.IGNORECASE
)

PASS2_POOL_SIZE = 12000
FAST_WORDS_RE = re.compile(
    r'\b(retrieval|recommendation|ranking|search|vector|embedding|nlp|rag|ndcg|'
    r'faiss|qdrant|pinecone|milvus|elasticsearch|weaviate|retrieval platform|'
    r'relevance|matching engine|candidate matching|recommendation pipeline|'
    r'search quality|personalization|ranking model)\b'
)
SHIP_FAST_RE = re.compile(r'\b(shipped|deployed|production|at scale|million|serving|built and)\b')

# ─────────────────────────────────────────────────────────────────────────────
# SPECIALIZATION PATHS (Step 8: CareerAnalyzer)
# ─────────────────────────────────────────────────────────────────────────────

_SPEC_RETRIEVAL_KW = ("retrieval", "search", "ranking", "recommendation", "matching",
                      "relevance", "recommender", "faiss", "elasticsearch")
_SPEC_RETRIEVAL_RE = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(_SPEC_RETRIEVAL_KW, key=len, reverse=True))) + r')\b')

_SPEC_ML_KW        = ("machine learning", "ml ", " ml", "deep learning", "data science",
                      "nlp", "computer vision", "artificial intelligence", " ai ", "ai engineer")
_SPEC_ML_RE = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(_SPEC_ML_KW, key=len, reverse=True))) + r')\b')

_SPEC_BACKEND_KW   = ("backend", "software engineer", "platform", "infrastructure",
                      "full stack", "java", "c++", "golang", "microservices")
_SPEC_BACKEND_RE = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(_SPEC_BACKEND_KW, key=len, reverse=True))) + r')\b')

_SPEC_DATA_KW      = ("data engineer", "data pipeline", "etl", "analytics",
                      "sql", "hadoop", "spark", "kafka")
_SPEC_DATA_RE = re.compile(r'\b(?:' + '|'.join(map(re.escape, sorted(_SPEC_DATA_KW, key=len, reverse=True))) + r')\b')


# ─────────────────────────────────────────────────────────────────────────────
# SKILL TAXONOMY — authoritative mapping used by generate_reasoning — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────
SKILL_RANKING = {
    "learning to rank", "ltr", "lambdamart", "xgboost ranking",
    "recommendation systems", "recommender", "recommendation engine",
    "recommendation system", "collaborative filtering", "matrix factorization",
    "bpr", "svd", "als",
}
SKILL_SEARCH = {
    "opensearch", "elasticsearch", "solr", "bm25", "lucene",
    "search ranking", "information retrieval", "tfidf", "tf-idf",
    "inverted index", "full-text search", "keyword search",
}
SKILL_VECTOR = {
    "faiss", "pinecone", "qdrant", "weaviate", "milvus",
    "pgvector", "chroma", "vespa", "annoy", "scann",
    "hnsw", "approximate nearest neighbor", "ann search",
    "vector database", "vector db", "vector store",
    "vector search", "dense retrieval",
}
SKILL_SEMANTIC = {
    "semantic search", "sentence transformers", "bi-encoder", "cross-encoder",
    "embeddings", "text embeddings", "bert", "nlp", "natural language processing",
    "neural retrieval", "dense passage retrieval", "dpr",
}
SKILL_RAG = {
    "rag", "retrieval-augmented generation", "langchain", "llamaindex",
    "llm", "gpt", "openai", "llama", "generative ai", "prompt engineering",
}
