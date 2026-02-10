import os

# Set environment variables for single-threaded performance
num_threads = "1"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["PYTORCH_NUM_THREADS"] = num_threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Project Root ===
# Get the project root directory (parent of RAG_Framework)
PROJECT_ROOT = "/Users/diogogomes/Documents/Uni/Tese Mestrado"

# === Constants ===
DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, "RAG_database")
MODEL_PATH = "/Users/diogogomes/.lmstudio/models/mlx-community/Ministral-3-8B-Instruct-2512-mixed-8-6-bit"
ADVANCED_REASONING_AGENT_PATH = "/Users/diogogomes/.lmstudio/models/mlx-community/Ministral-3-3B-Instruct-2512-mixed-8-6-bit"
ADVANCED_REASONING_RESPONSE_PATH = "/Users/diogogomes/.lmstudio/models/mlx-community/Ministral-3-8B-Instruct-2512-mixed-8-6-bit"
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-base'
MULTIVECTOR_INDEX_PATH = os.path.join(PROJECT_ROOT, "Indexes", "FAISS_index.pkl")
BM25_DATA_PATH = os.path.join(PROJECT_ROOT, "Indexes", "BM25_index.pkl")
METADATA_INDEX_PATH = os.path.join(PROJECT_ROOT, "Indexes", "metadata_index.pkl")
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "Indexes", "faiss_index.faiss")
RERANKER_MODEL_NAME = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
MAX_RESPONSE_TOKENS = 1000
EVAL = False

# === Chunking Configuration ===
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128  # 25% overlap to preserve boundary information
MIN_CHUNK_CHARS = 20  # Minimum chunk length; filters junk OCR/table fragments
PARENT_CHUNK_SIZE = 2048  # Parent chunks for hierarchical indexing

HIERARCHICAL_INDEXING = False  # Toggle: True = hierarchical, False = standard flat chunking
PARENT_STORE_PATH = os.path.join(PROJECT_ROOT, "Indexes", "parent_store.pkl")

# === BM25 Configuration ===
BM25_ENABLE_STEMMING = True
BM25_ENABLE_STOPWORDS = True
BM25_LANGUAGES = ['english', 'portuguese']  # Languages for stemmer and stop words

# === Retrieval Configuration ===
RETRIEVAL_TOP_K = 40        # Number of candidates retrieved from FAISS + BM25
RERANKER_TOP_N = 20         # Number of results kept after reranking
PARENT_TOP_N = 10            # Max parent chunks returned after dedup (hierarchical mode only)

# === FAISS Configuration ===
FAISS_INDEX_TYPE = 'auto'  # Options: 'flat', 'ivf', 'hnsw', 'auto'
FAISS_IVF_NPROBE_RATIO = 4  # nprobe = nlist // this ratio

# === Embedding Configuration ===
EMBEDDING_USE_PREFIX = True  # Add E5 instruction prefixes (query:/passage:)


# === Document Watcher Configuration ===
CHECK_NEW_DOCUMENTS_ON_START = False  # Check for new/modified documents at startup

# === Server Configuration ===
ENABLE_SERVER = True
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5050

# === Reasoning Configuration ===
ADVANCED_REASONING = False
MAX_REASONING_STEPS = 5
MIN_CONFIDENCE_THRESHOLD = 0.7

# === Google Custom Search Configuration ===
GOOGLE_API_KEY = "AIzaSyAXXtU2WSpdM-sUR2z7c19CcDBqXQ1zhug"
GOOGLE_CX = "27078d51accb54f1d" # Google Custom Search Engine ID

# === SQL Database Configuration ===
ENABLE_SQL_DATABASES = True
SQL_DATABASE_CONFIGS = {
    "demo_db": {
        "db_type": "sqlite",
        "connection_string": os.path.join(PROJECT_ROOT, "RAG_Framework", "data", "demo_data.db"),
        "description": "Demo database with products, customers, and orders for testing SQL queries",
        "max_rows": 100,
        "timeout": 30,
        "allowed_tables": ["products", "customers", "orders"],
    }
}

# === RAGAS Evaluation Configuration ===
EVAL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "RAG_Framework", "evaluation", "results")
EVAL_BENCHMARK_DATASET = "explodinggradients/amnesty_qa"
EVAL_BENCHMARK_DATASET_SUBSET = "english_v3"
EVAL_MAX_QUESTIONS = None  # None = all, or int to limit
EVAL_LLM_MAX_TOKENS = 2048
EVAL_LLM_TEMPERATURE = 0.1