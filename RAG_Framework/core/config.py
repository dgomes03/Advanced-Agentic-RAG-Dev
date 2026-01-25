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


# === Server Configuration ===
ENABLE_SERVER = True
SERVER_HOST = 'localhost'
SERVER_PORT = 5050

# === Reasoning Configuration ===
ADVANCED_REASONING = False
MAX_REASONING_STEPS = 5
MIN_CONFIDENCE_THRESHOLD = 0.7

# === Cache Configuration ===
ENABLE_SELECTIVE_CACHING = True  # Exclude tool results from KV cache to reduce RAM usage

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