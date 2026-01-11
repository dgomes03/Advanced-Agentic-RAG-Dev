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
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-base'
MULTIVECTOR_INDEX_PATH = os.path.join(PROJECT_ROOT, "Indexes", "FAISS_index.pkl")
BM25_DATA_PATH = os.path.join(PROJECT_ROOT, "Indexes", "BM25_index.pkl")
METADATA_INDEX_PATH = os.path.join(PROJECT_ROOT, "Indexes", "metadata_index.pkl")
FAISS_INDEX_PATH = os.path.join(PROJECT_ROOT, "Indexes", "faiss_index.faiss")
RERANKER_MODEL_NAME = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
MAX_RESPONSE_TOKENS = 1000
EVAL = False


# === Server Configuration ===
ENABLE_SERVER = False
SERVER_HOST = 'localhost'
SERVER_PORT = 5050

# === Reasoning Configuration ===
ADVANCED_REASONING = False
MAX_REASONING_STEPS = 5
MIN_CONFIDENCE_THRESHOLD = 0.7
GOOGLE_API_KEY = "AIzaSyAXXtU2WSpdM-sUR2z7c19CcDBqXQ1zhug"
GOOGLE_CX = "27078d51accb54f1d" # Google Custom Search Engine ID

if ENABLE_SERVER:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import threading
    import time
    from flask import Response, stream_with_context