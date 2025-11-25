"""
Configuration module for Advanced RAG System
Centralizes all configuration parameters and provides environment variable support
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Set environment variables for single-threaded performance
num_threads = "1"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["PYTORCH_NUM_THREADS"] = num_threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class RAGConfig:
    """Main configuration for the RAG system"""

    # Directory and Model Paths
    DOCUMENTS_DIR: str = os.getenv(
        "DOCUMENTS_DIR",
        "/Users/diogogomes/Documents/Uni/Tese Mestrado/RAG_database"
    )
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH",
        "/Users/diogogomes/.lmstudio/models/mlx-community/Ministral-8b-instruct-mixed-6-8-bit"
    )

    # Model Names
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    )
    RERANKER_MODEL_NAME: str = os.getenv(
        "RERANKER_MODEL_NAME",
        'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
    )

    # Index Paths
    MULTIVECTOR_INDEX_PATH: str = "Indexes/FAISS_index.pkl"
    BM25_DATA_PATH: str = "Indexes/BM25_index.pkl"
    METADATA_INDEX_PATH: str = "Indexes/metadata_index.pkl"
    FAISS_INDEX_PATH: str = "Indexes/faiss_index.faiss"

    # Generation Parameters
    MAX_RESPONSE_TOKENS: int = int(os.getenv("MAX_RESPONSE_TOKENS", "1000"))

    # Evaluation Mode
    EVAL: bool = os.getenv("EVAL", "False").lower() == "true"

    # Server Configuration
    ENABLE_SERVER: bool = os.getenv("ENABLE_SERVER", "False").lower() == "true"
    SERVER_HOST: str = os.getenv("SERVER_HOST", "localhost")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "5050"))

    # Advanced Reasoning Configuration
    ADVANCED_REASONING: bool = os.getenv("ADVANCED_REASONING", "False").lower() == "true"
    MAX_REASONING_STEPS: int = int(os.getenv("MAX_REASONING_STEPS", "5"))
    MIN_CONFIDENCE_THRESHOLD: float = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.7"))

    # API Keys (IMPORTANT: Use environment variables in production!)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_CX: str = os.getenv("GOOGLE_CX", "")

    # Retrieval Parameters
    DEFAULT_K: int = 7
    DEFAULT_WEIGHT_DENSE: float = 0.6
    DEFAULT_WEIGHT_SPARSE: float = 0.4
    DEFAULT_RERANK_TOP_N: int = 5

    # Processing Parameters
    MAX_CHUNK_LENGTH: int = 512
    EMBEDDING_BATCH_SIZE: int = 32
    PDF_PROCESSING_WORKERS: int = 4

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.ENABLE_SERVER and not self.SERVER_HOST:
            raise ValueError("SERVER_HOST must be set when ENABLE_SERVER is True")

        if self.ADVANCED_REASONING and self.MAX_REASONING_STEPS < 1:
            raise ValueError("MAX_REASONING_STEPS must be at least 1")

        # Warn about missing API keys if they're needed
        if not self.GOOGLE_API_KEY and not self.GOOGLE_CX:
            import warnings
            warnings.warn(
                "Google API credentials not set. Google Custom Search will not work. "
                "Set GOOGLE_API_KEY and GOOGLE_CX environment variables."
            )


# Global configuration instance
config = RAGConfig()
