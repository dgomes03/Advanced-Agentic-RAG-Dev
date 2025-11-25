# Advanced Agentic RAG System - Modular Architecture

## Overview

This is a modular refactoring of the Advanced RAG Combined Retrieval system. The code has been organized into logical modules for better maintainability, testability, and scalability.

## Project Structure

```
Advanced-Agentic-RAG-Dev/
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── indexing/                 # Document indexing
│   │   ├── __init__.py
│   │   └── indexer.py           # Indexer class
│   ├── retrieval/               # Document retrieval
│   │   ├── __init__.py
│   │   └── retriever.py         # Retriever class
│   ├── generation/              # LLM response generation
│   │   ├── __init__.py
│   │   └── generator.py         # Generator class
│   ├── agentic/                 # Agentic reasoning
│   │   ├── __init__.py
│   │   ├── models.py            # Data models (ReasoningPlan, etc.)
│   │   ├── planner.py           # AgenticPlanner
│   │   ├── evaluator.py         # AgenticEvaluator
│   │   └── generator.py         # AgenticGenerator
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── helpers.py           # Helper functions
├── main.py                      # Main entry point
├── Indexes/                     # Generated indices (created at runtime)
└── Advanced_RAG_CombinedRetrieval.py  # Original monolithic file (kept for reference)
```

## Key Modules

### 1. **config.py** - Configuration Management
- Centralized configuration using dataclass
- Environment variable support
- Validation and warnings for missing configuration

**Usage:**
```python
from src.config import config

print(config.DOCUMENTS_DIR)
print(config.MAX_RESPONSE_TOKENS)
```

**Environment Variables:**
- `DOCUMENTS_DIR`: Path to documents directory
- `MODEL_PATH`: Path to LLM model
- `EMBEDDING_MODEL_NAME`: Sentence transformer model name
- `GOOGLE_API_KEY`: Google API key for custom search
- `GOOGLE_CX`: Google Custom Search Engine ID
- `ADVANCED_REASONING`: Enable advanced reasoning (true/false)
- And more...

### 2. **indexing/** - Document Indexing
Contains the `Indexer` class for:
- PDF processing with parallel execution
- Embedding generation
- FAISS index building
- BM25 index building
- Metadata index building

### 3. **retrieval/** - Document Retrieval
Contains the `Retriever` class for:
- Dense retrieval (FAISS)
- Sparse retrieval (BM25)
- Hybrid retrieval with score fusion
- Query expansion
- Reranking with cross-encoder
- Metadata-based retrieval

### 4. **generation/** - Response Generation
Contains the `Generator` class for:
- LLM-based answer generation
- Tool calling (document search, Wikipedia, Google)
- Wikipedia API integration
- Google Custom Search integration
- Evaluation metrics

### 5. **agentic/** - Agentic Reasoning
Contains components for multi-step reasoning:
- **models.py**: Data structures (ReasoningPlan, ReasoningGoal, ReasoningState)
- **planner.py**: Query decomposition and replanning
- **evaluator.py**: Goal and overall completeness evaluation
- **generator.py**: Main agentic reasoning loop

### 6. **utils/** - Utilities
Helper functions:
- Pickle save/load
- Common utilities

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables (optional):
```bash
export DOCUMENTS_DIR="/path/to/documents"
export MODEL_PATH="/path/to/model"
export GOOGLE_API_KEY="your-api-key"
export GOOGLE_CX="your-cx-id"
export ADVANCED_REASONING="true"
```

Alternatively, create a `.env` file:
```
DOCUMENTS_DIR=/path/to/documents
MODEL_PATH=/path/to/model
GOOGLE_API_KEY=your-api-key
GOOGLE_CX=your-cx-id
ADVANCED_REASONING=true
```

## Usage

### Running the System

**Interactive Mode:**
```bash
python main.py
```

**Server Mode:**
Set `ENABLE_SERVER=true` in environment or config, then:
```bash
python main.py
```

### Programmatic Usage

```python
from src.indexing import Indexer
from src.retrieval import Retriever
from src.generation import Generator
from src.config import config

# Initialize indexer
indexer = Indexer()
multi_vector_index, bm25, metadata_index, faiss_index = indexer.load_indices()

# Create retriever
retriever = Retriever(
    multi_vector_index,
    embedding_model,
    faiss_index,
    bm25,
    metadata_index,
    llm_model,
    llm_tokenizer,
    reranker
)

# Generate answer
response = Generator.answer_query_with_llm(
    "What is machine learning?",
    llm_model,
    llm_tokenizer,
    retriever,
    prompt_cache
)
```

## Benefits of Modular Architecture

1. **Maintainability**: Each module has a clear responsibility
2. **Testability**: Individual components can be tested in isolation
3. **Scalability**: Easy to add new features or replace components
4. **Reusability**: Modules can be imported and used independently
5. **Configuration**: Centralized configuration with environment variable support
6. **Security**: Sensitive data (API keys) can be stored in environment variables

## Migration from Original File

The original `Advanced_RAG_CombinedRetrieval.py` has been kept for reference. To migrate:

1. **Use the new main.py**: `python main.py` instead of running the old file
2. **Update imports**: If you have external scripts importing from the old file, update to:
   ```python
   from src.indexing import Indexer
   from src.retrieval import Retriever
   from src.generation import Generator
   ```
3. **Configuration**: Move hardcoded values to environment variables

## Testing

To verify the modular system works:

```bash
# Check syntax
python -m py_compile main.py

# Check imports
python -c "from src.indexing import Indexer; from src.retrieval import Retriever; from src.generation import Generator; print('✓ All imports successful')"

# Run the system
python main.py
```

## Development

### Adding a New Retrieval Method

1. Add the method to `src/retrieval/retriever.py`:
```python
def new_retrieval_method(self, query, k=10):
    # Implementation
    return results
```

2. Use it in the `combined_retrieval` method or create a new tool

### Adding a New Tool

1. Define the tool in `src/generation/generator.py` in the `answer_query_with_llm` method
2. Implement the tool function (e.g., `search_new_source`)
3. Add tool execution logic in the tool calling loop

### Extending Configuration

1. Add new fields to the `RAGConfig` dataclass in `src/config.py`:
```python
NEW_PARAM: str = os.getenv("NEW_PARAM", "default_value")
```

2. Use it in your code:
```python
from src.config import config
print(config.NEW_PARAM)
```

## Troubleshooting

**Import Errors:**
- Ensure you're running from the project root directory
- Check that all `__init__.py` files are present

**Configuration Issues:**
- Verify environment variables are set correctly
- Check `src/config.py` for default values

**Index Not Found:**
- Run the system once to build indices
- Indices are saved in the `Indexes/` directory

## Future Improvements

- [ ] Add unit tests for each module
- [ ] Implement caching for query results
- [ ] Add logging with configurable levels
- [ ] Implement Reciprocal Rank Fusion (RRF)
- [ ] Add Maximum Marginal Relevance (MMR) for diversity
- [ ] Implement semantic chunking with overlap
- [ ] Add benchmarking and performance metrics

## License

[Your License Here]
