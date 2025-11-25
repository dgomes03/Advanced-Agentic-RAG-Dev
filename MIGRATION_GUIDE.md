# Migration Guide: Monolithic to Modular Architecture

## Overview

This guide helps you migrate from the original `Advanced_RAG_CombinedRetrieval.py` (1981 lines) to the new modular architecture.

## What Changed

### Before (Monolithic)
```
Advanced_RAG_CombinedRetrieval.py  (1981 lines)
```

### After (Modular)
```
src/
├── config.py                 (92 lines)
├── indexing/indexer.py      (289 lines)
├── retrieval/retriever.py   (250 lines)
├── generation/generator.py  (446 lines)
├── agentic/
│   ├── models.py            (56 lines)
│   ├── planner.py           (144 lines)
│   ├── evaluator.py         (125 lines)
│   └── generator.py         (387 lines)
└── utils/helpers.py         (19 lines)
main.py                      (154 lines)
```

## Import Changes

### Old Way
```python
# Everything in one file
from Advanced_RAG_CombinedRetrieval import Indexer, Retriever, Generator
```

### New Way
```python
# Organized imports
from src.config import config
from src.indexing import Indexer
from src.retrieval import Retriever
from src.generation import Generator
from src.agentic import AgenticGenerator, AgenticPlanner, AgenticEvaluator
```

## Configuration Changes

### Old Way (Hardcoded)
```python
# In the file
DOCUMENTS_DIR = "/Users/diogogomes/Documents/Uni/Tese Mestrado/RAG_database"
MODEL_PATH = "/Users/diogogomes/.lmstudio/models/..."
GOOGLE_API_KEY = "AIzaSyAXXtU2WSpdM-sUR2z7c19CcDBqXQ1zhug"  # EXPOSED!
```

### New Way (Environment Variables)
```bash
# .env file
DOCUMENTS_DIR=/path/to/documents
MODEL_PATH=/path/to/model
GOOGLE_API_KEY=your-api-key-here
GOOGLE_CX=your-cx-here
ADVANCED_REASONING=true
```

```python
# In code
from src.config import config
print(config.DOCUMENTS_DIR)
```

## Running the System

### Old Way
```bash
python Advanced_RAG_CombinedRetrieval.py
```

### New Way
```bash
# Set environment variables (optional)
export DOCUMENTS_DIR=/path/to/docs

# Run
python main.py
```

## Code Usage Examples

### Example 1: Using the Indexer

**Old:**
```python
# Inside Advanced_RAG_CombinedRetrieval.py
indexer = Indexer()
chunks, metadata = indexer.load_and_content_chunk_pdfs_parallel()
```

**New:**
```python
from src.indexing import Indexer

indexer = Indexer()
chunks, metadata = indexer.load_and_content_chunk_pdfs_parallel()
```

### Example 2: Using the Retriever

**Old:**
```python
# Inside Advanced_RAG_CombinedRetrieval.py
retriever = Retriever(multi_vector_index, embedding_model, faiss_index,
                     bm25, metadata_index, llm_model, llm_tokenizer, reranker)
results = retriever.combined_retrieval(query)
```

**New:**
```python
from src.retrieval import Retriever

retriever = Retriever(multi_vector_index, embedding_model, faiss_index,
                     bm25, metadata_index, llm_model, llm_tokenizer, reranker)
results = retriever.combined_retrieval(query)
```

### Example 3: Using the Generator

**Old:**
```python
# Inside Advanced_RAG_CombinedRetrieval.py
response = Generator.answer_query_with_llm(query, llm_model, llm_tokenizer, retriever)
```

**New:**
```python
from src.generation import Generator

response = Generator.answer_query_with_llm(query, llm_model, llm_tokenizer, retriever)
```

### Example 4: Using Agentic Components

**Old:**
```python
# Inside Advanced_RAG_CombinedRetrieval.py
plan = AgenticPlanner.create_initial_plan(query, llm_model, llm_tokenizer)
evaluation = AgenticEvaluator.evaluate_goal_completion(goal, context, llm_model, llm_tokenizer)
response = AgenticGenerator.agentic_answer_query(query, llm_model, llm_tokenizer, retriever)
```

**New:**
```python
from src.agentic import AgenticPlanner, AgenticEvaluator, AgenticGenerator

plan = AgenticPlanner.create_initial_plan(query, llm_model, llm_tokenizer)
evaluation = AgenticEvaluator.evaluate_goal_completion(goal, context, llm_model, llm_tokenizer)
response = AgenticGenerator.agentic_answer_query(query, llm_model, llm_tokenizer, retriever)
```

## External Scripts Migration

If you have external scripts that import from the old file:

### Before
```python
from Advanced_RAG_CombinedRetrieval import Indexer, Retriever, Generator, config

# Use components
indexer = Indexer(config.DOCUMENTS_DIR)
```

### After
```python
from src.config import config
from src.indexing import Indexer
from src.retrieval import Retriever
from src.generation import Generator

# Use components
indexer = Indexer()  # Uses config.DOCUMENTS_DIR automatically
```

## Testing Your Migration

### Step 1: Verify Syntax
```bash
python -m py_compile main.py
```

### Step 2: Check Imports
```bash
python -c "from src.indexing import Indexer; print('✓ Imports work')"
```

### Step 3: Run the System
```bash
python main.py
```

### Step 4: Test a Query
```python
# In the interactive prompt
Enter your query: What is machine learning?
```

## Common Issues and Solutions

### Issue 1: Import Errors
**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Make sure you're running from the project root:
```bash
cd /home/user/Advanced-Agentic-RAG-Dev
python main.py
```

### Issue 2: Configuration Not Found
**Error:** `FileNotFoundError: Documents directory not found`

**Solution:** Set the `DOCUMENTS_DIR` environment variable:
```bash
export DOCUMENTS_DIR=/path/to/your/documents
python main.py
```

### Issue 3: API Keys Not Working
**Error:** `Error: Google Custom Search Engine ID (CX) is not configured`

**Solution:** Set environment variables:
```bash
export GOOGLE_API_KEY=your-key
export GOOGLE_CX=your-cx
python main.py
```

## Benefits of Migration

1. **Better Organization**: Code is split into logical modules
2. **Easier Maintenance**: Changes are isolated to specific modules
3. **Improved Security**: API keys in environment variables, not code
4. **Better Testing**: Each module can be tested independently
5. **Reusability**: Modules can be imported in other projects
6. **Scalability**: Easier to add new features
7. **Configuration**: Centralized and flexible configuration

## Rollback Plan

If you need to rollback to the old system:

1. The original file `Advanced_RAG_CombinedRetrieval.py` is still present
2. Simply run: `python Advanced_RAG_CombinedRetrieval.py`

However, note that the new modular system is fully compatible and offers many advantages!

## Next Steps

1. ✅ Update your environment variables
2. ✅ Test the new system with a sample query
3. ✅ Update any external scripts to use new imports
4. ✅ Consider adding unit tests for your use case
5. ✅ Explore the new configuration options in `src/config.py`

## Support

For issues or questions:
- Check `README_MODULAR.md` for detailed documentation
- Review the code in each module
- Compare with the original `Advanced_RAG_CombinedRetrieval.py`

## Backward Compatibility

The new modular system maintains the same API as the original:
- Same class names
- Same method signatures
- Same functionality
- Enhanced with better configuration management

You can migrate gradually without breaking existing functionality!
