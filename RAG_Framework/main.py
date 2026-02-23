import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import faiss
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from RAG_Framework.core.config import (
    EMBEDDING_MODEL_NAME, MODEL_PATH, RERANKER_MODEL_NAME,
    ENABLE_SERVER, SERVER_HOST, SERVER_PORT, EVAL, num_threads,
    ADVANCED_REASONING, REASONING_MODEL, ENABLE_SQL_DATABASES, SQL_DATABASE_CONFIGS,
    CHECK_NEW_DOCUMENTS_ON_START, HIERARCHICAL_INDEXING
)

torch.set_num_threads(int(num_threads))
torch.set_num_interop_threads(int(num_threads))
faiss.omp_set_num_threads(int(num_threads))

from RAG_Framework.components.indexer import Indexer, HierarchicalIndexer, start_document_check
from RAG_Framework.components.retrievers import Retriever
from RAG_Framework.components.generators import Generator
from RAG_Framework.core.conversation_manager import ConversationManager

# Global RAG system for server mode
rag_system = None

# Server module
if ENABLE_SERVER:
    from RAG_Framework.server import run_server

if __name__ == "__main__":

    # Check if indices need building first
    print("Checking for saved indices...")
    if HIERARCHICAL_INDEXING:
        indexer = HierarchicalIndexer(enable_ocr=True)
    else:
        indexer = Indexer(enable_ocr=True)
    indices = indexer.load_indices()

    if indices[0] is None or indices[1] is None:
        print("No saved indices found. Proceeding to build indices.")
        print("Loading and chunking documents...")

        if HIERARCHICAL_INDEXING:
            chunks, chunk_metadata, parent_store = indexer.load_and_chunk_documents_parallel()
        else:
            chunks, chunk_metadata = indexer.load_and_chunk_documents_parallel()
            parent_store = None

        if not chunks:
            print("No documents were loaded or processed. Exiting.")
            exit()

        # Load embedding model temporarily for building indices
        print("\nLoading embedding model for index building...")
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print("\nGenerating embeddings...")
        chunk_embeddings = indexer.get_embeddings(chunks, model=embedding_model)
        print("\nBuilding the multi-vector index...")
        multi_vector_index = indexer.build_multi_vector_index(chunk_embeddings, chunk_metadata, chunks)

        print("\nBuilding indexes...")
        bm25 = indexer.build_bm25_index(chunks)
        faiss_index = indexer.build_faiss_index(multi_vector_index)
        metadata_index = indexer.build_metadata_index(chunk_metadata)
        if HIERARCHICAL_INDEXING:
            indexer.save_indices(multi_vector_index, bm25, metadata_index, faiss_index,
                                parent_store=parent_store)
        else:
            indexer.save_indices(multi_vector_index, bm25, metadata_index, faiss_index)
        print("Indices built and saved successfully.")

        # Free embedding model after building - will be lazy-loaded later if needed
        del embedding_model
        gc.collect()
    else:
        print("Saved indices found (will be lazy-loaded when needed).")

    # Load only LLM at startup (always needed)
    print("\nLoading LLM...")
    llm_model, llm_tokenizer = load(MODEL_PATH)
    prompt_cache = make_prompt_cache(llm_model)
    conversation_manager = ConversationManager(reasoning_model=REASONING_MODEL)

    # Create Retriever with lazy loading - embedding model, reranker, and indices
    # will only be loaded when document retrieval is first called
    print("\nCreating Retriever (models will be lazy-loaded on first use)...")
    retriever = Retriever(
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        embedding_model_name=EMBEDDING_MODEL_NAME,
        reranker_model_name=RERANKER_MODEL_NAME
    )

    # Attach the prompt cache and conversation manager to the retriever for easy access in server mode
    ############################# oq é q isto faz? ###############################################################################
    retriever.prompt_cache = prompt_cache
    retriever.conversation_manager = conversation_manager

    # Check for new documents in background (one-shot, non-blocking)
    if CHECK_NEW_DOCUMENTS_ON_START:
        start_document_check(retriever)

    # Initialize SQL databases if enabled
    if ENABLE_SQL_DATABASES and SQL_DATABASE_CONFIGS:
        print("\nInitializing SQL databases...")
        from RAG_Framework.tools.SQL_database import initialize_sql_connector, DatabaseConfig, DatabaseType

        db_configs = {}
        for db_name, config_dict in SQL_DATABASE_CONFIGS.items():
            try:
                db_configs[db_name] = DatabaseConfig(
                    db_type=DatabaseType(config_dict.get("db_type", "sqlite")),
                    connection_string=config_dict.get("connection_string", ""),
                    description=config_dict.get("description", ""),
                    max_rows=config_dict.get("max_rows", 100),
                    timeout=config_dict.get("timeout", 30),
                    allowed_tables=config_dict.get("allowed_tables")
                )
                print(f"  Configured database: {db_name} ({config_dict.get('db_type', 'sqlite')})")
            except Exception as e:
                print(f"  Warning: Failed to configure database '{db_name}': {e}")

        if db_configs:
            initialize_sql_connector(db_configs)
            print(f"SQL connector initialized with {len(db_configs)} database(s)")
        else:
            print("No valid SQL database configurations found")

    # Force cleanup
    import gc
    gc.collect()

    if ENABLE_SERVER:
        rag_system = retriever
        print(f"Starting RAG server on http://{SERVER_HOST}:{SERVER_PORT}")
        run_server(retriever, host=SERVER_HOST, port=SERVER_PORT)
    else:
        # Run in interactive console mode
        print("\nReady to answer queries. (Type 'exit' to quit)")

        if ADVANCED_REASONING:
            print("Using Advanced Reasoning mode (standard generator + advanced reasoning tool)")
        elif REASONING_MODEL:
            from RAG_Framework.components.generators.LRM import LRMGenerator

        try:
            while True:
                query = input("\nEnter your query: ")
                if query.lower() == "exit":
                    break
                if query.lower() == "clear":
                    conversation_manager.clear()
                    prompt_cache = make_prompt_cache(llm_model)
                    retriever.prompt_cache = prompt_cache
                    print("Conversation and cache cleared.")
                    continue

                # Use the appropriate generator
                if REASONING_MODEL and not ADVANCED_REASONING:
                    rag_response = LRMGenerator.answer_query_with_llm(
                        query,
                        llm_model,
                        llm_tokenizer,
                        retriever,
                        prompt_cache,
                        conversation_manager=conversation_manager
                    )
                else:
                    # Standard generator — includes activate_advanced_reasoning tool
                    # when ADVANCED_REASONING = True in config
                    rag_response = Generator.answer_query_with_llm(
                        query,
                        llm_model,
                        llm_tokenizer,
                        retriever,
                        prompt_cache,
                        conversation_manager=conversation_manager
                    )

                # Ensure rag_response_tuple is a tuple before unpacking
                if isinstance(rag_response, tuple):
                    response_text, _ = rag_response
                else:
                    response_text = rag_response

                #print("\nRAG Response:\n", response_text)

                if EVAL:
                    print("==========")
                    # Pass the query and the response text (not the tuple) to eval
                    Generator.eval(
                        query,
                        llm_model,
                        llm_tokenizer,
                        response_text
                    )
        except KeyboardInterrupt:
            print("\nExiting program.")