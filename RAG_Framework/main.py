import os
import sys
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from RAG_Framework.core.config import (
    EMBEDDING_MODEL_NAME, MODEL_PATH, RERANKER_MODEL_NAME,
    ENABLE_SERVER, SERVER_HOST, SERVER_PORT, EVAL, num_threads,
    ADVANCED_REASONING, ENABLE_SQL_DATABASES, SQL_DATABASE_CONFIGS
)

torch.set_num_threads(int(num_threads))
torch.set_num_interop_threads(int(num_threads))
faiss.omp_set_num_threads(int(num_threads))

from RAG_Framework.components.indexer import Indexer
from RAG_Framework.components.retrievers import Retriever
from RAG_Framework.components.generators import Generator
from RAG_Framework.core.conversation_manager import ConversationManager

# Global RAG system for server mode
rag_system = None

# Server module
if ENABLE_SERVER:
    from RAG_Framework.server import run_server

if __name__ == "__main__":

    print("Attempting to load saved FAISS index and BM25 index...")
    indexer = Indexer(enable_ocr=True)
    multi_vector_index, bm25, metadata_index, faiss_index = indexer.load_indices()

    if multi_vector_index is None or bm25 is None:
        print("No saved indices found. Proceeding to build indices.")
        print("Loading and chunking documents...")
        chunks, chunk_metadata = indexer.load_and_content_chunk_pdfs_parallel()
        if not chunks:
            print("No documents were loaded or processed. Exiting.")
            exit()

        print("\nLoading AI models...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        llm_model, llm_tokenizer = load(MODEL_PATH) # por adapter path aqui

        prompt_cache = make_prompt_cache(llm_model)
        conversation_manager = ConversationManager()

        reranker = CrossEncoder(RERANKER_MODEL_NAME)

        print("\nGenerating embeddings...")
        chunk_embeddings = indexer.get_embeddings(chunks, model=embedding_model)
        print("\nBuilding the multi-vector index...")
        multi_vector_index = indexer.build_multi_vector_index(chunk_embeddings, chunk_metadata, chunks)

        print("\nBuilding indexes...")
        bm25 = indexer.build_bm25_index(chunks)
        faiss_index = indexer.build_faiss_index(multi_vector_index)
        metadata_index = indexer.build_metadata_index(chunk_metadata)
        indexer.save_indices(multi_vector_index, bm25, metadata_index, faiss_index)
        print("Indices built and saved successfully.")
    else:
        print("Loaded saved indices from disk.")
        print("\nLoading AI models...")
    
        llm_model, llm_tokenizer = load(MODEL_PATH) # por adapter path aqui

        prompt_cache = make_prompt_cache(llm_model)
        conversation_manager = ConversationManager()

        reranker = CrossEncoder(RERANKER_MODEL_NAME) #TODO: dar load/off-load deste modelo apenas quando for necessario
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME) #TODO: dar load/off-load deste modelo apenas quando for necessario


    print("\nCreating Retriever...") # sets up retriever
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
    print(f"Index contains {len(multi_vector_index)} chunks.")

    # Attach the prompt cache and conversation manager to the retriever for easy access in server mode
    ############################# oq é q isto faz? ###############################################################################
    retriever.prompt_cache = prompt_cache
    retriever.conversation_manager = conversation_manager

    # Initialize SQL databases if enabled
    if ENABLE_SQL_DATABASES and SQL_DATABASE_CONFIGS:
        print("\nInitializing SQL databases...")
        from RAG_Framework.components.database import initialize_sql_connector, DatabaseConfig, DatabaseType

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
            from RAG_Framework.components.generators.reasoning import AgenticGenerator
            print("Using Advanced Reasoning mode with agentic generator")

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

                # Use the appropriate generator based on ADVANCED_REASONING
                if ADVANCED_REASONING:
                    rag_response = AgenticGenerator.agentic_answer_query(
                        query,
                        llm_model,
                        llm_tokenizer,
                        retriever,
                        prompt_cache,
                        conversation_manager
                    )
                else:
                    rag_response = Generator.answer_query_with_llm(
                        query,
                        llm_model,
                        llm_tokenizer,
                        retriever,
                        prompt_cache,
                        conversation_manager=conversation_manager #TODO: does this need the =conversation_manager ??????????????????
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