import os
import sys
import gc

# Add parent directory to path to allow imports to work from any location
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

# Import configuration
from RAG_Framework.core.config import (
    EMBEDDING_MODEL_NAME, MODEL_PATH, RERANKER_MODEL_NAME,
    ENABLE_SERVER, SERVER_HOST, SERVER_PORT, EVAL, num_threads,
    ADVANCED_REASONING
)

# Set threading limits
torch.set_num_threads(int(num_threads))
torch.set_num_interop_threads(int(num_threads))
faiss.omp_set_num_threads(int(num_threads))

# Import components
from RAG_Framework.components.indexer import Indexer
from RAG_Framework.components.retrievers import Retriever
from RAG_Framework.components.generators import Generator

# Global RAG system for server mode
rag_system = None

# Server module (only import if server is enabled)
if ENABLE_SERVER:
    from RAG_Framework.server import run_server


# ==== Main Runtime ====
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
        reranker = CrossEncoder(RERANKER_MODEL_NAME)

        print("\nGenerating embeddings...")
        chunk_embeddings = indexer.get_embeddings(chunks, model=embedding_model)
        print("\nBuilding the multi-vector index...")
        multi_vector_index = indexer.build_multi_vector_index(chunk_embeddings, chunk_metadata, chunks)

        print("\nBuilding BM25 index...")
        bm25 = indexer.build_bm25_index(chunks)

        print("\nBuilding FAISS index...")
        faiss_index = indexer.build_faiss_index(multi_vector_index)

        print("\nBuilding Metadata index...")
        metadata_index = indexer.build_metadata_index(chunk_metadata)

        print("\nSaving all indices...")
        indexer.save_indices(multi_vector_index, bm25, metadata_index, faiss_index)
        print("Indices built and saved successfully.")
    else:
        print("Loaded saved indices from disk.")
        print("\nLoading AI models...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        llm_model, llm_tokenizer = load(MODEL_PATH) # por adapter path aqui
        prompt_cache = make_prompt_cache(llm_model)
        reranker = CrossEncoder(RERANKER_MODEL_NAME)

    print("\nCreating Retriever...")
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
    # Attach the prompt cache to the retriever instance for easy access if needed in server mode
    retriever.prompt_cache = prompt_cache

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

        # Import the appropriate generator based on ADVANCED_REASONING
        if ADVANCED_REASONING:
            from RAG_Framework.components.generators.reasoning import AgenticGenerator
            print("Using Advanced Reasoning mode with agentic generator")

        try:
            while True:
                query = input("\nEnter your query: ")
                if query.lower() == "exit":
                    break

                # Use the appropriate generator based on ADVANCED_REASONING
                if ADVANCED_REASONING:
                    rag_response = AgenticGenerator.agentic_answer_query(
                        query,
                        llm_model,
                        llm_tokenizer,
                        retriever,
                        prompt_cache
                    )
                else:
                    rag_response = Generator.answer_query_with_llm(
                        query,
                        llm_model,
                        llm_tokenizer,
                        retriever,
                        prompt_cache
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
                        response_text # FIXED: Pass the response text, not the tuple
                    )
        except KeyboardInterrupt:
            print("\nExiting program.")