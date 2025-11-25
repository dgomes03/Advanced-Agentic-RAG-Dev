#!/usr/bin/env python3
"""
Advanced Agentic RAG System - Main Entry Point
"""

import gc
import torch
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from src.config import config
from src.indexing import Indexer
from src.retrieval import Retriever
from src.generation import Generator

# Set torch threads
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
faiss.omp_set_num_threads(1)

# Global RAG system for server mode
rag_system = None


def setup_server():
    """Setup Flask server for RAG system"""
    from flask import Flask, request, jsonify, Response, stream_with_context
    from flask_cors import CORS
    import json
    import time

    app = Flask(__name__)
    CORS(app)

    @app.route('/query', methods=['POST'])
    def handle_query():
        data = request.get_json()
        query = data.get('query', '')
        if not query:
            return jsonify({'error': 'No query provided'}), 400

        def generate():
            try:
                # Stream the response token by token
                for text_chunk in Generator.answer_query_with_llm(
                    query,
                    rag_system.llm_model,
                    rag_system.llm_tokenizer,
                    rag_system,
                    []
                ):
                    yield f"data: {json.dumps({'text': text_chunk})}\n"
                    time.sleep(0.01)
                yield "data: [DONE]\n"
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                yield f"data: {json.dumps({'error': error_msg})}\n"
                yield "data: [DONE]\n"

        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    @app.route('/status', methods=['GET'])
    def status():
        return jsonify({'status': 'ready'})

    return app


def main():
    """Main function to run the RAG system"""
    global rag_system

    print("Attempting to load saved FAISS index and BM25 index...")
    indexer = Indexer()
    multi_vector_index, bm25, metadata_index, faiss_index = indexer.load_indices()

    if multi_vector_index is None or bm25 is None:
        print("No saved indices found. Proceeding to build indices.")
        print("Loading and chunking documents...")
        chunks, chunk_metadata = indexer.load_and_content_chunk_pdfs_parallel()

        if not chunks:
            print("No documents were loaded or processed. Exiting.")
            return

        print("\nLoading AI models...")
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        llm_model, llm_tokenizer = load(config.MODEL_PATH)
        prompt_cache = make_prompt_cache(llm_model)
        reranker = CrossEncoder(config.RERANKER_MODEL_NAME)

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
        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        llm_model, llm_tokenizer = load(config.MODEL_PATH)
        prompt_cache = make_prompt_cache(llm_model)
        reranker = CrossEncoder(config.RERANKER_MODEL_NAME)

    print("\nCreating Retriever...")
    retriever = Retriever(
        multi_vector_index,
        embedding_model,
        faiss_index,
        bm25,
        metadata_index,
        llm_model,
        llm_tokenizer,
        reranker,
    )
    print(f"Index contains {len(multi_vector_index)} chunks.")

    # Attach the prompt cache to the retriever instance
    retriever.prompt_cache = prompt_cache

    # Force cleanup
    gc.collect()

    if config.ENABLE_SERVER:
        rag_system = retriever
        print(f"Starting RAG server on http://{config.SERVER_HOST}:{config.SERVER_PORT}")
        app = setup_server()
        app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=False, use_reloader=False)
    else:
        # Run in interactive console mode
        print("\nReady to answer queries. (Type 'exit' to quit)")
        try:
            while True:
                query = input("\nEnter your query: ")
                if query.lower() == "exit":
                    break

                rag_response = Generator.answer_query_with_llm(
                    query,
                    llm_model,
                    llm_tokenizer,
                    retriever,
                    prompt_cache
                )

                # Ensure rag_response is handled correctly
                if isinstance(rag_response, tuple):
                    response_text, _ = rag_response
                else:
                    response_text = rag_response

                if config.EVAL:
                    print("==========")
                    Generator.eval(
                        query,
                        llm_model,
                        llm_tokenizer,
                        response_text
                    )
        except KeyboardInterrupt:
            print("\nExiting program.")


if __name__ == "__main__":
    main()
