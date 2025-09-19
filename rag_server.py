import os
import io
import json
import random
import string
import pickle
import numpy as np
from collections import defaultdict
import faiss
import pymupdf as fitz
from sentence_transformers import SentenceTransformer, CrossEncoder
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.models.cache import make_prompt_cache
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import torch
import pdfplumber
import re
import pytesseract
from PIL import Image
import concurrent.futures
from nltk.tokenize import word_tokenize
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
from flask import Response, stream_with_context
from Advanced_RAG_CombinedRetrieval import Indexer, Retriever, Generator

# === Constants ===
DOCUMENTS_DIR = "/Users/diogogomes/Documents/Uni/Tese Mestrado/RAG_database"
MODEL_PATH = "/Users/diogogomes/.lmstudio/models/mlx-community/Ministral-8b-instruct-mixed-6-8-bit"
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
MULTIVECTOR_INDEX_PATH = "Indexes/FAISS_index.pkl"
BM25_DATA_PATH = "Indexes/BM25_index.pkl"
RERANKER_MODEL_NAME = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
MAX_RESPONSE_TOKENS = 1000

# Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global RAG system
rag_system = None

def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        return None

# Flask routes
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
                rag_system.search_documents_tool,
                []
            ):
                yield f"data: {json.dumps({'text': text_chunk})}\n\n"
                time.sleep(0.01)  # Small delay to make streaming visible
                
            # Signal the end of the stream
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield f"data: {json.dumps({'error': error_msg})}\n\n"
            yield "data: [DONE]\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'ready'})

def initialize_rag_system():
    global rag_system
    print("Attempting to load saved FAISS index and BM25 index...")
    indexer = Indexer()
    multi_vector_index, bm25, faiss_index = indexer.load_indices()

    if multi_vector_index is None or bm25 is None:
        print("No saved indices found. Proceeding to build indices.")
        print("Loading and chunking documents...")
        chunks, chunk_metadata = indexer.load_and_content_chunk_pdfs_parallel()
        if not chunks:
            print("No documents were loaded. Exiting.")
            exit()

        print("\nLoading AI models...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        llm_model, llm_tokenizer = load(MODEL_PATH)
        reranker = CrossEncoder(RERANKER_MODEL_NAME)

        print("\nGenerating embeddings...")
        chunk_embeddings = indexer.get_embeddings(chunks, model=embedding_model)
        
        print("\nBuilding the multi-vector index...")
        multi_vector_index = indexer.build_multi_vector_index(chunk_embeddings, chunk_metadata, chunks)
        indexer.save_indices(multi_vector_index, None, None)  # Save BM25

        print("\nBuilding BM25 index...")
        bm25 = indexer.build_bm25_index(chunks)
        indexer.save_indices(multi_vector_index, bm25, None)

    else:
        print("Loaded saved FAISS and BM25 indices from disk.")
        print("\nLoading AI models...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        llm_model, llm_tokenizer = load(MODEL_PATH) #, adapter_path='/Users/diogogomes/Documents/Uni/Tese Mestrado/Fine-tuning/adapters')
        reranker = CrossEncoder(RERANKER_MODEL_NAME)

    if faiss_index is None and multi_vector_index is not None:
        print("Building FAISS index...")
        faiss_index = Indexer.build_faiss_index(multi_vector_index)
        indexer.save_indices(multi_vector_index, bm25, faiss_index)

    print("\nCreating Retriever...")
    rag_system = Retriever(
        multi_vector_index,
        embedding_model,
        faiss_index,  #FAISS index
        bm25,
        llm_model,
        llm_tokenizer,
        reranker,
    )

    print(f"Index contains {len(multi_vector_index)} chunks.")
    # Force cleanup
    import gc
    gc.collect()
    print("RAG system initialized successfully!")

if __name__ == '__main__':
    # Initialize the RAG system
    initialize_rag_system()
    
    # Start the Flask server
    print("Starting RAG server on http://localhost:5050")
    app.run(host='localhost', port=5050, debug=False, use_reloader=False)