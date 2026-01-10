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
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from sentence_transformers import SentenceTransformer, CrossEncoder
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import torch
import pdfplumber
import re
import easyocr
from PIL import Image
from multiprocessing import Pool # ADDED: Import Pool for parallel processing

# Set environment variables for single-threaded performance
num_threads = "1"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["PYTORCH_NUM_THREADS"] = num_threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(int(num_threads))
torch.set_num_interop_threads(int(num_threads))
faiss.omp_set_num_threads(int(num_threads))

# === Constants ===
DOCUMENTS_DIR = "/Users/diogogomes/Documents/Uni/Tese Mestrado/RAG_database"
MODEL_PATH = "/Users/diogogomes/.lmstudio/models/mlx-community/Ministral-3-8B-Instruct-2512-mixed-8-6-bit"
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-base'
MULTIVECTOR_INDEX_PATH = "Indexes/FAISS_index.pkl"
BM25_DATA_PATH = "Indexes/BM25_index.pkl"
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

# Global RAG system for server mode
rag_system = None

def save_pickle(obj, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True) # Ensure directory exists
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
    else:
        return None

# Server routes (only if server is enabled)
if ENABLE_SERVER:
    app = Flask(__name__)
    CORS(app)  # Enable Cross-Origin Resource Sharing

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
                    yield f"data: {json.dumps({'text': text_chunk})}\n"
                    time.sleep(0.01)  # Small delay to make streaming visible
                # Signal the end of the stream
                yield "data: [DONE]\n"
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                yield f"data: {json.dumps({'error': error_msg})}\n"
                yield "data: [DONE]\n"

        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    @app.route('/status', methods=['GET'])
    def status():
        return jsonify({'status': 'ready'})


class Indexer:
    def __init__(self, documents_dir=DOCUMENTS_DIR, embedding_model_name=EMBEDDING_MODEL_NAME,
                 faiss_index_path=MULTIVECTOR_INDEX_PATH, bm25_data_path=BM25_DATA_PATH,
                 metadata_index_path="Indexes/metadata_index.pkl"):
        self.documents_dir = documents_dir
        self.embedding_model_name = embedding_model_name
        self.faiss_index_path = faiss_index_path
        self.bm25_data_path = bm25_data_path
        self.metadata_index_path = metadata_index_path

    @staticmethod
    def process_single_pdf(file_info):
        file_path, doc_id, max_chunk_length = file_info
        doc_chunks = []
        doc_metadata = []
        filename = os.path.basename(file_path)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_number, page in enumerate(pdf.pages):
                    elements = []

                    # Extract tables
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table and any(cell for row in table for cell in row if cell): # Check for non-empty table
                            table_text = '\n'.join([' | '.join(map(str, row)) for row in table if row])
                            elements.append({
                                'type': 'table',
                                'content': table_text,
                                'meta': {
                                    'page': page_number,
                                    'filename': filename,
                                    'doc_id': doc_id,
                                    'table_idx': table_idx,
                                    'document_name': filename,
                                    'full_document_id': f"{filename}_{doc_id}"
                                }
                            })

                    # Extract paragraph text
                    text = page.extract_text()
                    if text:
                        # Use sentence-aware splitting instead of fixed length
                        sentences = re.split(r'(?<=[.!?])\s+', text)
                        current_chunk = []
                        current_length = 0
                        chunks = []
                        for sentence in sentences:
                            if current_length + len(sentence) > max_chunk_length and current_chunk:
                                chunks.append(" ".join(current_chunk).strip()) # Strip whitespace
                                current_chunk = [sentence]
                                current_length = len(sentence)
                            else:
                                current_chunk.append(sentence)
                                current_length += len(sentence) + 1
                        if current_chunk:
                            chunks.append(" ".join(current_chunk).strip()) # Strip whitespace
                        for chunk_idx, chunk in enumerate(chunks):
                            if chunk: # Only add non-empty chunks
                                elements.append({
                                    'type': 'text',
                                    'content': chunk,
                                    'meta': {
                                        'page': page_number,
                                        'filename': filename,
                                        'doc_id': doc_id,
                                        'chunk_idx': chunk_idx,
                                        'document_name': filename,
                                        'full_document_id': f"{filename}_{doc_id}"
                                    }
                                })

                    # OCR for images
                    if page.images:
                        # Initialize EasyOCR reader (will be cached after first use in the process)
                        if not hasattr(Indexer, '_easyocr_reader'):
                            Indexer._easyocr_reader = easyocr.Reader(['en', 'pt'], gpu=False)

                        for img_idx, img_dict in enumerate(page.images):
                            try:
                                # Check if image has coordinates
                                if 'x0' in img_dict and 'y0' in img_dict and 'x1' in img_dict and 'y1' in img_dict:
                                    # Get coordinates and ensure they're valid
                                    img_x0 = img_dict["x0"]
                                    img_top = img_dict["top"]
                                    img_x1 = img_dict["x1"]
                                    img_bottom = img_dict["bottom"]

                                    # Validate coordinates: ensure x1 > x0 and bottom > top
                                    if img_x1 <= img_x0 or img_bottom <= img_top:
                                        continue  # Skip invalid images

                                    # Clamp coordinates to page boundaries
                                    img_x0 = max(0, min(img_x0, page.width))
                                    img_x1 = max(0, min(img_x1, page.width))
                                    img_top = max(0, min(img_top, page.height))
                                    img_bottom = max(0, min(img_bottom, page.height))

                                    # Skip if area is too small (likely invalid)
                                    if (img_x1 - img_x0) < 10 or (img_bottom - img_top) < 10:
                                        continue

                                    image = page.crop((img_x0, img_top, img_x1, img_bottom)).to_image(resolution=150)
                                    pil_img = image.original

                                    # Use EasyOCR to extract text
                                    result = Indexer._easyocr_reader.readtext(np.array(pil_img))
                                    ocr_text = ' '.join([text for (_, text, _) in result]).strip()

                                    if ocr_text:
                                        elements.append({
                                            'type': 'image_ocr',
                                            'content': ocr_text,
                                            'meta': {
                                                'page': page_number,
                                                'filename': filename,
                                                'doc_id': doc_id,
                                                'img_idx': img_idx,
                                                'document_name': filename,
                                                'full_document_id': f"{filename}_{doc_id}"
                                            }
                                        })
                            except Exception as e:
                                print(f"Failed OCR on image {img_idx} of page {page_number} in {filename}: {e}")

                    # Save elements
                    for elem in elements:
                        doc_chunks.append(elem['content'])
                        meta = elem['meta'].copy()
                        meta['type'] = elem['type']
                        doc_metadata.append(meta)
            print(f"Processed {filename}")
        except Exception as e:
            print(f"Could not read/process {filename}: {e}")
        # FIXED: Return the correct variables
        return doc_chunks, doc_metadata

    def build_metadata_index(self, all_metadata):
        """
        Build a reverse index mapping document names to chunk indices
        Returns: dict with structure {document_name: [list_of_chunk_indices]}
        """
        metadata_index = {}
        for idx, meta in enumerate(all_metadata):
            doc_name = meta.get('document_name')
            full_doc_id = meta.get('full_document_id')
            if doc_name and doc_name not in metadata_index: # Check if doc_name exists
                metadata_index[doc_name] = []
            if doc_name:
                metadata_index[doc_name].append(idx)
            if full_doc_id and full_doc_id not in metadata_index: # Check if full_doc_id exists
                metadata_index[full_doc_id] = []
            if full_doc_id:
                metadata_index[full_doc_id].append(idx)
        return metadata_index

    # ADDED: Method to get embeddings
    def get_embeddings(self, texts, model, batch_size=32):
        """Generate embeddings for a list of texts in batches."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
            batch = texts[i:i+batch_size]
            embeddings = model.encode(batch)
            all_embeddings.extend(embeddings)
        return np.array(all_embeddings, dtype=np.float32)


    # ADDED: Method to build FAISS index from embeddings
    def build_faiss_index(self, multi_vector_index):
        if not multi_vector_index:
            print("Warning: Multi-vector index is empty. Cannot build FAISS index.")
            return None
        embeddings = np.array([item["embedding"] for item in multi_vector_index]).astype('float32')
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension) # Inner Product for normalized vectors
        faiss.normalize_L2(embeddings) # Ensure embeddings are normalized
        index.add(embeddings)
        return index

    # ADDED: Method to build BM25 index
    def build_bm25_index(self, texts):
        tokenized_corpus = [doc.lower().split() for doc in texts]
        bm25_index = BM25Okapi(tokenized_corpus)
        return bm25_index

    def build_multi_vector_index(self, embeddings, metadata, texts):
        index = []
        for emb, meta, text in zip(embeddings, metadata, texts):
            index.append({
                "embedding": np.array(emb, dtype=np.float32),
                "text": text,
                "metadata": meta
            })
        return index

    def save_indices(self, multi_vector_index, bm25, metadata_index, faiss_index=None):
        """Save all indices including the new metadata index"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
        save_pickle(multi_vector_index, self.faiss_index_path)
        save_pickle(bm25, self.bm25_data_path)
        save_pickle(metadata_index, self.metadata_index_path)  # Save metadata index
        if faiss_index:
            faiss.write_index(faiss_index, "Indexes/faiss_index.faiss")

    def load_indices(self):
        """Load all indices including metadata index"""
        multi_vector_index = load_pickle(self.faiss_index_path)
        bm25 = load_pickle(self.bm25_data_path)
        metadata_index = load_pickle(self.metadata_index_path)
        faiss_index = None
        if os.path.exists("Indexes/faiss_index.faiss"):
            faiss_index = faiss.read_index("Indexes/faiss_index.faiss")
        return multi_vector_index, bm25, metadata_index, faiss_index

    def retrieve_by_metadata(self, metadata_index, multi_vector_index, document_name):
        """
        Retrieve all chunks for a specific document name
        Returns: list of chunks with their metadata
        """
        if document_name not in metadata_index:
            return []
        chunk_indices = metadata_index[document_name]
        results = []
        for idx in chunk_indices:
            if idx < len(multi_vector_index):
                results.append(multi_vector_index[idx])
        return results

    def load_and_content_chunk_pdfs_parallel(self, max_chunk_length=512):
        """Load and process PDF documents in parallel"""
        pdf_files = [os.path.join(self.documents_dir, f) for f in os.listdir(self.documents_dir)
                    if f.lower().endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files to process.")
        if not pdf_files:
            return [], []

        file_infos = [(f, i, max_chunk_length) for i, f in enumerate(pdf_files)]

        # Process files in parallel
        with Pool(processes=4) as pool: # ADDED: Pool import and usage
            results = pool.map(self.process_single_pdf, file_infos)

        all_chunks = []
        all_metadata = []
        for chunks, metadata in results:
            all_chunks.extend(chunks)
            all_metadata.extend(metadata)
        return all_chunks, all_metadata


class Retriever:
    def __init__(self, multi_vector_index, embedding_model, faiss_index,
                 bm25, metadata_index,
                 llm_model, llm_tokenizer, reranker):
        self.multi_vector_index = multi_vector_index
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.bm25 = bm25
        self.metadata_index = metadata_index
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.reranker = reranker
        self.last_retrieved_metadata = [] # ADDED: Initialize attribute

    @staticmethod
    def normalize_scores(scores):
        if not scores:
            return {}
        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)
        if max_score - min_score < 1e-9:
            return {k: 1.0 for k in scores}
        return {k: (v - min_score) / (max_score - min_score) for k, v in scores.items()}

    @staticmethod
    def retrieve_with_faiss(query_embedding, faiss_index, k=10):
        if faiss_index is None:
             print("Warning: FAISS index is not available.")
             return []
        query_vec = np.array(query_embedding).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_vec)
        distances, indices = faiss_index.search(query_vec, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((idx, float(dist)))
        return results

    @staticmethod
    def retrieve_with_bm25(tokenized_query, bm25, top_k=10):
        if bm25 is None:
             print("Warning: BM25 index is not available.")
             return []
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(idx, scores[idx]) for idx in top_indices]

    def retrieve_by_metadata(self, document_name, sort_by_page=True):
        """
        Retrieve all chunks for a specific document name
        Returns: list of chunks with their metadata, sorted by page number
        """
        original_query = document_name
        if document_name not in self.metadata_index:
            # Try to find partial matches for more flexible retrieval
            matching_docs = [doc for doc in self.metadata_index.keys()
                           if document_name.lower() in doc.lower()]
            if not matching_docs:
                return []
            # Use the first match, or could return all matches
            document_name = matching_docs[0]
            print(f"Using partial match: '{document_name}' for query: '{original_query}'")

        chunk_indices = self.metadata_index[document_name]
        results = []
        for idx in chunk_indices:
            if idx < len(self.multi_vector_index):
                results.append({
                    'text': self.multi_vector_index[idx]["text"],
                    'metadata': self.multi_vector_index[idx]["metadata"],
                    'index': idx
                })

        # Sort by page number and chunk index for coherent reading order
        if sort_by_page and results:
            results.sort(key=lambda x: (
                x['metadata'].get('page', 0),
                x['metadata'].get('chunk_idx', 0),
                x['metadata'].get('table_idx', 0),
                x['metadata'].get('img_idx', 0)
            ))
        return results

    def expand_query(self, query):
        system_prompt = (
            "You are a query rephraser and expander for document search.\n"
            "Generate two alternative phrasings "
            "that capture different ways the same question could be asked for retrieving relevant context.\n"
            "Also provide between 1 to 5 buzz words from the phrases that can be relevant for retrieving relevant context.\n"
            "IMPORTANT: Dont include any other commentary like 'Buzz words:' at the end!"
        )
        message = (
            f"ORIGINAL QUERY:\n{query}")
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        prompt = self.llm_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        response = generate(
            self.llm_model,
            self.llm_tokenizer,
            prompt=prompt,
            max_tokens=200,
            verbose=False)
        response = re.sub(r'(?i)\b(?:here\s+are\s+the\s+)?buzz\s*words?\s*:\s*', '', response)
        alternatives = []
        for line in response.split('\n'):
            line = line.strip('"-â ¢* 0123456789.').strip()
            if 2 < len(line):
                alternatives.append(line)
        if not alternatives:
            alternatives = [query]
        return alternatives

    def rerank_chunks_with_metadata(self, query, chunks, metadata_list, top_n=None):
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.reranker.predict(pairs)
        combined = list(zip(chunks, metadata_list, scores))
        reranked = sorted(combined, key=lambda x: x[2], reverse=True)
        if top_n:
            reranked = reranked[:top_n]
        return [(chunk, meta) for chunk, meta, score in reranked]

    def combined_retrieval(
            self,
            query,
            k=7,
            weight_dense=0.6,
            weight_sparse=0.4,
            rerank_top_n=5,
            use_summarization=False,
    ):
        if not self.faiss_index or not self.bm25 or not self.multi_vector_index:
             print("Warning: FAISS index, BM25 index, or multi-vector index is not available for retrieval.")
             return "No documents available for retrieval."

        # Expand query variants for retrieval
        expanded_queries = [query] + self.expand_query(query)

        # Dense retrieval scoring
        dense_scores = defaultdict(float)
        query_embeddings = self.embedding_model.encode(expanded_queries)
        for query_emb in query_embeddings:
            faiss_results = self.retrieve_with_faiss(query_emb, self.faiss_index, k=k)
            for idx, score in faiss_results:
                dense_scores[idx] += score
        dense_scores_norm = self.normalize_scores(dense_scores)

        # Sparse retrieval scoring (BM25)
        sparse_scores = defaultdict(float)
        for exp_query in expanded_queries:
            tokenized_query = exp_query.lower().split()
            bm25_results = self.retrieve_with_bm25(tokenized_query, self.bm25, top_k=k)
            for idx, score in bm25_results:
                sparse_scores[idx] += score
        sparse_scores_norm = self.normalize_scores(sparse_scores)

        # Combine scores
        all_doc_indices = set(dense_scores_norm.keys()) | set(sparse_scores_norm.keys())
        combined_scores = {}
        for idx in all_doc_indices:
            dscore = dense_scores_norm.get(idx, 0)
            sscore = sparse_scores_norm.get(idx, 0)
            combined_scores[idx] = weight_dense * dscore + weight_sparse * sscore

        # Select top-k indices
        top_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:k]

        # Retrieve chunks and their metadata for top candidates
        retrieved_chunks = [self.multi_vector_index[idx]["text"] for idx in top_indices]
        retrieved_metadata = [self.multi_vector_index[idx]["metadata"] for idx in top_indices]

        # Rerank chunks while preserving metadata
        reranked = self.rerank_chunks_with_metadata(query, retrieved_chunks, retrieved_metadata, top_n=rerank_top_n)

        # Store metadata for citation tracking
        self.last_retrieved_metadata = [meta for _, meta in reranked]

        if use_summarization:
            reranked_texts = [chunk for chunk, _ in reranked]
            final_context = Generator.summarize_passages(reranked_texts, self.llm_model, self.llm_tokenizer)
        else:
            final_context = "\n---\n".join([chunk for chunk, _ in reranked])
        return final_context

    # Existing semantic search tool
    def search_documents_tool(self, query: str) -> str:
        """Tool for semantic search across all documents"""
        try:
            return self.combined_retrieval(
                query=query,
                k=20, # ADDED: Reduced k for initial retrieval
                weight_dense=0.6,
                weight_sparse=0.4,
                rerank_top_n=10,
                use_summarization=False
            )
        except Exception as e:
            return f"Error during search: {str(e)}"

    # New metadata-based retrieval tool
    def retrieve_document_by_name_tool(self, document_name: str) -> str:
        """
        Tool for retrieving entire documents by name
        Returns all chunks from the specified document
        """
        try:
            results = self.retrieve_by_metadata(document_name)
            if not results:
                # More helpful error message with better matching suggestions
                all_docs = list(self.metadata_index.keys())
                # Try to find similar document names
                similar = [doc for doc in all_docs if document_name.lower() in doc.lower() or doc.lower() in document_name.lower()]

                error_msg = f"Document '{document_name}' not found.\n"
                if similar:
                    error_msg += f"\nDid you mean one of these?\n"
                    for doc in similar[:5]:
                        error_msg += f"  - {doc}\n"
                else:
                    error_msg += f"\nAvailable documents (first 10):\n"
                    for doc in sorted(all_docs)[:10]:
                        error_msg += f"  - {doc}\n"
                    if len(all_docs) > 10:
                        error_msg += f"\n... and {len(all_docs) - 10} more. Use list_available_documents to see all."

                return error_msg

            # Combine all chunks with their metadata for context
            document_content = []
            for result in results:
                chunk_text = result['text']
                metadata = result['metadata']
                # Add citation info
                citation = f"[Page {metadata.get('page', 'N/A')}, Chunk {metadata.get('chunk_idx', 'N/A')}]"
                document_content.append(f"{chunk_text}\n{citation}")

            full_document = "\n--- Page Break ---\n".join(document_content)

            # Add document summary header
            doc_metadata = results[0]['metadata']
            header = f"DOCUMENT: {doc_metadata.get('document_name', document_name)}\n"
            header += f"TOTAL CHUNKS: {len(results)}\n"
            header += f"PAGES: {max(r['metadata'].get('page', 0) for r in results) + 1}\n"
            header += "="*50 + "\n"

            return header + full_document
        except Exception as e:
            import traceback
            return f"Error retrieving document '{document_name}': {str(e)}\n{traceback.format_exc()}"

    # Helper tool to list available documents
    def list_available_documents_tool(self, filter_keyword: str = "") -> str:
        """Tool to list all available documents in the index, optionally filtered by keyword"""
        try:
            doc_names = list(self.metadata_index.keys())
            # Group documents: prefer entries without _N suffix (actual filenames)
            # but include _N entries if they're the only option
            seen_base_names = set()
            meaningful_docs = []

            # First pass: add documents that don't match the full_document_id pattern
            for doc_name in sorted(doc_names):
                # Check if this looks like a full_document_id (ends with _number)
                if '_' in doc_name:
                    parts = doc_name.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        base_name = parts[0]
                        seen_base_names.add(base_name)
                        continue
                meaningful_docs.append(doc_name)

            # Second pass: add full_document_id entries only if base name wasn't found
            for doc_name in sorted(doc_names):
                if '_' in doc_name:
                    parts = doc_name.rsplit('_', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        base_name = parts[0]
                        if base_name not in meaningful_docs and doc_name not in meaningful_docs:
                            meaningful_docs.append(doc_name)

            if not meaningful_docs:
                meaningful_docs = doc_names

            # Apply filter if provided
            if filter_keyword:
                filter_lower = filter_keyword.lower()
                filtered_docs = [doc for doc in meaningful_docs if filter_lower in doc.lower()]
                if not filtered_docs:
                    return f"No documents found matching '{filter_keyword}'. Try a different search term or use list_available_documents without a filter to see all documents."
                meaningful_docs = filtered_docs

            document_info = []
            for doc_name in sorted(meaningful_docs):
                chunk_count = len(self.metadata_index[doc_name])
                document_info.append(f"- {doc_name} ({chunk_count} chunks)")

            total_docs = len(meaningful_docs)
            if filter_keyword:
                header = f"Documents matching '{filter_keyword}' ({total_docs} found):\n"
            else:
                header = f"Available documents ({total_docs} total):\n"

            return header + "\n".join(document_info)
        except Exception as e:
            return f"Error listing documents: {str(e)}"

# === Agentic Components ===

class ReasoningState(Enum):
    """States in the reasoning process"""
    INITIAL_QUERY = "initial_query"
    PLANNING = "planning"
    SEARCHING = "searching"
    EVALUATING = "evaluating"
    REPLANNING = "replanning"
    ANSWERING = "answering"
    COMPLETE = "complete"

@dataclass
class ReasoningGoal:
    """Represents a sub-goal in multi-step reasoning"""
    description: str
    priority: int
    status: str = "pending"  # pending, in_progress, completed, failed
    retrieved_info: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
@dataclass
class ReasoningPlan:
    """Complete reasoning plan with multiple goals"""
    main_query: str
    goals: List[ReasoningGoal] = field(default_factory=list)
    current_step: int = 0
    total_confidence: float = 0.0
    
    def add_goal(self, goal: ReasoningGoal):
        self.goals.append(goal)
    
    def get_next_goal(self) -> Optional[ReasoningGoal]:
        """Get next pending or in_progress goal"""
        for goal in sorted(self.goals, key=lambda g: g.priority):
            if goal.status in ["pending", "in_progress"]:
                return goal
        return None
    
    def is_complete(self) -> bool:
        """Check if all goals are completed"""
        return all(g.status == "completed" for g in self.goals)
    
    def get_completion_rate(self) -> float:
        """Get percentage of completed goals"""
        if not self.goals:
            return 0.0
        completed = sum(1 for g in self.goals if g.status == "completed")
        return completed / len(self.goals)


class AgenticPlanner:
    """Handles planning and replanning of reasoning steps"""
    
    @staticmethod
    def create_initial_plan(query: str, llm_model, llm_tokenizer) -> ReasoningPlan:
        """Create initial reasoning plan by decomposing the query"""

        system_prompt = """You are a query decomposition expert. Break down queries into sub-goals.
            For each sub-goal:
            1. Describe what information is needed
            2. Assign priority (1=highest, 5=lowest)
            3. Keep descriptions clear and searchable

            Output ONLY valid JSON in this exact format:
            {
                "goals": [
                    {"description": "goal description", "priority": 1},
                    {"description": "another goal", "priority": 2}
                ]
            }"""
        
        user_prompt = f"""Query: {query}

            Decompose this into 2-3 searchable sub-goals. Output JSON only."""
        
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = llm_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        response = generate(
            llm_model,
            llm_tokenizer,
            prompt=prompt,
            max_tokens=500,
            verbose=False
        )
        
        # Parse the response
        plan = ReasoningPlan(main_query=query)
        try:
            # Clean response and extract JSON
            response = response.strip()
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()
            
            plan_data = json.loads(response)
            for goal_data in plan_data.get("goals", []):
                goal = ReasoningGoal(
                    description=goal_data["description"],
                    priority=goal_data.get("priority", 3)
                )
                plan.add_goal(goal)
            
            print(f"\nCreated plan with {len(plan.goals)} goals:")
            for i, g in enumerate(plan.goals, 1):
                print(f"  {i}. [{g.priority}] {g.description}")
                
        except Exception as e:
            print(f"Plan parsing failed: {e}")
            # Fallback: treat entire query as single goal
            plan.add_goal(ReasoningGoal(
                description=query,
                priority=1
            ))
        
        return plan
    
    @staticmethod
    def replan(plan: ReasoningPlan, evaluation: Dict, llm_model, llm_tokenizer) -> ReasoningPlan:

        """Replan based on evaluation results"""

        system_prompt = """You are a reasoning coordinator. Based on the evaluation, decide if we need additional search goals.

            Output ONLY valid JSON:
            {
                "needs_replanning": true/false,
                "new_goals": [
                    {"description": "new goal if needed", "priority": 1}
                ],
                "reasoning": "brief explanation"
            }"""
        
        current_state = {
            "completed_goals": [g.description for g in plan.goals if g.status == "completed"],
            "pending_goals": [g.description for g in plan.goals if g.status != "completed"],
            "evaluation": evaluation
        }
        
        user_prompt = f"""Current state: {json.dumps(current_state, indent=2)}

            Should we add more search goals? Output JSON only."""
        
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = llm_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        response = generate(
            llm_model,
            llm_tokenizer,
            prompt=prompt,
            max_tokens=400,
            verbose=False
        )
        
        try:
            # Clean and parse response
            response = response.strip()
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()
            
            replan_data = json.loads(response)
            
            if replan_data.get("needs_replanning", False):
                print(f"\nReplanning: {replan_data.get('reasoning', 'Adding new goals')}")
                for goal_data in replan_data.get("new_goals", []):
                    new_goal = ReasoningGoal(
                        description=goal_data["description"],
                        priority=goal_data.get("priority", 3)
                    )
                    plan.add_goal(new_goal)
                    print(f"  + Added: {new_goal.description}")
        except Exception as e:
            print(f"Replan parsing failed: {e}")
        
        return plan


class AgenticEvaluator:
    """Evaluates completeness and quality of retrieved information"""
    
    @staticmethod
    def evaluate_goal_completion(
        goal: ReasoningGoal,
        retrieved_context: str,
        llm_model,
        llm_tokenizer
    ) -> Dict[str, Any]:
        """Evaluate if a goal has been satisfactorily completed"""
        system_prompt = """You are an information completeness evaluator. Assess if the retrieved context adequately addresses the goal.

            Output ONLY valid JSON:
            {
                "is_complete": true/false,
                "confidence": 0.0-1.0,
                "missing_aspects": ["aspect1", "aspect2"],
                "reasoning": "brief explanation"
            }"""
        
        user_prompt = f"""Goal: {goal.description}

Retrieved context:
{retrieved_context[:1000]}...

Is this sufficient? Output JSON only."""
        
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = llm_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        response = generate(
            llm_model,
            llm_tokenizer,
            prompt=prompt,
            max_tokens=300,
            verbose=False
        )
        
        try:
            # Clean and parse
            response = response.strip()
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()
            
            eval_result = json.loads(response)
            return eval_result
        except Exception as e:
            print(f"Evaluation parsing failed: {e}")
            # Fallback evaluation
            return {
                "is_complete": len(retrieved_context) > 100,
                "confidence": 0.5,
                "missing_aspects": [],
                "reasoning": "Fallback evaluation"
            }
    
    @staticmethod
    def evaluate_overall_completeness(
        plan: ReasoningPlan,
        llm_model,
        llm_tokenizer
    ) -> Dict[str, Any]:
        """Evaluate overall completeness of the reasoning process"""
        all_info = []
        for goal in plan.goals:
            if goal.retrieved_info:
                all_info.extend(goal.retrieved_info)
        
        system_prompt = """You are a final completeness evaluator. Assess if we have enough information to answer the original query comprehensively.

            Output ONLY valid JSON:
            {
                "can_answer": true/false,
                "overall_confidence": 0.0-1.0,
                "coverage_assessment": "brief assessment",
                "needs_more_search": true/false
            }"""
        
        # adicionar KV-Cache a isto !!!!!!!!!
        combined_info = "\n".join(all_info[:8000])  # Limit context
        
        user_prompt = f"""Original query: {plan.main_query}

            Completed goals: {plan.get_completion_rate()*100:.0f}%

            Retrieved information:
            {combined_info}

            Can we answer comprehensively? Output JSON only."""
        
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = llm_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        response = generate(
            llm_model,
            llm_tokenizer,
            prompt=prompt,
            max_tokens=300,
            verbose=False
        )
        
        try:
            response = response.strip()
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                response = response.split('```')[1].split('```')[0].strip()
            
            eval_result = json.loads(response)
            return eval_result
        except Exception as e:
            print(f"⚠️  Overall evaluation parsing failed: {e}")
            return {
                "can_answer": plan.get_completion_rate() > 0.6,
                "overall_confidence": plan.get_completion_rate(),
                "coverage_assessment": "Fallback assessment",
                "needs_more_search": plan.get_completion_rate() < 0.6
            }


class AgenticGenerator:

    """Main agentic reasoning coordinator"""
    
    @staticmethod
    def agentic_answer_query(
        query: str,
        llm_model,
        llm_tokenizer,
        retriever,
        prompt_cache=None
    ) -> str:
        """
        Main agentic loop with multi-step reasoning, evaluation, and replanning
        """
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}\n")
        
        # Step 1: Create initial plan
        print("Creating reasoning plan...")
        plan = AgenticPlanner.create_initial_plan(query, llm_model, llm_tokenizer)
        
        reasoning_step = 0
        max_steps = MAX_REASONING_STEPS
        
        # Step 2: Iterative reasoning loop
        while reasoning_step < max_steps:
            reasoning_step += 1
            print(f"\n{'─'*60}")
            print(f"REASONING STEP {reasoning_step}/{max_steps}")
            print(f"{'─'*60}")
            
            # Get next goal to work on
            current_goal = plan.get_next_goal()
            
            if current_goal is None:
                print("All goals completed!")
                break
            
            print(f"Current Goal: {current_goal.description}")
            print(f"Priority: {current_goal.priority} | Status: {current_goal.status}")
            
            # Mark goal as in progress
            current_goal.status = "in_progress"
            
            # Step 3: Search for information
            print(f"\nSearching for relevant information...")
            try:
                retrieved_context = AgenticGenerator.available_tools(current_goal.description, llm_model, llm_tokenizer, retriever)
                current_goal.retrieved_info.append(retrieved_context)
                print(f"Retrieved {len(retrieved_context)} characters of context")
            except Exception as e:
                print(f"Search failed: {e}")
                current_goal.status = "failed"
                continue
            
            # Step 4: Evaluate goal completion
            print(f"\nEvaluating goal completion...")
            evaluation = AgenticEvaluator.evaluate_goal_completion(
                current_goal,
                retrieved_context,
                llm_model,
                llm_tokenizer
            )
            
            current_goal.confidence = evaluation.get("confidence", 0.5)
            
            print(f"Complete: {evaluation.get('is_complete', False)}")
            print(f"Confidence: {current_goal.confidence:.2f}")
            print(f"Reasoning: {evaluation.get('reasoning', 'N/A')}")
            
            if evaluation.get("missing_aspects"):
                print(f"Missing: {', '.join(evaluation.get('missing_aspects', []))}")
            
            # Update goal status based on evaluation
            if evaluation.get("is_complete", False) and current_goal.confidence >= MIN_CONFIDENCE_THRESHOLD:
                current_goal.status = "completed"
                print(f"Goal completed successfully!")
            elif current_goal.confidence < MIN_CONFIDENCE_THRESHOLD:
                print(f"Low confidence - may need more information")
                current_goal.status = "completed"  # Move on but flag low confidence
            else:
                current_goal.status = "completed"  # Mark as done even if not perfect
            
            # Step 5: Check overall progress and decide if replanning needed
            print(f"\nOverall Progress: {plan.get_completion_rate()*100:.0f}% complete")
            
            # Evaluate overall completeness
            if plan.get_completion_rate() >= 0.5:  # Check after 50% completion
                print(f"\nEvaluating overall completeness...")
                overall_eval = AgenticEvaluator.evaluate_overall_completeness(
                    plan,
                    llm_model,
                    llm_tokenizer
                )
                
                print(f"Can answer: {overall_eval.get('can_answer', False)}")
                print(f"Overall confidence: {overall_eval.get('overall_confidence', 0):.2f}")
                print(f"Assessment: {overall_eval.get('coverage_assessment', 'N/A')}")
                
                # Step 6: Autonomous stopping decision
                if overall_eval.get("can_answer", False) and overall_eval.get("overall_confidence", 0) >= MIN_CONFIDENCE_THRESHOLD:
                    print(f"\nAUTONOMOUS STOP: Sufficient information gathered")
                    print(f"Confidence threshold met: {overall_eval.get('overall_confidence', 0):.2f} >= {MIN_CONFIDENCE_THRESHOLD}")
                    break
                
                # Step 7: Dynamic replanning if needed
                if overall_eval.get("needs_more_search", False) and reasoning_step < max_steps - 1:
                    print(f"\nReplanning needed...")
                    plan = AgenticPlanner.replan(plan, overall_eval, llm_model, llm_tokenizer)
        
        # Step 8: Generate final answer
        print(f"\n{'='*60}")
        print(f"GENERATING FINAL ANSWER")
        print(f"{'='*60}")
        
        current_response = Generator.answer_query_with_llm(query, llm_model, llm_tokenizer, retriever, prompt_cache=prompt_cache)
        

        return current_response
    
    @staticmethod
    def available_tools(query, llm_model, llm_tokenizer, retriever, prompt_cache=None):
        # Updated tools list with metadata retrieval capabilities
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_documents",
                    "description": "Search for documents relevant to the user's query. Use for general questions or when you need to find information across all documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "retrieve_document_by_name",
                    "description": "Retrieve an entire document by its filename. Use when user specifically asks for a particular book, report, or document by name.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_name": {
                                "type": "string",
                                "description": "The name of the document to retrieve (e.g., 'annual_report.pdf')"
                            }
                        },
                        "required": ["document_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_available_documents",
                    "description": "List available documents in the system. Use when user asks what documents are available. Can filter by keyword - if user asks about specific topics (e.g., 'documents about allometric'), use the filter_keyword parameter to search document names.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filter_keyword": {
                                "type": "string",
                                "description": "Optional keyword to filter document names (e.g., 'allometric', 'climate', 'policy'). Leave empty to list all documents."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_wikipedia",
                    "description": "Search Wikipedia for factual information and general knowledge.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query string for Wikipedia."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "google_custom_search",
                    "description": "Search the internet using Google Custom Search JSON API. Use for current events or information not found in documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query string."}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        # Temperature and top sampling
        sampler = make_sampler(temp=0.7, top_k=50, top_p=0.9)
        logits_processors = make_logits_processors(repetition_penalty=1.1, repetition_context_size=128)
        current_response = None

        # Enhanced system prompt to guide tool selection
        conversation = [
            {"role": "system", "content": "You are a helpful assistant with document search and Wikipedia search capabilities. "
            "Decide whether you need tools. If you use tools, answer based *only* on the provided results. "
            "If you're not sure if the user wants you to access tools, ask the user. "
            "After receiving tool results, provide a final answer. "
            "If not enough information is found after tool calling, alert the user that there's no available information to answer the user. "
            "At the end of an informative response, ask if the user needs more information or wants to explore more a certain fact. "
            "**Do not make sequential tool calls**!"}
        ]

        # Add current query
        conversation.append({"role": "user", "content": query})
        
        while True: ############### este loop é mesmo necessário? ##################
            prompt = llm_tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tools=tools,
                tokenize=False
            )

            #print("\nFORMATTED PROMPT:") 
            #print(prompt)

            response = generate(
                model=llm_model,
                tokenizer=llm_tokenizer,
                prompt=prompt,
                max_tokens=MAX_RESPONSE_TOKENS,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=prompt_cache,
                verbose=True
            )

            response_text = response.strip()

            # tool call present?
            if "[TOOL_CALLS]" in response_text:
                print(f"\nModel requested to use tools. Processing tool calls...")
                try:
                    tool_calls = []
                    
                    # Check which format we're dealing with
                    if "[TOOL_CALLS][" in response_text:
                        # OLD FORMAT: [TOOL_CALLS][{"name": "...", "arguments": {...}}]
                        start_marker = "[TOOL_CALLS]["
                        start_idx = response_text.find(start_marker) + len(start_marker) - 1
                        bracket_count = 1
                        end_idx = start_idx + 1
                        while end_idx < len(response_text) and bracket_count > 0:
                            if response_text[end_idx] == '[':
                                bracket_count += 1
                            elif response_text[end_idx] == ']':
                                bracket_count -= 1
                            end_idx += 1
                        tool_json = response_text[start_idx:end_idx]
                        tool_calls = json.loads(tool_json)
                        
                    elif "[ARGS]" in response_text:
                        # NEW FORMAT: [TOOL_CALLS]tool_name[ARGS]{"key": "value"}
                        tool_section = response_text.split("[TOOL_CALLS]")[1]
                        tool_name, args_part = tool_section.split("[ARGS]", 1)
                        tool_name = tool_name.strip()
                        
                        # Extract JSON object
                        args_part = args_part.strip()
                        brace_count = 0
                        end_idx = 0
                        for i, char in enumerate(args_part):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i + 1
                                    break
                        
                        args_json = args_part[:end_idx] if end_idx > 0 else "{}"
                        tool_args = json.loads(args_json) if args_json else {}
                        tool_calls = [{"name": tool_name, "arguments": tool_args}]
                    
                    else:
                        print("Unknown tool call format")
                        break

                    if not tool_calls:
                        print("No valid tool calls found")
                        break

                    print(f"Found {len(tool_calls)} tool call(s): {[call.get('name', 'unknown') for call in tool_calls]}")

                    # Add tool call to conversation
                    formatted_tool_calls = []
                    for call in tool_calls:
                        call_id = ''.join(random.choices(string.ascii_letters + string.digits, k=9))
                        # Handle different argument structures for different tools
                        if isinstance(call.get("arguments"), dict):
                            arguments_str = json.dumps(call["arguments"])
                        elif isinstance(call.get("arguments"), str):
                            try:
                                # Try to parse as JSON first
                                parsed_args = json.loads(call["arguments"])
                                arguments_str = call["arguments"]
                            except json.JSONDecodeError:
                                # If it's a string, create appropriate JSON based on tool
                                tool_name = call.get("name", "")
                                if tool_name in ["search_documents", "search_wikipedia"]:
                                    arguments_str = json.dumps({"query": call["arguments"]})
                                elif tool_name == "retrieve_document_by_name":
                                    arguments_str = json.dumps({"document_name": call["arguments"]})
                                else:
                                    arguments_str = json.dumps({"query": str(call.get("arguments", ""))})
                        else:
                            arguments_str = "{}"

                        formatted_tool_calls.append({
                            "id": call_id,
                            "function": {
                                "name": call["name"],
                                "arguments": arguments_str
                            },
                            "type": "function"
                        })

                    # Add to conversation
                    conversation.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": formatted_tool_calls
                    })

                    # Execute tools sequentially
                    tool_results = []
                    for i, tool_call in enumerate(formatted_tool_calls):
                        tool_name = tool_call["function"]["name"]
                        args_str = tool_call["function"]["arguments"]
                        try:
                            tool_args = json.loads(args_str)
                        except json.JSONDecodeError as e:
                            tool_result = f"Tool call error: Invalid arguments format - {str(e)}"
                        else:
                            print(f"Executing tool {i+1}/{len(formatted_tool_calls)}: {tool_name} with args: {tool_args}")
                            # Execute the appropriate tool with proper argument extraction
                            if tool_name == "search_documents":
                                query_str = tool_args.get("query", "")
                                tool_result = retriever.search_documents_tool(query_str)
                            elif tool_name == "retrieve_document_by_name":
                                doc_name = tool_args.get("document_name", "")
                                tool_result = retriever.retrieve_document_by_name_tool(doc_name)
                            elif tool_name == "list_available_documents":
                                filter_keyword = tool_args.get("filter_keyword", "")
                                tool_result = retriever.list_available_documents_tool(filter_keyword)
                            elif tool_name == "search_wikipedia":
                                query_str = tool_args.get("query", "")
                                tool_result = Generator.search_wikipedia(query_str)
                            elif tool_name == "agentic_generator":
                                query_str = tool_args.get("query", "")
                                tool_result = AgenticGenerator.agentic_answer_query(query_str, llm_model, llm_tokenizer, retriever)
                            elif tool_name == "google_custom_search":
                                query_str = tool_args.get("query", "")
                                tool_result = Generator.google_custom_search(query_str)
                            else:
                                tool_result = f"Error: Unknown tool: {tool_name}"

                        # Convert result to string if it's not already
                        if not isinstance(tool_result, str):
                            tool_result = json.dumps(tool_result, ensure_ascii=False)

                        #print(tool_result) # mostrar resultados da tool use antes de resposta

                        # Add tool result to conversation
                        conversation.append({
                            "role": "tool",
                            "name": tool_name,
                            "content": tool_result,
                            "tool_call_id": tool_call["id"]
                        })
                        tool_results.append(tool_result)

                    print("All tool executions completed. Preparing final response...")
                    # Continue to next iteration to generate final response
                    continue
                except Exception as e:
                    print(f"Error processing tool calls: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    break
            else:
                # No tool calls needed, return the response
                print("No tool calls detected, returning response")
                # FIXED: Return the actual response text and prompt
                return response_text, prompt
        # If we reach maximum iterations, return the current response
        print(f"Fatal Error")
        return current_response, prompt


# === // ===


class Generator:
    @staticmethod
    def summarize_passages(passages, llm_model, llm_tokenizer): # LLM summarizes retrieved info/context.
        context = "\n".join(passages)
        prompt = (
            "Summarize the following retrieved passages.\n"
            f"{context}\n"
            "Summary:"
        )
        summary = generate(llm_model, llm_tokenizer, prompt=prompt, max_tokens=700, verbose=False)
        return summary
    
    @staticmethod
    def search_wikipedia(query: str) -> str:
            """Fetch Wikipedia results with structured data for RAG"""
            try:
                import urllib.parse
                import urllib.request
                import json
                # Search for pages
                encoded_query = urllib.parse.quote(query)
                search_url = (
                    f"https://en.wikipedia.org/w/api.php?action=query&list=search"
                    f"&srsearch={encoded_query}&format=json&srlimit=1&srnamespace=0"
                )
                headers = {'User-Agent': 'YourApp/1.0 (contact@example.com)'}
                search_req = urllib.request.Request(search_url, headers=headers)
                with urllib.request.urlopen(search_req, timeout=10) as response:
                    search_data = json.loads(response.read().decode('utf-8'))

                if not search_data.get('query', {}).get('search'):
                    return {'error': 'No results found'}

                # Get page content
                page_ids = [str(item['pageid']) for item in search_data['query']['search']]
                content_url = (
                    f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts|info"
                    f"&pageids={'|'.join(page_ids)}&format=json&explaintext=true"
                    f"&inprop=url"
                )
                content_req = urllib.request.Request(content_url, headers=headers)
                with urllib.request.urlopen(content_req, timeout=10) as response:
                    content_data = json.loads(response.read().decode('utf-8'))

                # Structure the results for better LLM understanding
                articles = []
                for page_id in page_ids:
                    page_info = content_data['query']['pages'].get(page_id)
                    if page_info and 'extract' in page_info:
                        content = page_info['extract'].strip()
                        words = content.split()
                        if len(words) > 1000:
                            content = ' '.join(words[:1000]) + '... (content truncated to 1000 words)'
                            wordcount = 1000
                        else:
                            wordcount = len(words)
                        articles.append({
                            'title': page_info['title'],
                            'content': content,
                            'pageid': page_info['pageid'],
                            'url': page_info.get('fullurl', f"https://en.wikipedia.org/?curid={page_id}"),
                            'wordcount': wordcount
                        })

                return {
                    'query': query,
                    'articles': articles,
                    'total_results': len(articles)
                }
            except Exception as e:
                return {'error': str(e)}

    @staticmethod
    def google_custom_search(query: str) -> str:
        """Fetch Google Custom Search results"""
        try:
            import urllib.parse
            import urllib.request
            import json

            if GOOGLE_CX == "YOUR_GOOGLE_CX_HERE":
                return "Error: Google Custom Search Engine ID (CX) is not configured. Please set GOOGLE_CX in the code."

            encoded_query = urllib.parse.quote(query)
            search_url = (
                f"https://www.googleapis.com/customsearch/v1?"
                f"key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&q={encoded_query}"
            )
            
            with urllib.request.urlopen(search_url, timeout=10) as response:
                search_data = json.loads(response.read().decode('utf-8'))

            if 'items' not in search_data:
                return "No results found."

            results = []
            for item in search_data['items'][:5]: # Limit to top 5 results
                title = item.get('title', 'No title')
                snippet = item.get('snippet', 'No snippet')
                link = item.get('link', 'No link')
                results.append(f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n")

            return "\n---\n".join(results)
        except Exception as e:
            return f"Error performing Google Search: {str(e)}"

    @staticmethod
    def answer_query_with_llm(query, llm_model, llm_tokenizer, retriever, prompt_cache=None):
        if ADVANCED_REASONING:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "agentic_generator",
                        "description": "Call advanced task agent if user is asking for a difficult task.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query string."
                                }
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_documents",
                        "description": "Search for documents relevant to the user's query. Use for general questions or when you need to find information across all documents.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query string."
                                }
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "retrieve_document_by_name",
                        "description": "Retrieve an entire document by its filename. Use when user specifically asks for a particular book, report, or document by name.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "document_name": {
                                    "type": "string",
                                    "description": "The name of the document to retrieve (e.g., 'annual_report.pdf')"
                                }
                            },
                            "required": ["document_name"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "list_available_documents",
                        "description": "List available documents in the system. Use when user asks what documents are available. Can filter by keyword - if user asks about specific topics (e.g., 'documents about allometric'), use the filter_keyword parameter to search document names.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filter_keyword": {
                                    "type": "string",
                                    "description": "Optional keyword to filter document names (e.g., 'allometric', 'climate', 'policy'). Leave empty to list all documents."
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_wikipedia",
                        "description": "Search Wikipedia for factual information and general knowledge.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query string for Wikipedia."}
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "google_custom_search",
                        "description": "Search the internet using Google Custom Search JSON API. Use for current events or information not found in documents.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query string."}
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]
        else:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_documents",
                        "description": "Search for documents relevant to the user's query. Use for general questions or when you need to find information across all documents.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query string."
                                }
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "retrieve_document_by_name",
                        "description": "Retrieve an entire document by its filename. Use when user specifically asks for a particular book, report, or document by name.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "document_name": {
                                    "type": "string",
                                    "description": "The name of the document to retrieve (e.g., 'annual_report.pdf')"
                                }
                            },
                            "required": ["document_name"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "list_available_documents",
                        "description": "List available documents in the system. Use when user asks what documents are available. Can filter by keyword - if user asks about specific topics (e.g., 'documents about allometric'), use the filter_keyword parameter to search document names.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "filter_keyword": {
                                    "type": "string",
                                    "description": "Optional keyword to filter document names (e.g., 'allometric', 'climate', 'policy'). Leave empty to list all documents."
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "search_wikipedia",
                        "description": "Search Wikipedia for factual information and general knowledge.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query string for Wikipedia."}
                            },
                            "required": ["query"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "google_custom_search",
                        "description": "Search the internet using Google Custom Search JSON API. Use for current events or information not found in documents.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query string."}
                            },
                            "required": ["query"]
                        }
                    }
                }
            ]

        # Temperature and top sampling
        sampler = make_sampler(temp=0.7, top_k=50, top_p=0.9)
        logits_processors = make_logits_processors(repetition_penalty=1.1, repetition_context_size=128)
        current_response = None

        # Enhanced system prompt to guide tool selection
        conversation = [
            {"role": "system", "content": "You are a helpful assistant with document search and Wikipedia search capabilities. "
            "Decide whether you need tools. If you use tools, answer based *only* on the provided results. "
            "If you're not sure if the user wants you to access tools, ask the user. "
            "After receiving tool results, provide a final answer. "
            "If not enough information is found after tool calling, alert the user that there's no available information to answer the user. "
            "At the end of an informative response, ask if the user needs more information or wants to explore more a certain fact. "
            "**Do not make sequential tool calls**!"}
        ]

        # Add current query
        conversation.append({"role": "user", "content": query})
        
        while True:
            prompt = llm_tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tools=tools,
                tokenize=False
            )

            print("\nFORMATTED PROMPT:") 
            print(prompt)

            response = generate(
                model=llm_model,
                tokenizer=llm_tokenizer,
                prompt=prompt,
                max_tokens=MAX_RESPONSE_TOKENS,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=prompt_cache,
                verbose=True
            )

            response_text = response.strip()

            # tool call present?
            if "[TOOL_CALLS]" in response_text:
                print(f"\nModel requested to use tools. Processing tool calls...")
                try:
                    tool_calls = []
                    
                    # Check which format we're dealing with
                    if "[TOOL_CALLS][" in response_text:
                        # OLD FORMAT: [TOOL_CALLS][{"name": "...", "arguments": {...}}]
                        start_marker = "[TOOL_CALLS]["
                        start_idx = response_text.find(start_marker) + len(start_marker) - 1
                        bracket_count = 1
                        end_idx = start_idx + 1
                        while end_idx < len(response_text) and bracket_count > 0:
                            if response_text[end_idx] == '[':
                                bracket_count += 1
                            elif response_text[end_idx] == ']':
                                bracket_count -= 1
                            end_idx += 1
                        tool_json = response_text[start_idx:end_idx]
                        tool_calls = json.loads(tool_json)
                        
                    elif "[ARGS]" in response_text:
                        # NEW FORMAT: [TOOL_CALLS]tool_name[ARGS]{"key": "value"}
                        tool_section = response_text.split("[TOOL_CALLS]")[1]
                        tool_name, args_part = tool_section.split("[ARGS]", 1)
                        tool_name = tool_name.strip()
                        
                        # Extract JSON object
                        args_part = args_part.strip()
                        brace_count = 0
                        end_idx = 0
                        for i, char in enumerate(args_part):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i + 1
                                    break
                        
                        args_json = args_part[:end_idx] if end_idx > 0 else "{}"
                        tool_args = json.loads(args_json) if args_json else {}
                        tool_calls = [{"name": tool_name, "arguments": tool_args}]
                    
                    else:
                        print("Unknown tool call format")
                        break

                    if not tool_calls:
                        print("No valid tool calls found")
                        break

                    print(f"Found {len(tool_calls)} tool call(s): {[call.get('name', 'unknown') for call in tool_calls]}")

                    # Add tool call to conversation
                    formatted_tool_calls = []
                    for call in tool_calls:
                        call_id = ''.join(random.choices(string.ascii_letters + string.digits, k=9))
                        # Handle different argument structures for different tools
                        if isinstance(call.get("arguments"), dict):
                            arguments_str = json.dumps(call["arguments"])
                        elif isinstance(call.get("arguments"), str):
                            try:
                                # Try to parse as JSON first
                                parsed_args = json.loads(call["arguments"])
                                arguments_str = call["arguments"]
                            except json.JSONDecodeError:
                                # If it's a string, create appropriate JSON based on tool
                                tool_name = call.get("name", "")
                                if tool_name in ["search_documents", "search_wikipedia"]:
                                    arguments_str = json.dumps({"query": call["arguments"]})
                                elif tool_name == "retrieve_document_by_name":
                                    arguments_str = json.dumps({"document_name": call["arguments"]})
                                else:
                                    arguments_str = json.dumps({"query": str(call.get("arguments", ""))})
                        else:
                            arguments_str = "{}"

                        formatted_tool_calls.append({
                            "id": call_id,
                            "function": {
                                "name": call["name"],
                                "arguments": arguments_str
                            },
                            "type": "function"
                        })

                    # Add to conversation
                    conversation.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": formatted_tool_calls
                    })

                    # Execute tools sequentially
                    tool_results = []
                    for i, tool_call in enumerate(formatted_tool_calls):
                        tool_name = tool_call["function"]["name"]
                        args_str = tool_call["function"]["arguments"]
                        try:
                            tool_args = json.loads(args_str)
                        except json.JSONDecodeError as e:
                            tool_result = f"Tool call error: Invalid arguments format - {str(e)}"
                        else:
                            print(f"Executing tool {i+1}/{len(formatted_tool_calls)}: {tool_name} with args: {tool_args}")
                            # Execute the appropriate tool with proper argument extraction
                            if tool_name == "search_documents":
                                query_str = tool_args.get("query", "")
                                tool_result = retriever.search_documents_tool(query_str)
                            elif tool_name == "retrieve_document_by_name":
                                doc_name = tool_args.get("document_name", "")
                                tool_result = retriever.retrieve_document_by_name_tool(doc_name)
                            elif tool_name == "list_available_documents":
                                filter_keyword = tool_args.get("filter_keyword", "")
                                tool_result = retriever.list_available_documents_tool(filter_keyword)
                            elif tool_name == "search_wikipedia":
                                query_str = tool_args.get("query", "")
                                tool_result = Generator.search_wikipedia(query_str)
                            elif tool_name == "agentic_generator":
                                query_str = tool_args.get("query", "")
                                tool_result = AgenticGenerator.agentic_answer_query(query_str, llm_model, llm_tokenizer, retriever)
                            elif tool_name == "google_custom_search":
                                query_str = tool_args.get("query", "")
                                tool_result = Generator.google_custom_search(query_str)
                            else:
                                tool_result = f"Error: Unknown tool: {tool_name}"

                        # Convert result to string if it's not already
                        if not isinstance(tool_result, str):
                            tool_result = json.dumps(tool_result, ensure_ascii=False)

                        #print(tool_result) # mostrar resultados da tool use antes de resposta

                        # Add tool result to conversation
                        conversation.append({
                            "role": "tool",
                            "name": tool_name,
                            "content": tool_result,
                            "tool_call_id": tool_call["id"]
                        })
                        tool_results.append(tool_result)

                    print("All tool executions completed. Preparing final response...")
                    # Continue to next iteration to generate final response
                    continue
                except Exception as e:
                    print(f"Error processing tool calls: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    break
            else:
                # No tool calls needed, return the response
                print("No tool calls detected, returning response")
                # FIXED: Return the actual response text and prompt
                return response_text, prompt

        # If we reach maximum iterations, return the current response
        print(f"Fatal Error")
        return current_response, prompt

    @staticmethod
    def eval(query, llm_model, llm_tokenizer, rag_response):
        system_prompt = f"""
            You are a RAG evaluator judge. Please evaluate the following response based on these criteria with a 0-10 scale:
            1. Context Relevance: How well retrieved documents align with the user's query and are able to address it.
                Example: Question: when was the biggest earthquake in Lisbon? Answer: The biggest earthquake in Lisbon was in 1755. Retrieved chunks: The biggest earthquake in Lisbon was in 1755, which killed more than 30000 people. Score: 10; Question: when was the biggest earthquake in Lisbon? Answer: The biggest earthquake in Lisbon was in 1755. Retrieved chunks: The white house is where the president lives. Score: 0; Question: when was the biggest earthquake in Lisbon? Answer: The biggest earthquake in Lisbon was in 1755. Retrieved chunks: The earthquake. Lisbon. Earthquake in Lisbon. The earthquake in Lisbon. The earthquake in Lisbon. The biggest earthquake in Lisbon Score: 3;
            2. Groundedness: How accurately the response is based on the retrieved context.
            3. Answer Relevance: How well the response addresses the original query.
            4. Faithfulness: Is the output contradicting the retrieved facts?
            5. Contextual Recall: Did we retrieve ALL the info needed?
            6. Contextual Relevancy: What % of retrieved chunks actually matter?
            Be precise and objective on your scores! Be very critic.
        """
        user_prompt = f"""
User Query: {query}
            Response and context: {rag_response}
        """
        eval_conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
            ]
        sampler = make_sampler(temp=0.2, top_k=30, top_p=0.6)
        prompt = llm_tokenizer.apply_chat_template(
            eval_conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        eval_response = generate(
            model=llm_model,
            tokenizer=llm_tokenizer,
            prompt=prompt,
            sampler=sampler,
            verbose=False
        )
        print("Evaluation Results:\n", eval_response)


# ==== Main Runtime ====
if __name__ == "__main__":
    print("Attempting to load saved FAISS index and BM25 index...")
    indexer = Indexer()
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
        app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False, use_reloader=False)
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