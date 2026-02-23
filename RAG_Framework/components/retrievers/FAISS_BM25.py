import math
import os
import re
import threading
import numpy as np
import faiss
from collections import defaultdict
from mlx_lm import generate

from RAG_Framework.core.config import (
    BM25_ENABLE_STEMMING, BM25_ENABLE_STOPWORDS, BM25_LANGUAGES,
    EMBEDDING_USE_PREFIX, RETRIEVAL_TOP_K, RERANKER_TOP_N, PARENT_TOP_N,
    RERANKER_CONFIDENCE_THRESHOLD,
)
from RAG_Framework.core.text_processing import tokenize_for_bm25, prepare_for_embedding


class Retriever:
    def __init__(self, llm_model, llm_tokenizer,
                 embedding_model_name=None, reranker_model_name=None,
                 index_paths=None):
        # LLM needed immediately for query expansion
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer

        # Store config for lazy loading
        self._embedding_model_name = embedding_model_name
        self._reranker_model_name = reranker_model_name
        self._index_paths = index_paths

        # Private backing fields (None until loaded)
        self._embedding_model = None
        self._reranker = None
        self._multi_vector_index = None
        self._bm25 = None
        self._metadata_index = None
        self._faiss_index = None
        self._parent_store = None
        self._indices_loaded = False

        # Thread safety for lazy loading
        self._load_lock = threading.Lock()

        self.last_retrieved_metadata = []

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            with self._load_lock:
                if self._embedding_model is None:
                    print(f"Loading embedding model: {self._embedding_model_name}...")
                    from sentence_transformers import SentenceTransformer
                    self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    @property
    def reranker(self):
        if self._reranker is None:
            with self._load_lock:
                if self._reranker is None:
                    print(f"Loading reranker: {self._reranker_model_name}...")
                    from sentence_transformers import CrossEncoder
                    self._reranker = CrossEncoder(self._reranker_model_name)
        return self._reranker

    def _ensure_indices_loaded(self):
        if not self._indices_loaded:
            with self._load_lock:
                if not self._indices_loaded:
                    print("Loading indices...")
                    from RAG_Framework.core.config import HIERARCHICAL_INDEXING
                    if HIERARCHICAL_INDEXING:
                        from RAG_Framework.components.indexer import HierarchicalIndexer
                        indexer = HierarchicalIndexer()
                        self._multi_vector_index, self._bm25, self._metadata_index, self._faiss_index, self._parent_store = indexer.load_indices()
                    else:
                        from RAG_Framework.components.indexer import Indexer
                        indexer = Indexer()
                        self._multi_vector_index, self._bm25, self._metadata_index, self._faiss_index = indexer.load_indices()
                    self._indices_loaded = True

    @property
    def multi_vector_index(self):
        self._ensure_indices_loaded()
        return self._multi_vector_index

    @property
    def faiss_index(self):
        self._ensure_indices_loaded()
        return self._faiss_index

    @property
    def bm25(self):
        self._ensure_indices_loaded()
        return self._bm25

    @property
    def metadata_index(self):
        self._ensure_indices_loaded()
        return self._metadata_index

    @property
    def parent_store(self):
        self._ensure_indices_loaded()
        return self._parent_store

    def update_indices(self, multi_vector_index, bm25, metadata_index, faiss_index,
                       parent_store=None):
        """Thread-safe swap of all in-memory indices."""
        with self._load_lock:
            self._multi_vector_index = multi_vector_index
            self._bm25 = bm25
            self._metadata_index = metadata_index
            self._faiss_index = faiss_index
            if parent_store is not None:
                self._parent_store = parent_store
            self._indices_loaded = True

    def _resolve_parents(self, reranked_pairs, top_n=None):
        """
        Resolve child chunks to their parent chunks, deduplicating by parent.

        Args:
            reranked_pairs: List of (child_text, child_metadata) tuples
            top_n: Max number of results to return (applied after dedup)

        Returns:
            List of (text, metadata) tuples — parent chunks if available,
            otherwise children as-is.
        """
        if self._parent_store is None:
            return reranked_pairs

        resolved = []
        seen_parent_ids = set()

        for child_text, child_meta in reranked_pairs:
            parent_idx = child_meta.get('parent_idx')
            if parent_idx is None:
                # No parent mapping — return child as-is
                resolved.append((child_text, child_meta))
                continue

            if parent_idx in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_idx)

            if parent_idx < len(self._parent_store):
                parent = self._parent_store[parent_idx]
                resolved.append((parent["text"], parent["metadata"]))
            else:
                # Fallback if parent_idx is out of range
                resolved.append((child_text, child_meta))

        if top_n:
            resolved = resolved[:top_n]

        return resolved

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
                x['metadata'].get('slide', 0),
                x['metadata'].get('sheet_index', 0),
                x['metadata'].get('chunk_idx', 0),
                x['metadata'].get('table_idx', 0),
                x['metadata'].get('img_idx', 0)
            ))
        return results

    def expand_query(self, query):
        system_prompt = (
            "You are a query expander for document retrieval.\n\n"
            "OUTPUT FORMAT (exactly 3 lines):\n"
            "1. [First rephrasing - use synonyms]\n"
            "2. [Second rephrasing - broader/narrower scope]\n"
            "3. Keywords: term1, term2, term3, term4, term5\n\n"
            "GUIDELINES:\n"
            "- Include technical terms and common alternatives\n"
            "- Extract 3-5 core searchable keywords\n"
            "- No explanations, just the output"
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

    def rerank_chunks_with_metadata(self, query, chunks, metadata_list, top_n=None, threshold=None):
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.reranker.predict(pairs)
        combined = list(zip(chunks, metadata_list, scores))
        reranked = sorted(combined, key=lambda x: x[2], reverse=True)
        if top_n:
            reranked = reranked[:top_n]
        for chunk, meta, score in reranked:
            print(f"  score={1.0 / (1.0 + math.exp(-float(score))):.3f}  {chunk[:60]}", flush=True)
        if threshold is not None:
            # Convert raw reranker logits to sigmoid probabilities for thresholding
            filtered = [
                (chunk, meta, score) for chunk, meta, score in reranked
                if 1.0 / (1.0 + math.exp(-float(score))) >= threshold
            ]
            if not filtered and reranked:
                # Always keep at least the best result to avoid empty context
                best = reranked[0]
                best_prob = 1.0 / (1.0 + math.exp(-float(best[2])))
                print(
                    f"Warning: All {len(reranked)} chunks below confidence threshold "
                    f"{threshold:.2f} (best: {best_prob:.3f}). Keeping top result."
                )
                filtered = [best]
            reranked = filtered
        return [(chunk, meta) for chunk, meta, score in reranked]

    def combined_retrieval(
            self,
            query,
            k=RETRIEVAL_TOP_K,
            weight_dense=0.6,
            weight_sparse=0.4,
            rerank_top_n=RERANKER_TOP_N,
            use_summarization=False,
    ):
        if not self.faiss_index or not self.bm25 or not self.multi_vector_index:
             print("Warning: FAISS index, BM25 index, or multi-vector index is not available for retrieval.")
             return "No documents available for retrieval."

        # Expand query variants for retrieval
        expanded_queries = [query] + self.expand_query(query)

        # Dense retrieval scoring with E5 query prefixes
        dense_scores = defaultdict(float)
        # Add query prefix for E5 models if enabled
        prefixed_queries = [
            prepare_for_embedding(q, is_query=True, use_prefix=EMBEDDING_USE_PREFIX)
            for q in expanded_queries
        ]
        query_embeddings = self.embedding_model.encode(prefixed_queries)
        for query_emb in query_embeddings:
            faiss_results = self.retrieve_with_faiss(query_emb, self.faiss_index, k=k)
            for idx, score in faiss_results:
                dense_scores[idx] += score
        dense_scores_norm = self.normalize_scores(dense_scores)

        # Sparse retrieval scoring (BM25) with improved tokenization
        sparse_scores = defaultdict(float)
        for exp_query in expanded_queries:
            # Use improved tokenization with stemming and stop word removal
            tokenized_query = tokenize_for_bm25(
                exp_query,
                enable_stemming=BM25_ENABLE_STEMMING,
                enable_stopwords=BM25_ENABLE_STOPWORDS,
                languages=BM25_LANGUAGES
            )
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
        reranked = self.rerank_chunks_with_metadata(
            query, retrieved_chunks, retrieved_metadata,
            top_n=rerank_top_n, threshold=RERANKER_CONFIDENCE_THRESHOLD,
        )

        # Resolve to parent chunks if hierarchical mode is active
        resolved = self._resolve_parents(reranked, top_n=PARENT_TOP_N)

        # Store metadata for citation tracking
        self.last_retrieved_metadata = [meta for _, meta in resolved]

        if use_summarization:
            from RAG_Framework.components.generators import Generator
            resolved_texts = [chunk for chunk, _ in resolved]
            final_context = Generator.summarize_passages(resolved_texts, self.llm_model, self.llm_tokenizer)
        else:
            final_context = "\n---\n".join([chunk for chunk, _ in resolved])
        return final_context

    def combined_retrieval_with_chunks(
            self,
            query,
            k=RETRIEVAL_TOP_K,
            weight_dense=0.6,
            weight_sparse=0.4,
            rerank_top_n=RERANKER_TOP_N,
    ):
        """Same as combined_retrieval but returns (context_string, individual_chunks_list)."""
        if not self.faiss_index or not self.bm25 or not self.multi_vector_index:
            return "No documents available for retrieval.", []

        expanded_queries = [query] + self.expand_query(query)

        dense_scores = defaultdict(float)
        prefixed_queries = [
            prepare_for_embedding(q, is_query=True, use_prefix=EMBEDDING_USE_PREFIX)
            for q in expanded_queries
        ]
        query_embeddings = self.embedding_model.encode(prefixed_queries)
        for query_emb in query_embeddings:
            faiss_results = self.retrieve_with_faiss(query_emb, self.faiss_index, k=k)
            for idx, score in faiss_results:
                dense_scores[idx] += score
        dense_scores_norm = self.normalize_scores(dense_scores)

        sparse_scores = defaultdict(float)
        for exp_query in expanded_queries:
            tokenized_query = tokenize_for_bm25(
                exp_query,
                enable_stemming=BM25_ENABLE_STEMMING,
                enable_stopwords=BM25_ENABLE_STOPWORDS,
                languages=BM25_LANGUAGES
            )
            bm25_results = self.retrieve_with_bm25(tokenized_query, self.bm25, top_k=k)
            for idx, score in bm25_results:
                sparse_scores[idx] += score
        sparse_scores_norm = self.normalize_scores(sparse_scores)

        all_doc_indices = set(dense_scores_norm.keys()) | set(sparse_scores_norm.keys())
        combined_scores = {}
        for idx in all_doc_indices:
            dscore = dense_scores_norm.get(idx, 0)
            sscore = sparse_scores_norm.get(idx, 0)
            combined_scores[idx] = weight_dense * dscore + weight_sparse * sscore

        top_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)[:k]
        retrieved_chunks = [self.multi_vector_index[idx]["text"] for idx in top_indices]
        retrieved_metadata = [self.multi_vector_index[idx]["metadata"] for idx in top_indices]

        reranked = self.rerank_chunks_with_metadata(
            query, retrieved_chunks, retrieved_metadata,
            top_n=rerank_top_n, threshold=RERANKER_CONFIDENCE_THRESHOLD,
        )

        # Resolve to parent chunks if hierarchical mode is active
        resolved = self._resolve_parents(reranked, top_n=PARENT_TOP_N)

        self.last_retrieved_metadata = [meta for _, meta in resolved]

        chunks_list = [chunk for chunk, _ in resolved]
        context_string = "\n---\n".join(chunks_list)
        return context_string, chunks_list

    def search_documents_tool(self, query: str, weight_dense: float = 0.6, weight_sparse: float = 0.4) -> str:
        """Tool for semantic search across all documents.

        Args:
            query: Search query string.
            weight_dense: Weight for dense (FAISS) retrieval (default 0.6).
            weight_sparse: Weight for sparse (BM25) retrieval (default 0.4).
        """
        try:
            return self.combined_retrieval(
                query=query,
                weight_dense=weight_dense,
                weight_sparse=weight_sparse,
                use_summarization=False
            )
        except Exception as e:
            return f"Error during search: {str(e)}"

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

            # Determine document format for citation style
            doc_ext = os.path.splitext(document_name)[1].lower() if '.' in document_name else ''

            # Combine all chunks with their metadata for context
            document_content = []
            for result in results:
                chunk_text = result['text']
                metadata = result['metadata']
                # Format-aware citation
                if doc_ext == '.pptx':
                    citation = f"[Slide {metadata.get('slide', metadata.get('page', 0) + 1)}, Chunk {metadata.get('chunk_idx', metadata.get('table_idx', metadata.get('img_idx', 'N/A')))}]"
                elif doc_ext in ('.xlsx', '.xls'):
                    sheet = metadata.get('sheet', 'Sheet1')
                    citation = f"[Sheet '{sheet}', Chunk {metadata.get('chunk_idx', 'N/A')}]"
                elif doc_ext == '.csv':
                    citation = f"[Chunk {metadata.get('chunk_idx', 'N/A')}]"
                elif doc_ext == '.pdf':
                    citation = f"[Page {metadata.get('page', 'N/A')}, Chunk {metadata.get('chunk_idx', metadata.get('table_idx', metadata.get('img_idx', 'N/A')))}]"
                else:
                    citation = f"[Chunk {metadata.get('chunk_idx', metadata.get('img_idx', 'N/A'))}]"
                document_content.append(f"{chunk_text}\n{citation}")

            full_document = "\n--- Page Break ---\n".join(document_content)

            # Add document summary header
            doc_metadata = results[0]['metadata']
            header = f"DOCUMENT: {doc_metadata.get('document_name', document_name)}\n"
            header += f"TOTAL CHUNKS: {len(results)}\n"
            if doc_ext == '.pptx':
                header += f"SLIDES: {max(r['metadata'].get('slide', r['metadata'].get('page', 0) + 1) for r in results)}\n"
            elif doc_ext in ('.xlsx', '.xls'):
                sheets = set(r['metadata'].get('sheet', '') for r in results)
                header += f"SHEETS: {', '.join(sorted(s for s in sheets if s))}\n"
            else:
                header += f"PAGES: {max(r['metadata'].get('page', 0) for r in results) + 1}\n"
            header += "="*50 + "\n"

            return header + full_document
        except Exception as e:
            import traceback
            return f"Error retrieving document '{document_name}': {str(e)}\n{traceback.format_exc()}"

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
