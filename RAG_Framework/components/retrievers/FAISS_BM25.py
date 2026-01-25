import re
import numpy as np
import faiss
from collections import defaultdict
from mlx_lm import generate


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
            from RAG_Framework.components.generators import Generator
            reranked_texts = [chunk for chunk, _ in reranked]
            final_context = Generator.summarize_passages(reranked_texts, self.llm_model, self.llm_tokenizer)
        else:
            final_context = "\n---\n".join([chunk for chunk, _ in reranked])
        return final_context

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
