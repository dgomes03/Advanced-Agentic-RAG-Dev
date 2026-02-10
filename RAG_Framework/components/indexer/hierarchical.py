import os

from RAG_Framework.core.config import (
    PARENT_CHUNK_SIZE, CHUNK_SIZE, CHUNK_OVERLAP, PARENT_STORE_PATH
)
from RAG_Framework.core.utils import save_pickle, load_pickle
from RAG_Framework.components.indexer.standard import Indexer


class HierarchicalIndexer(Indexer):
    """
    Hierarchical chunking indexer: indexes small child chunks (identical to
    flat mode) for precise retrieval, but groups them into larger parent
    chunks for richer LLM context.

    Key design: children are chunked at CHUNK_SIZE exactly like standard mode,
    then grouped into parents. This guarantees identical retrieval quality
    while providing more context to the LLM.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parent_store_path = PARENT_STORE_PATH

    @staticmethod
    def build_hierarchical_chunks(child_chunks, child_metadata,
                                  parent_chunk_size=PARENT_CHUNK_SIZE,
                                  parent_idx_offset=0):
        """
        Group consecutive same-page child chunks into larger parent chunks.

        Children are kept as-is (identical to flat mode). Parents are formed
        by joining consecutive children from the same page/document until
        parent_chunk_size is exceeded.

        Mutates child_metadata in-place to add parent_idx and child_idx.

        Args:
            child_chunks: List of child-sized text chunks (from flat chunking)
            child_metadata: List of metadata dicts for each child
            parent_chunk_size: Max character size for parent chunks
            parent_idx_offset: Offset for parent_idx (for incremental indexing)

        Returns:
            (child_chunks, child_metadata, parent_store)
        """
        if not child_chunks:
            return [], [], []

        parent_store = []
        group_start = 0
        group_length = 0
        prev_page_key = None

        def flush_group(start, end):
            parent_text = " ".join(child_chunks[start:end])
            parent_meta = child_metadata[start].copy()
            parent_meta.pop('parent_idx', None)
            parent_meta.pop('child_idx', None)
            parent_store.append({"text": parent_text, "metadata": parent_meta})
            parent_idx = parent_idx_offset + len(parent_store) - 1
            for pos, idx in enumerate(range(start, end)):
                child_metadata[idx]['parent_idx'] = parent_idx
                child_metadata[idx]['child_idx'] = pos

        for i, (c_text, c_meta) in enumerate(zip(child_chunks, child_metadata)):
            page_key = (c_meta.get('filename'), c_meta.get('page', 0))

            # Start new parent group on page change or size overflow
            if i > group_start and (
                page_key != prev_page_key or
                group_length + len(c_text) > parent_chunk_size
            ):
                flush_group(group_start, i)
                group_start = i
                group_length = 0

            prev_page_key = page_key
            group_length += len(c_text)

        # Flush last group
        if group_start < len(child_chunks):
            flush_group(group_start, len(child_chunks))

        return child_chunks, child_metadata, parent_store

    def save_parent_store(self, parent_store):
        """Save parent store to disk."""
        save_pickle(parent_store, self.parent_store_path)

    def load_parent_store(self):
        """Load parent store from disk. Returns None if file missing."""
        return load_pickle(self.parent_store_path)

    def load_and_chunk_documents_parallel(self, max_chunk_length=None, chunk_overlap=None):
        """
        Load documents, chunk at child size (identical to flat mode),
        then group children into parents.

        Returns:
            (child_chunks, child_metadata, parent_store) — 3-tuple
        """
        # Get flat chunks identical to standard mode (CHUNK_SIZE)
        child_chunks, child_metadata = super().load_and_chunk_documents_parallel()

        if not child_chunks:
            return [], [], []

        # Group flat chunks into parents
        child_chunks, child_metadata, parent_store = self.build_hierarchical_chunks(
            child_chunks, child_metadata
        )
        print(f"Grouped {len(child_chunks)} child chunks into {len(parent_store)} parents")

        return child_chunks, child_metadata, parent_store

    def save_indices(self, multi_vector_index, bm25, metadata_index, faiss_index=None,
                     parent_store=None):
        """Save all indices plus parent store."""
        super().save_indices(multi_vector_index, bm25, metadata_index, faiss_index)
        if parent_store is not None:
            self.save_parent_store(parent_store)

    def load_indices(self):
        """Load all indices plus parent store.

        Returns:
            (multi_vector_index, bm25, metadata_index, faiss_index, parent_store) — 5-tuple
        """
        mvi, bm25, metadata_index, faiss_index = super().load_indices()
        parent_store = self.load_parent_store()
        return mvi, bm25, metadata_index, faiss_index, parent_store

    def index_new_documents(self, multi_vector_index, bm25, metadata_index,
                            faiss_index, embedding_model, parent_store=None):
        """
        Detect and index new documents, handling hierarchical parent/child chunks.

        Returns:
            (mvi, bm25, metadata_index, faiss_index, new_filenames, parent_store)
            or None if no new documents found.
        """
        from RAG_Framework.components.indexer.processors import (
            process_single_document, SUPPORTED_EXTENSIONS as _SE
        )
        from multiprocessing import Pool, cpu_count
        from tqdm import tqdm
        import numpy as np
        import faiss as faiss_lib

        # List all supported files
        all_files = [
            f for f in os.listdir(self.documents_dir)
            if os.path.splitext(f)[1].lower() in _SE
        ]

        # Determine which are already indexed
        indexed_filenames = set()
        for key in metadata_index:
            if os.path.splitext(key)[1]:
                indexed_filenames.add(key)

        new_filenames = [f for f in all_files if f not in indexed_filenames]

        if not new_filenames:
            return None

        print(f"New documents detected: {new_filenames}")

        # Build file_infos with CHUNK_SIZE (flat chunking, identical to standard)
        existing_doc_count = len(indexed_filenames)
        file_infos = [
            (
                os.path.join(self.documents_dir, f),
                existing_doc_count + i,
                CHUNK_SIZE,
                CHUNK_OVERLAP,
                self.enable_ocr,
                self.ocr_min_width,
                self.ocr_min_height,
                self.ocr_resolution
            )
            for i, f in enumerate(new_filenames)
        ]

        # Process new documents in parallel
        num_processes = max(1, min(cpu_count() - 1, 8))
        print(f"Processing {len(new_filenames)} new document(s) with {num_processes} processes...")

        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_single_document, file_infos),
                total=len(file_infos),
                desc="Processing new documents"
            ))

        new_chunks = []
        new_metadata = []
        for chunks, metadata in results:
            new_chunks.extend(chunks)
            new_metadata.extend(metadata)

        if not new_chunks:
            print("No content extracted from new documents.")
            return None

        # Group flat chunks into parents with offset
        existing_parent_count = len(parent_store) if parent_store else 0
        new_chunks, new_metadata, new_parent_store = self.build_hierarchical_chunks(
            new_chunks, new_metadata, parent_idx_offset=existing_parent_count
        )

        print(f"Extracted {len(new_chunks)} child chunks grouped into {len(new_parent_store)} parents")

        # Extend parent_store
        if parent_store is None:
            parent_store = []
        parent_store.extend(new_parent_store)

        # Generate embeddings for new child chunks
        print("Generating embeddings for new chunks...")
        new_embeddings = self.get_embeddings(new_chunks, model=embedding_model)

        # Calculate offset for new chunk indices
        offset = len(multi_vector_index)

        # Extend multi_vector_index
        new_entries = self.build_multi_vector_index(new_embeddings, new_metadata, new_chunks)
        multi_vector_index.extend(new_entries)

        # Add new vectors to FAISS index
        embeddings_array = np.array([item["embedding"] for item in new_entries]).astype('float32')
        faiss_lib.normalize_L2(embeddings_array)
        faiss_index.add(embeddings_array)

        # Rebuild BM25 from all texts
        all_texts = [item["text"] for item in multi_vector_index]
        bm25 = self.build_bm25_index(all_texts)

        # Update metadata_index with new entries
        for idx, meta in enumerate(new_metadata):
            doc_name = meta.get('document_name')
            full_doc_id = meta.get('full_document_id')
            global_idx = offset + idx
            if doc_name:
                if doc_name not in metadata_index:
                    metadata_index[doc_name] = []
                metadata_index[doc_name].append(global_idx)
            if full_doc_id:
                if full_doc_id not in metadata_index:
                    metadata_index[full_doc_id] = []
                metadata_index[full_doc_id].append(global_idx)

        # Save updated indices to disk
        self.save_indices(multi_vector_index, bm25, metadata_index, faiss_index,
                          parent_store=parent_store)
        print(f"Indices updated and saved. Total child chunks: {len(multi_vector_index)}")

        return multi_vector_index, bm25, metadata_index, faiss_index, new_filenames, parent_store
