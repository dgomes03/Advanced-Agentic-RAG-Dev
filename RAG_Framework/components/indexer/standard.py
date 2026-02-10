import os
import re
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image
import faiss
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from RAG_Framework.core.config import (
    DOCUMENTS_DIR, EMBEDDING_MODEL_NAME,
    MULTIVECTOR_INDEX_PATH, BM25_DATA_PATH,
    METADATA_INDEX_PATH, FAISS_INDEX_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS,
    BM25_ENABLE_STEMMING, BM25_ENABLE_STOPWORDS, BM25_LANGUAGES,
    FAISS_INDEX_TYPE, FAISS_IVF_NPROBE_RATIO,
    EMBEDDING_USE_PREFIX
)
from RAG_Framework.core.utils import save_pickle, load_pickle
from RAG_Framework.core.text_processing import (
    tokenize_for_bm25, clean_text, prepare_for_embedding
)


class Indexer:
    def __init__(self, documents_dir=DOCUMENTS_DIR, embedding_model_name=EMBEDDING_MODEL_NAME,
                 faiss_index_path=MULTIVECTOR_INDEX_PATH, bm25_data_path=BM25_DATA_PATH,
                 metadata_index_path=METADATA_INDEX_PATH, faiss_binary_path=FAISS_INDEX_PATH,
                 enable_ocr=True, ocr_min_width=50, ocr_min_height=50, ocr_resolution=100):
        self.documents_dir = documents_dir
        self.embedding_model_name = embedding_model_name
        self.faiss_index_path = faiss_index_path
        self.bm25_data_path = bm25_data_path
        self.metadata_index_path = metadata_index_path
        self.faiss_binary_path = faiss_binary_path
        self.enable_ocr = enable_ocr
        self.ocr_min_width = ocr_min_width
        self.ocr_min_height = ocr_min_height
        self.ocr_resolution = ocr_resolution

    @staticmethod
    def chunk_text_with_overlap(text, max_chunk_length, chunk_overlap):
        """
        Split text into chunks with overlap to preserve boundary information.

        Args:
            text: Text to chunk
            max_chunk_length: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        overlap_sentences = []  # Sentences to carry over to next chunk

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds max length and we have content
            if current_length + sentence_len > max_chunk_length and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # Determine overlap: keep sentences from the end that fit within overlap size
                overlap_sentences = []
                overlap_length = 0
                for sent in reversed(current_chunk):
                    if overlap_length + len(sent) <= chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent) + 1
                    else:
                        break

                # Start new chunk with overlap sentences
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) + 1 for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_len + 1

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    @staticmethod
    def process_single_pdf(file_info):
        file_path, doc_id, max_chunk_length, chunk_overlap, enable_ocr, ocr_min_width, ocr_min_height, ocr_resolution = file_info
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
                        # Import text cleaning (lazy import to avoid circular deps in static method)
                        from RAG_Framework.core.text_processing import clean_text

                        # Clean and normalize text before chunking
                        text = clean_text(text)

                        # Use sentence-aware splitting with overlap
                        chunks = Indexer.chunk_text_with_overlap(text, max_chunk_length, chunk_overlap)

                        for chunk_idx, chunk in enumerate(chunks):
                            if chunk:  # Only add non-empty chunks
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

                    # OCR for images (if enabled)
                    if enable_ocr and page.images:
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

                                    # Calculate image dimensions
                                    img_width = img_x1 - img_x0
                                    img_height = img_bottom - img_top

                                    # Skip if image is too small (likely decorative or icon)
                                    if img_width < ocr_min_width or img_height < ocr_min_height:
                                        continue

                                    # Crop and convert to image with optimized resolution
                                    image = page.crop((img_x0, img_top, img_x1, img_bottom)).to_image(resolution=ocr_resolution)
                                    pil_img = image.original

                                    # Use Tesseract OCR (much faster than EasyOCR)
                                    ocr_text = pytesseract.image_to_string(pil_img, lang='eng+por').strip()

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
                                # Silently skip failed OCR to avoid cluttering output
                                pass

                    # Save elements
                    for elem in elements:
                        doc_chunks.append(elem['content'])
                        meta = elem['meta'].copy()
                        meta['type'] = elem['type']
                        doc_metadata.append(meta)
            print(f"Processed {filename}")
        except Exception as e:
            print(f"Could not read/process {filename}: {e}")
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

    def get_embeddings(self, texts, model, batch_size=32, is_query=False):
        """
        Generate embeddings for a list of texts in batches.

        Args:
            texts: List of texts to embed
            model: Sentence transformer model
            batch_size: Batch size for encoding
            is_query: If True, use "query: " prefix; if False, use "passage: " prefix

        Returns:
            numpy array of embeddings
        """
        # Apply E5 prefixes if enabled
        if EMBEDDING_USE_PREFIX:
            texts = [prepare_for_embedding(text, is_query=is_query) for text in texts]

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
            batch = texts[i:i+batch_size]
            embeddings = model.encode(batch)
            all_embeddings.extend(embeddings)
        return np.array(all_embeddings, dtype=np.float32)

    def build_faiss_index(self, multi_vector_index, index_type=None):
        """
        Build FAISS index with automatic selection based on corpus size.

        Args:
            multi_vector_index: List of dicts with 'embedding' key
            index_type: Override index type ('flat', 'ivf', 'hnsw', 'auto')
                       If None, uses FAISS_INDEX_TYPE from config

        Index selection:
            - < 10,000 vectors: IndexFlatIP (exact search)
            - 10,000 - 1,000,000: IndexIVFFlat (inverted file)
            - > 1,000,000: IndexHNSWFlat (graph-based)

        Returns:
            FAISS index
        """
        if not multi_vector_index:
            print("Warning: Multi-vector index is empty. Cannot build FAISS index.")
            return None

        if index_type is None:
            index_type = FAISS_INDEX_TYPE

        embeddings = np.array([item["embedding"] for item in multi_vector_index]).astype('float32')
        n_vectors = embeddings.shape[0]
        dimension = embeddings.shape[1]

        # Normalize embeddings for inner product similarity
        faiss.normalize_L2(embeddings)

        # Auto-select index type based on corpus size
        if index_type == 'auto':
            if n_vectors < 10000:
                index_type = 'flat'
            elif n_vectors < 1000000:
                index_type = 'ivf'
            else:
                index_type = 'hnsw'

        print(f"Building FAISS index: type={index_type}, vectors={n_vectors}, dimension={dimension}")

        if index_type == 'flat':
            # Exact search - O(n) but 100% recall
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)

        elif index_type == 'ivf':
            # IVF for medium scale - O(sqrt(n)) with ~95-99% recall
            # nlist = number of clusters (sqrt(n) is a good default)
            nlist = max(int(np.sqrt(n_vectors)), 1)
            nlist = min(nlist, n_vectors)  # Can't have more clusters than vectors

            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

            # Training required for IVF
            print(f"Training IVF index with nlist={nlist}...")
            index.train(embeddings)
            index.add(embeddings)

            # Set nprobe for search (tradeoff: higher = better recall, slower)
            index.nprobe = max(nlist // FAISS_IVF_NPROBE_RATIO, 1)
            print(f"IVF index built: nlist={nlist}, nprobe={index.nprobe}")

        elif index_type == 'hnsw':
            # HNSW for large scale - O(log n) with ~95% recall
            # M = number of neighbors per node (32 is a good default)
            M = 32
            index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 40  # Build-time accuracy
            index.hnsw.efSearch = 16  # Search-time accuracy (can adjust later)
            index.add(embeddings)
            print(f"HNSW index built: M={M}, efConstruction=40, efSearch=16")

        else:
            raise ValueError(f"Unknown FAISS index type: {index_type}. Use 'flat', 'ivf', 'hnsw', or 'auto'")

        return index

    def build_bm25_index(self, texts):
        """
        Build BM25 index with improved tokenization.

        Uses stemming and stop word removal for better matching.
        """
        tokenized_corpus = [
            tokenize_for_bm25(
                doc,
                enable_stemming=BM25_ENABLE_STEMMING,
                enable_stopwords=BM25_ENABLE_STOPWORDS,
                languages=BM25_LANGUAGES
            )
            for doc in tqdm(texts, desc="Tokenizing for BM25")
        ]
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
            faiss.write_index(faiss_index, self.faiss_binary_path)

    def load_indices(self):
        """Load all indices including metadata index"""
        multi_vector_index = load_pickle(self.faiss_index_path)
        bm25 = load_pickle(self.bm25_data_path)
        metadata_index = load_pickle(self.metadata_index_path)
        faiss_index = None
        if os.path.exists(self.faiss_binary_path):
            faiss_index = faiss.read_index(self.faiss_binary_path)
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

    def index_new_documents(self, multi_vector_index, bm25, metadata_index, faiss_index, embedding_model):
        """
        Detect new documents in DOCUMENTS_DIR, process only those, and merge into existing indices.

        Args:
            multi_vector_index: Existing multi-vector index (list of dicts)
            bm25: Existing BM25 index
            metadata_index: Existing metadata index (dict mapping doc names to chunk indices)
            faiss_index: Existing FAISS index
            embedding_model: Loaded SentenceTransformer model

        Returns:
            Tuple of (multi_vector_index, bm25, metadata_index, faiss_index, new_filenames)
            or None if no new documents were found.
        """
        from RAG_Framework.components.indexer.processors import process_single_document, SUPPORTED_EXTENSIONS as _SE

        # List all supported files in the documents directory
        all_files = [
            f for f in os.listdir(self.documents_dir)
            if os.path.splitext(f)[1].lower() in _SE
        ]

        # Determine which are already indexed by checking metadata_index keys
        indexed_filenames = set()
        for key in metadata_index:
            # Keys that have a file extension are document names
            if os.path.splitext(key)[1]:
                indexed_filenames.add(key)

        new_filenames = [f for f in all_files if f not in indexed_filenames]

        if not new_filenames:
            return None

        print(f"New documents detected: {new_filenames}")

        # Build file_infos for new files only
        max_chunk_length = CHUNK_SIZE
        chunk_overlap = CHUNK_OVERLAP
        existing_doc_count = len(indexed_filenames)

        file_infos = [
            (
                os.path.join(self.documents_dir, f),
                existing_doc_count + i,
                max_chunk_length,
                chunk_overlap,
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
            for chunk, meta in zip(chunks, metadata):
                if len(chunk) >= MIN_CHUNK_CHARS:
                    new_chunks.append(chunk)
                    new_metadata.append(meta)

        if not new_chunks:
            print("No content extracted from new documents.")
            return None

        print(f"Extracted {len(new_chunks)} chunks from {len(new_filenames)} new document(s)")

        # Generate embeddings for new chunks only
        print("Generating embeddings for new chunks...")
        new_embeddings = self.get_embeddings(new_chunks, model=embedding_model)

        # Calculate offset for new chunk indices
        offset = len(multi_vector_index)

        # Extend multi_vector_index
        new_entries = self.build_multi_vector_index(new_embeddings, new_metadata, new_chunks)
        multi_vector_index.extend(new_entries)

        # Add new vectors to FAISS index
        embeddings_array = np.array([item["embedding"] for item in new_entries]).astype('float32')
        faiss.normalize_L2(embeddings_array)
        faiss_index.add(embeddings_array)

        # Rebuild BM25 from all texts (fast — just tokenization)
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
        self.save_indices(multi_vector_index, bm25, metadata_index, faiss_index)
        print(f"Indices updated and saved. Total chunks: {len(multi_vector_index)}")

        return multi_vector_index, bm25, metadata_index, faiss_index, new_filenames

    def load_and_chunk_documents_parallel(self, max_chunk_length=None, chunk_overlap=None):
        """Load and process all supported documents in parallel."""
        from RAG_Framework.components.indexer.processors import process_single_document, SUPPORTED_EXTENSIONS as _SE

        # Use config defaults if not specified
        if max_chunk_length is None:
            max_chunk_length = CHUNK_SIZE
        if chunk_overlap is None:
            chunk_overlap = CHUNK_OVERLAP

        doc_files = [
            os.path.join(self.documents_dir, f)
            for f in os.listdir(self.documents_dir)
            if os.path.splitext(f)[1].lower() in _SE
        ]
        print(f"Found {len(doc_files)} document(s) to process.")
        if not doc_files:
            return [], []

        # Pass OCR configuration and chunk overlap to each process
        file_infos = [
            (f, i, max_chunk_length, chunk_overlap, self.enable_ocr, self.ocr_min_width, self.ocr_min_height, self.ocr_resolution)
            for i, f in enumerate(doc_files)
        ]

        # Use optimal number of processes (cpu_count - 1, minimum 1, maximum 8)
        num_processes = max(1, min(cpu_count() - 1, 8))
        print(f"Processing with {num_processes} parallel processes (OCR {'enabled' if self.enable_ocr else 'disabled'})")

        # Process files in parallel with progress bar
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_single_document, file_infos),
                total=len(file_infos),
                desc="Processing documents"
            ))

        all_chunks = []
        all_metadata = []
        for chunks, metadata in results:
            for chunk, meta in zip(chunks, metadata):
                if len(chunk) >= MIN_CHUNK_CHARS:
                    all_chunks.append(chunk)
                    all_metadata.append(meta)

        print(f"Extracted {len(all_chunks)} chunks from {len(doc_files)} documents")
        return all_chunks, all_metadata

    # Backward-compatible alias
    def load_and_content_chunk_pdfs_parallel(self, max_chunk_length=None, chunk_overlap=None):
        """Deprecated: use load_and_chunk_documents_parallel instead."""
        return self.load_and_chunk_documents_parallel(max_chunk_length, chunk_overlap)
