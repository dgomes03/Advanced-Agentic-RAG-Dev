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
    METADATA_INDEX_PATH, FAISS_INDEX_PATH
)
from RAG_Framework.core.utils import save_pickle, load_pickle


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
    def process_single_pdf(file_info):
        file_path, doc_id, max_chunk_length, enable_ocr, ocr_min_width, ocr_min_height, ocr_resolution = file_info
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

    def get_embeddings(self, texts, model, batch_size=32):
        """Generate embeddings for a list of texts in batches."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
            batch = texts[i:i+batch_size]
            embeddings = model.encode(batch)
            all_embeddings.extend(embeddings)
        return np.array(all_embeddings, dtype=np.float32)

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

    def load_and_content_chunk_pdfs_parallel(self, max_chunk_length=512):
        """Load and process PDF documents in parallel"""
        pdf_files = [os.path.join(self.documents_dir, f) for f in os.listdir(self.documents_dir)
                    if f.lower().endswith('.pdf')]
        print(f"Found {len(pdf_files)} PDF files to process.")
        if not pdf_files:
            return [], []

        # Pass OCR configuration to each process
        file_infos = [
            (f, i, max_chunk_length, self.enable_ocr, self.ocr_min_width, self.ocr_min_height, self.ocr_resolution)
            for i, f in enumerate(pdf_files)
        ]

        # Use optimal number of processes (cpu_count - 1, minimum 1, maximum 8)
        num_processes = max(1, min(cpu_count() - 1, 8))
        print(f"Processing with {num_processes} parallel processes (OCR {'enabled' if self.enable_ocr else 'disabled'})")

        # Process files in parallel with progress bar
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(self.process_single_pdf, file_infos),
                total=len(file_infos),
                desc="Processing PDFs"
            ))

        all_chunks = []
        all_metadata = []
        for chunks, metadata in results:
            all_chunks.extend(chunks)
            all_metadata.extend(metadata)

        print(f"Extracted {len(all_chunks)} chunks from {len(pdf_files)} documents")
        return all_chunks, all_metadata
