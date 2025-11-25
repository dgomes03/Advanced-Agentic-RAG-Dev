"""Document indexing and processing"""

import os
import numpy as np
import faiss
import pdfplumber
import pytesseract
import re
from multiprocessing import Pool
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from ..config import config
from ..utils import save_pickle, load_pickle


class Indexer:
    """Handles document loading, chunking, and index building"""

    def __init__(
        self,
        documents_dir=None,
        embedding_model_name=None,
        faiss_index_path=None,
        bm25_data_path=None,
        metadata_index_path=None
    ):
        self.documents_dir = documents_dir or config.DOCUMENTS_DIR
        self.embedding_model_name = embedding_model_name or config.EMBEDDING_MODEL_NAME
        self.faiss_index_path = faiss_index_path or config.MULTIVECTOR_INDEX_PATH
        self.bm25_data_path = bm25_data_path or config.BM25_DATA_PATH
        self.metadata_index_path = metadata_index_path or config.METADATA_INDEX_PATH

    @staticmethod
    def process_single_pdf(file_info):
        """Process a single PDF file and extract chunks"""
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
                        if table and any(cell for row in table for cell in row if cell):
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
                        # Use sentence-aware splitting
                        sentences = re.split(r'(?<=[.!?])\s+', text)
                        current_chunk = []
                        current_length = 0
                        chunks = []

                        for sentence in sentences:
                            if current_length + len(sentence) > max_chunk_length and current_chunk:
                                chunks.append(" ".join(current_chunk).strip())
                                current_chunk = [sentence]
                                current_length = len(sentence)
                            else:
                                current_chunk.append(sentence)
                                current_length += len(sentence) + 1

                        if current_chunk:
                            chunks.append(" ".join(current_chunk).strip())

                        for chunk_idx, chunk in enumerate(chunks):
                            if chunk:
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
                        for img_idx, img_dict in enumerate(page.images):
                            try:
                                if 'x0' in img_dict and 'y0' in img_dict and 'x1' in img_dict and 'y1' in img_dict:
                                    img_x0 = max(0, img_dict["x0"])
                                    img_top = max(0, img_dict["top"])
                                    img_x1 = min(page.width, img_dict["x1"])
                                    img_bottom = min(page.height, img_dict["bottom"])
                                    image = page.crop((img_x0, img_top, img_x1, img_bottom)).to_image(resolution=150)
                                    pil_img = image.original
                                    ocr_text = pytesseract.image_to_string(pil_img).strip()
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

            if doc_name and doc_name not in metadata_index:
                metadata_index[doc_name] = []
            if doc_name:
                metadata_index[doc_name].append(idx)

            if full_doc_id and full_doc_id not in metadata_index:
                metadata_index[full_doc_id] = []
            if full_doc_id:
                metadata_index[full_doc_id].append(idx)

        return metadata_index

    def get_embeddings(self, texts, model, batch_size=None):
        """Generate embeddings for a list of texts in batches"""
        batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
            batch = texts[i:i+batch_size]
            embeddings = model.encode(batch)
            all_embeddings.extend(embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    def build_faiss_index(self, multi_vector_index):
        """Build FAISS index from multi-vector index"""
        if not multi_vector_index:
            print("Warning: Multi-vector index is empty. Cannot build FAISS index.")
            return None

        embeddings = np.array([item["embedding"] for item in multi_vector_index]).astype('float32')
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return index

    def build_bm25_index(self, texts):
        """Build BM25 index from texts"""
        tokenized_corpus = [doc.lower().split() for doc in texts]
        bm25_index = BM25Okapi(tokenized_corpus)
        return bm25_index

    def build_multi_vector_index(self, embeddings, metadata, texts):
        """Build multi-vector index combining embeddings, metadata, and text"""
        index = []
        for emb, meta, text in zip(embeddings, metadata, texts):
            index.append({
                "embedding": np.array(emb, dtype=np.float32),
                "text": text,
                "metadata": meta
            })
        return index

    def save_indices(self, multi_vector_index, bm25, metadata_index, faiss_index=None):
        """Save all indices to disk"""
        os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
        save_pickle(multi_vector_index, self.faiss_index_path)
        save_pickle(bm25, self.bm25_data_path)
        save_pickle(metadata_index, self.metadata_index_path)

        if faiss_index:
            faiss.write_index(faiss_index, config.FAISS_INDEX_PATH)

    def load_indices(self):
        """Load all indices from disk"""
        multi_vector_index = load_pickle(self.faiss_index_path)
        bm25 = load_pickle(self.bm25_data_path)
        metadata_index = load_pickle(self.metadata_index_path)
        faiss_index = None

        if os.path.exists(config.FAISS_INDEX_PATH):
            faiss_index = faiss.read_index(config.FAISS_INDEX_PATH)

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

    def load_and_content_chunk_pdfs_parallel(self, max_chunk_length=None):
        """Load and process PDF documents in parallel"""
        max_chunk_length = max_chunk_length or config.MAX_CHUNK_LENGTH

        pdf_files = [
            os.path.join(self.documents_dir, f)
            for f in os.listdir(self.documents_dir)
            if f.lower().endswith('.pdf')
        ]

        print(f"Found {len(pdf_files)} PDF files to process.")
        if not pdf_files:
            return [], []

        file_infos = [(f, i, max_chunk_length) for i, f in enumerate(pdf_files)]

        # Process files in parallel
        with Pool(processes=config.PDF_PROCESSING_WORKERS) as pool:
            results = pool.map(self.process_single_pdf, file_infos)

        all_chunks = []
        all_metadata = []
        for chunks, metadata in results:
            all_chunks.extend(chunks)
            all_metadata.extend(metadata)

        return all_chunks, all_metadata
