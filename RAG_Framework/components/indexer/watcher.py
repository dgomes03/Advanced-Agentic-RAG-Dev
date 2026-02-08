import threading
import traceback

from RAG_Framework.components.indexer.standard import Indexer


def check_for_new_documents(retriever):
    """
    One-shot check: detect new PDFs in DOCUMENTS_DIR and index them incrementally.
    Runs in a background thread so it doesn't block startup.
    """
    try:
        indexer = Indexer(enable_ocr=True)

        # Triggers lazy load of indices if needed
        multi_vector_index = retriever.multi_vector_index
        bm25 = retriever.bm25
        metadata_index = retriever.metadata_index
        faiss_index = retriever.faiss_index

        if multi_vector_index is None or metadata_index is None or faiss_index is None:
            print("DocumentWatcher: No existing indices found, skipping incremental check.")
            return

        # Work on copies to avoid mutating live data during indexing
        mvi_copy = list(multi_vector_index)
        meta_copy = dict(metadata_index)

        embedding_model = retriever.embedding_model

        result = indexer.index_new_documents(
            mvi_copy, bm25, meta_copy, faiss_index, embedding_model
        )

        if result is not None:
            updated_mvi, updated_bm25, updated_meta, updated_faiss, new_files = result
            retriever.update_indices(updated_mvi, updated_bm25, updated_meta, updated_faiss)
            print(f"DocumentWatcher: Indexed {len(new_files)} new document(s): {', '.join(new_files)}")
        else:
            print("DocumentWatcher: No new documents found.")
    except Exception as e:
        print(f"DocumentWatcher error: {e}")
        traceback.print_exc()


def start_document_check(retriever):
    """Launch check_for_new_documents in a daemon background thread."""
    print("Checking for new documents...")
    thread = threading.Thread(target=check_for_new_documents, args=(retriever,), daemon=True, name="DocumentWatcher")
    thread.start()
