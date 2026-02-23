"""
Tests for the Retriever class (components/retrievers/FAISS_BM25.py).

Run from the RAG_Framework directory:
    pytest tests/test_retriever.py -v
"""

import math
import threading

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from RAG_Framework.components.retrievers.FAISS_BM25 import Retriever


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_retriever() -> Retriever:
    """Instantiate a Retriever with no real models or indices loaded."""
    r = Retriever.__new__(Retriever)
    r.llm_model = None
    r.llm_tokenizer = None
    r._embedding_model_name = "dummy-embed"
    r._reranker_model_name = "dummy-rerank"
    r._index_paths = None
    r._embedding_model = None
    r._reranker = None
    r._multi_vector_index = None
    r._bm25 = None
    r._metadata_index = None
    r._faiss_index = None
    r._parent_store = None
    r._indices_loaded = True  # prevents _ensure_indices_loaded from hitting disk
    r._load_lock = threading.Lock()
    r.last_retrieved_metadata = []
    return r


@pytest.fixture
def retriever() -> Retriever:
    return _make_retriever()


# ---------------------------------------------------------------------------
# normalize_scores
# ---------------------------------------------------------------------------

class TestNormalizeScores:

    def test_empty_dict_returns_empty(self):
        assert Retriever.normalize_scores({}) == {}

    def test_all_same_values_return_one(self):
        result = Retriever.normalize_scores({0: 3.0, 1: 3.0, 2: 3.0})
        assert all(v == pytest.approx(1.0) for v in result.values())

    def test_min_max_normalization(self):
        result = Retriever.normalize_scores({0: 0.0, 1: 5.0, 2: 10.0})
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_single_entry_returns_one(self):
        result = Retriever.normalize_scores({42: 7.5})
        assert result[42] == pytest.approx(1.0)

    def test_negative_scores_handled(self):
        result = Retriever.normalize_scores({0: -10.0, 1: 0.0, 2: 10.0})
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_keys_preserved(self):
        result = Retriever.normalize_scores({7: 1.0, 99: 3.0})
        assert set(result.keys()) == {7, 99}


# ---------------------------------------------------------------------------
# retrieve_with_faiss
# ---------------------------------------------------------------------------

class TestRetrieveWithFaiss:

    def test_returns_empty_when_index_is_none(self):
        assert Retriever.retrieve_with_faiss([0.1, 0.2], None, k=5) == []

    def test_filters_minus_one_indices(self):
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]]),
            np.array([[0,   1,   -1]]),
        )
        results = Retriever.retrieve_with_faiss([0.0] * 4, mock_index, k=3)
        assert all(idx != -1 for idx, _ in results)
        assert len(results) == 2

    def test_returns_correct_idx_score_pairs(self):
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.95, 0.80]]),
            np.array([[3,    7   ]]),
        )
        results = Retriever.retrieve_with_faiss([0.0] * 4, mock_index, k=2)
        assert results[0] == (3, pytest.approx(0.95))
        assert results[1] == (7, pytest.approx(0.80))

    def test_all_minus_one_returns_empty(self):
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[-1.0, -1.0]]),
            np.array([[-1,   -1  ]]),
        )
        results = Retriever.retrieve_with_faiss([0.0] * 4, mock_index, k=2)
        assert results == []


# ---------------------------------------------------------------------------
# retrieve_with_bm25
# ---------------------------------------------------------------------------

class TestRetrieveWithBm25:

    def test_returns_empty_when_bm25_is_none(self):
        assert Retriever.retrieve_with_bm25(["token"], None, top_k=5) == []

    def test_returns_top_k_by_score(self):
        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = np.array([1.0, 0.5, 3.0, 0.2, 2.0])
        results = Retriever.retrieve_with_bm25(["query"], mock_bm25, top_k=2)
        indices = [idx for idx, _ in results]
        assert indices[0] == 2   # highest score
        assert indices[1] == 4   # second highest

    def test_scores_attached_correctly(self):
        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = np.array([0.0, 5.0, 2.0])
        results = Retriever.retrieve_with_bm25(["q"], mock_bm25, top_k=3)
        scores_dict = dict(results)
        assert scores_dict[1] == pytest.approx(5.0)
        assert scores_dict[2] == pytest.approx(2.0)

    def test_top_k_does_not_exceed_available(self):
        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = np.array([1.0, 2.0])
        results = Retriever.retrieve_with_bm25(["q"], mock_bm25, top_k=10)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# rerank_chunks_with_metadata
# ---------------------------------------------------------------------------

class TestRerankChunksWithMetadata:
    CHUNKS = ["chunk_A", "chunk_B", "chunk_C"]
    METAS  = [{"id": "A"}, {"id": "B"}, {"id": "C"}]

    def _set_reranker(self, retriever, scores):
        mock = MagicMock()
        mock.predict.return_value = np.array(scores, dtype=float)
        retriever._reranker = mock

    def test_sorted_descending_by_score(self, retriever):
        self._set_reranker(retriever, [1.0, 3.0, 2.0])
        result = retriever.rerank_chunks_with_metadata("q", self.CHUNKS, self.METAS)
        assert [c for c, _ in result] == ["chunk_B", "chunk_C", "chunk_A"]

    def test_top_n_limits_output(self, retriever):
        self._set_reranker(retriever, [1.0, 3.0, 2.0])
        result = retriever.rerank_chunks_with_metadata("q", self.CHUNKS, self.METAS, top_n=2)
        assert len(result) == 2
        assert result[0][0] == "chunk_B"

    def test_metadata_preserved_after_rerank(self, retriever):
        self._set_reranker(retriever, [1.0, 3.0, 2.0])
        result = retriever.rerank_chunks_with_metadata("q", self.CHUNKS, self.METAS)
        assert result[0] == ("chunk_B", {"id": "B"})

    def test_no_threshold_returns_all(self, retriever):
        self._set_reranker(retriever, [-5.0, -4.0, -3.0])
        result = retriever.rerank_chunks_with_metadata("q", self.CHUNKS, self.METAS, threshold=None)
        assert len(result) == 3

    def test_threshold_filters_low_confidence_chunks(self, retriever):
        # sigmoid(5) ≈ 0.993 → above 0.3 threshold
        # sigmoid(-10) ≈ 0.00005 → below 0.3 threshold
        self._set_reranker(retriever, [5.0, -10.0, 5.0])
        result = retriever.rerank_chunks_with_metadata("q", self.CHUNKS, self.METAS, threshold=0.3)
        texts = [c for c, _ in result]
        assert "chunk_B" not in texts
        assert "chunk_A" in texts
        assert "chunk_C" in texts

    def test_threshold_keeps_best_when_all_below(self, retriever):
        # All very negative → sigmoid ≈ 0, all below threshold=0.3
        self._set_reranker(retriever, [-20.0, -15.0, -18.0])
        result = retriever.rerank_chunks_with_metadata("q", self.CHUNKS, self.METAS, threshold=0.3)
        # Safety net: must still return exactly 1 result
        assert len(result) == 1
        # Best raw score is -15.0 → chunk_B
        assert result[0][0] == "chunk_B"

    def test_empty_chunks_returns_empty(self, retriever):
        self._set_reranker(retriever, [])
        result = retriever.rerank_chunks_with_metadata("q", [], [], threshold=0.5)
        assert result == []

    def test_threshold_zero_keeps_everything(self, retriever):
        # sigmoid(any score) >= 0 always → threshold=0.0 filters nothing
        self._set_reranker(retriever, [-100.0, -50.0, -1.0])
        result = retriever.rerank_chunks_with_metadata("q", self.CHUNKS, self.METAS, threshold=0.0)
        assert len(result) == 3

    def test_threshold_one_keeps_only_perfect(self, retriever):
        # sigmoid(x) == 1.0 only in the limit → threshold=1.0 filters everything
        # → safety net kicks in and returns 1
        self._set_reranker(retriever, [5.0, 3.0, 1.0])
        result = retriever.rerank_chunks_with_metadata("q", self.CHUNKS, self.METAS, threshold=1.0)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _resolve_parents
# ---------------------------------------------------------------------------

class TestResolveParents:

    def test_no_parent_store_returns_children_unchanged(self, retriever):
        retriever._parent_store = None
        pairs = [("child_text", {"parent_idx": 0})]
        assert retriever._resolve_parents(pairs) == pairs

    def test_resolves_to_parent_text(self, retriever):
        retriever._parent_store = [
            {"text": "parent_0", "metadata": {"page": 1}},
        ]
        result = retriever._resolve_parents([("child", {"parent_idx": 0})])
        assert result[0][0] == "parent_0"
        assert result[0][1] == {"page": 1}

    def test_deduplicates_by_parent_idx(self, retriever):
        retriever._parent_store = [
            {"text": "parent_0", "metadata": {}},
        ]
        pairs = [
            ("child_1", {"parent_idx": 0}),
            ("child_2", {"parent_idx": 0}),
        ]
        result = retriever._resolve_parents(pairs)
        assert len(result) == 1
        assert result[0][0] == "parent_0"

    def test_different_parent_idxs_both_resolved(self, retriever):
        retriever._parent_store = [
            {"text": "parent_0", "metadata": {}},
            {"text": "parent_1", "metadata": {}},
        ]
        pairs = [
            ("child_a", {"parent_idx": 0}),
            ("child_b", {"parent_idx": 1}),
        ]
        result = retriever._resolve_parents(pairs)
        assert len(result) == 2
        assert result[0][0] == "parent_0"
        assert result[1][0] == "parent_1"

    def test_top_n_limits_resolved_results(self, retriever):
        retriever._parent_store = [
            {"text": f"parent_{i}", "metadata": {}} for i in range(5)
        ]
        pairs = [(f"child_{i}", {"parent_idx": i}) for i in range(5)]
        result = retriever._resolve_parents(pairs, top_n=3)
        assert len(result) == 3

    def test_out_of_range_parent_idx_falls_back_to_child(self, retriever):
        retriever._parent_store = [{"text": "parent_0", "metadata": {}}]
        pairs = [("child_x", {"parent_idx": 999})]
        result = retriever._resolve_parents(pairs)
        assert result[0][0] == "child_x"

    def test_missing_parent_idx_key_returns_child(self, retriever):
        retriever._parent_store = [{"text": "parent_0", "metadata": {}}]
        pairs = [("child_no_parent", {})]
        result = retriever._resolve_parents(pairs)
        assert result[0][0] == "child_no_parent"


# ---------------------------------------------------------------------------
# retrieve_by_metadata
# ---------------------------------------------------------------------------

class TestRetrieveByMetadata:
    MULTI_VECTOR = [
        {"text": "chunk_0", "metadata": {"page": 0, "chunk_idx": 0, "slide": 0, "sheet_index": 0, "table_idx": 0, "img_idx": 0}},
        {"text": "chunk_1", "metadata": {"page": 1, "chunk_idx": 0, "slide": 0, "sheet_index": 0, "table_idx": 0, "img_idx": 0}},
        {"text": "chunk_2", "metadata": {"page": 0, "chunk_idx": 1, "slide": 0, "sheet_index": 0, "table_idx": 0, "img_idx": 0}},
    ]

    def test_exact_match_returns_chunks(self, retriever):
        retriever._multi_vector_index = self.MULTI_VECTOR
        retriever._metadata_index = {"doc.pdf": [0, 1]}
        result = retriever.retrieve_by_metadata("doc.pdf")
        assert len(result) == 2
        assert {r["text"] for r in result} == {"chunk_0", "chunk_1"}

    def test_partial_match_used_as_fallback(self, retriever):
        retriever._multi_vector_index = self.MULTI_VECTOR
        retriever._metadata_index = {"allometric_model.pdf": [0]}
        result = retriever.retrieve_by_metadata("allometric")
        assert len(result) == 1
        assert result[0]["text"] == "chunk_0"

    def test_no_match_returns_empty_list(self, retriever):
        retriever._multi_vector_index = self.MULTI_VECTOR
        retriever._metadata_index = {"other.pdf": [0]}
        result = retriever.retrieve_by_metadata("nonexistent")
        assert result == []

    def test_sorted_by_page_then_chunk_idx(self, retriever):
        retriever._multi_vector_index = self.MULTI_VECTOR
        # Provide indices out of order: chunk_2 (page 0, chunk 1), chunk_1 (page 1), chunk_0 (page 0, chunk 0)
        retriever._metadata_index = {"doc.pdf": [2, 1, 0]}
        result = retriever.retrieve_by_metadata("doc.pdf", sort_by_page=True)
        assert [r["text"] for r in result] == ["chunk_0", "chunk_2", "chunk_1"]

    def test_sort_disabled_preserves_index_order(self, retriever):
        retriever._multi_vector_index = self.MULTI_VECTOR
        retriever._metadata_index = {"doc.pdf": [2, 0]}
        result = retriever.retrieve_by_metadata("doc.pdf", sort_by_page=False)
        assert result[0]["text"] == "chunk_2"
        assert result[1]["text"] == "chunk_0"

    def test_result_contains_text_metadata_and_index_keys(self, retriever):
        retriever._multi_vector_index = self.MULTI_VECTOR
        retriever._metadata_index = {"doc.pdf": [0]}
        result = retriever.retrieve_by_metadata("doc.pdf")
        assert "text" in result[0]
        assert "metadata" in result[0]
        assert "index" in result[0]


# ---------------------------------------------------------------------------
# combined_retrieval  (integration — all heavy deps mocked)
# ---------------------------------------------------------------------------

class TestCombinedRetrieval:

    def _wire_mocks(self, retriever):
        """Attach minimal mocks for a full combined_retrieval call."""
        retriever._multi_vector_index = [
            {"text": "relevant chunk", "metadata": {"page": 0}},
            {"text": "another chunk",  "metadata": {"page": 1}},
        ]

        mock_faiss = MagicMock()
        mock_faiss.search.return_value = (
            np.array([[0.9, 0.5]]),
            np.array([[0,   1  ]]),
        )
        retriever._faiss_index = mock_faiss

        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = np.array([0.8, 0.3])
        retriever._bm25 = mock_bm25

        mock_embed = MagicMock()
        mock_embed.encode.return_value = np.random.rand(2, 4).astype("float32")
        retriever._embedding_model = mock_embed

        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = np.array([3.0, 1.0])
        retriever._reranker = mock_reranker

    def test_returns_string_context(self, retriever):
        self._wire_mocks(retriever)
        with patch.object(retriever, "expand_query", return_value=[]):
            result = retriever.combined_retrieval("test query")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_indices_returns_error_string(self, retriever):
        retriever._faiss_index = None
        retriever._bm25 = None
        retriever._multi_vector_index = None
        result = retriever.combined_retrieval("test query")
        assert "No documents" in result

    def test_highest_reranked_chunk_comes_first(self, retriever):
        self._wire_mocks(retriever)
        with patch.object(retriever, "expand_query", return_value=[]):
            result = retriever.combined_retrieval("test query")
        # "relevant chunk" has reranker score 3.0 (highest)
        assert result.startswith("relevant chunk")

    def test_last_retrieved_metadata_populated(self, retriever):
        self._wire_mocks(retriever)
        with patch.object(retriever, "expand_query", return_value=[]):
            retriever.combined_retrieval("test query")
        assert len(retriever.last_retrieved_metadata) > 0

    def test_chunks_joined_by_separator(self, retriever):
        self._wire_mocks(retriever)
        with patch.object(retriever, "expand_query", return_value=[]):
            result = retriever.combined_retrieval("test query")
        assert "---" in result

    def test_safety_net_when_all_chunks_below_threshold(self, retriever):
        self._wire_mocks(retriever)
        # All reranker scores very negative → sigmoid ≈ 0, below any threshold
        retriever._reranker.predict.return_value = np.array([-20.0, -20.0])
        with patch.object(retriever, "expand_query", return_value=[]):
            with patch("RAG_Framework.components.retrievers.FAISS_BM25.RERANKER_CONFIDENCE_THRESHOLD", 0.3):
                result = retriever.combined_retrieval("test query")
        # Safety net: at least 1 chunk must always be returned
        assert isinstance(result, str)
        assert len(result) > 0
