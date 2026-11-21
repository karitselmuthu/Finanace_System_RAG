from __future__ import annotations

import pytest
from unittest.mock import patch

from src.retrieval.reranker import rerank


class TestReranker:
    def _make_hits(self):
        return [
            {"id": "c1", "text": "Credit deposit 500.00 completed", "metadata": {}, "score": 0.85, "distance": 0.15},
            {"id": "c2", "text": "Account statement header page 1", "metadata": {}, "score": 0.72, "distance": 0.28},
            {"id": "c3", "text": "Debit payment failed 200.00", "metadata": {}, "score": 0.78, "distance": 0.22},
        ]

    def test_returns_sorted_by_rerank_score(self):
        hits = self._make_hits()
        result = rerank(hits, "failed debit transactions")
        scores = [h["rerank_score"] for h in result]
        assert scores == sorted(scores, reverse=True)

    def test_all_hits_have_rerank_score(self):
        hits = self._make_hits()
        result = rerank(hits, "deposits")
        for h in result:
            assert "rerank_score" in h

    def test_top_n_limits_results(self):
        hits = self._make_hits()
        result = rerank(hits, "credit deposit", top_n=2)
        assert len(result) == 2

    def test_empty_hits(self):
        result = rerank([], "any query")
        assert result == []


class TestRetriever:
    def test_retrieve_filters_by_threshold(self):
        mock_hits = [
            {"id": "a", "text": "tx", "metadata": {}, "distance": 0.1},   # score=0.9 ✓
            {"id": "b", "text": "tx", "metadata": {}, "distance": 0.8},   # score=0.2 ✗
        ]
        with patch("src.retrieval.retriever.query_similar", return_value=mock_hits):
            with patch("src.retrieval.retriever.settings") as mock_settings:
                mock_settings.top_k_results = 10
                mock_settings.similarity_threshold = 0.5
                from src.retrieval.retriever import retrieve
                results = retrieve("test query")

        assert len(results) == 1
        assert results[0]["id"] == "a"
