"""
API integration tests for all four classification endpoints and the health check.

All three route-level classifiers and the compare_service instance are replaced
with MagicMocks via monkeypatch so tests are fully isolated and instant.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import app.api.routes as routes_module
from app.main import app

client = TestClient(app)

ABSTRACT = "This paper studies neural networks and deep learning for natural language processing."


# ---------------------------------------------------------------------------
# Result builders – mimic the dicts each classifier returns
# ---------------------------------------------------------------------------


def kw_result(label: str, score: float) -> dict:
    return {
        "predicted_category": label,
        "scores": [
            {
                "label": label,
                "score": score,
                "matched_keywords": ["neural network", "deep learning"],
            },
            {"label": "Physics", "score": 0.0, "matched_keywords": []},
        ],
    }


def tfidf_result(label: str | None, score: float) -> dict:
    return {
        "predicted_category": label,
        "scores": [
            {"label": label or "AI", "score": score},
            {"label": "Physics", "score": 0.0},
        ],
    }


def emb_result(label: str | None, score: float) -> dict:
    return {
        "predicted_category": label,
        "scores": [
            {"label": label or "AI", "score": score},
            {"label": "Physics", "score": 0.0},
        ],
    }


def cmp_result(label: str | None) -> dict:
    return {
        "keyword_matching": {"predicted_category": label, "score": 0.8},
        "tfidf": {"predicted_category": label, "score": 0.75},
        "embedding": {"predicted_category": label, "score": 0.92},
    }


# ---------------------------------------------------------------------------
# Fixture – replace all route-level singletons with mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mocks(monkeypatch):
    km = MagicMock()
    tc = MagicMock()
    ec = MagicMock()
    cs = MagicMock()

    ec.embedding_model_name = "mock-model"

    monkeypatch.setattr(routes_module, "keyword_matching_classifier", km)
    monkeypatch.setattr(routes_module, "tfidf_classifier", tc)
    monkeypatch.setattr(routes_module, "embedding_classifier", ec)
    monkeypatch.setattr(routes_module, "compare_service", cs)

    return km, tc, ec, cs


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_body_is_ok(self):
        assert client.get("/health").json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /classify/keyword-matching
# ---------------------------------------------------------------------------


class TestKeywordMatchingEndpoint:
    URL = "/classify/keyword-matching"

    def test_success_returns_200(self, mocks):
        km, _, _, _ = mocks
        km.classify.return_value = kw_result("AI", 0.8)
        assert client.post(self.URL, json={"abstract": ABSTRACT}).status_code == 200

    def test_response_method_field(self, mocks):
        km, _, _, _ = mocks
        km.classify.return_value = kw_result("AI", 0.8)
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert body["method"] == "keyword_matching"

    def test_predicted_category_in_response(self, mocks):
        km, _, _, _ = mocks
        km.classify.return_value = kw_result("AI", 0.8)
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert body["predicted_category"] == "AI"

    def test_scores_present_and_correct(self, mocks):
        km, _, _, _ = mocks
        km.classify.return_value = kw_result("AI", 0.8)
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert len(body["scores"]) > 0
        assert body["scores"][0]["label"] == "AI"
        assert body["scores"][0]["score"] == 0.8

    def test_reasoning_has_matched_keywords(self, mocks):
        km, _, _, _ = mocks
        km.classify.return_value = kw_result("AI", 0.8)
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert "matched_keywords" in body["reasoning"]
        assert "AI" in body["reasoning"]["matched_keywords"]

    def test_empty_abstract_returns_422(self, mocks):
        assert client.post(self.URL, json={"abstract": ""}).status_code == 422

    def test_missing_abstract_key_returns_422(self, mocks):
        assert client.post(self.URL, json={}).status_code == 422

    def test_classifier_value_error_returns_400(self, mocks):
        km, _, _, _ = mocks
        km.classify.side_effect = ValueError("bad input")
        assert client.post(self.URL, json={"abstract": ABSTRACT}).status_code == 400

    def test_no_keyword_match_returns_400(self, mocks):
        km, _, _, _ = mocks
        km.classify.return_value = {
            "predicted_category": None,
            "scores": [{"label": "AI", "score": 0.0, "matched_keywords": []}],
        }
        response = client.post(self.URL, json={"abstract": ABSTRACT})
        assert response.status_code == 400
        assert "No keywords found" in response.json()["detail"]

    def test_classifier_called_with_abstract(self, mocks):
        km, _, _, _ = mocks
        km.classify.return_value = kw_result("AI", 0.8)
        client.post(self.URL, json={"abstract": ABSTRACT})
        km.classify.assert_called_once_with(ABSTRACT)


# ---------------------------------------------------------------------------
# POST /classify/tfidf
# ---------------------------------------------------------------------------


class TestTfidfEndpoint:
    URL = "/classify/tfidf"

    def test_success_returns_200(self, mocks):
        _, tc, _, _ = mocks
        tc.classify.return_value = tfidf_result("Physics", 0.75)
        tc.get_top_matching_terms.return_value = []
        assert client.post(self.URL, json={"abstract": ABSTRACT}).status_code == 200

    def test_response_method_field(self, mocks):
        _, tc, _, _ = mocks
        tc.classify.return_value = tfidf_result("Physics", 0.75)
        tc.get_top_matching_terms.return_value = []
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert body["method"] == "TF-IDF_with_cosine_similarity"

    def test_predicted_category_in_response(self, mocks):
        _, tc, _, _ = mocks
        tc.classify.return_value = tfidf_result("Physics", 0.75)
        tc.get_top_matching_terms.return_value = []
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert body["predicted_category"] == "Physics"

    def test_reasoning_has_ngram_range(self, mocks):
        _, tc, _, _ = mocks
        tc.classify.return_value = tfidf_result("Physics", 0.75)
        tc.get_top_matching_terms.return_value = []
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert body["reasoning"]["ngram_range"] == [1, 2]

    def test_reasoning_has_top_matching_terms(self, mocks):
        _, tc, _, _ = mocks
        tc.classify.return_value = tfidf_result("Physics", 0.75)
        tc.get_top_matching_terms.return_value = [{"term": "quantum", "score": 0.5}]
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert body["reasoning"]["top_matching_terms"] == [
            {"term": "quantum", "score": 0.5}
        ]

    def test_no_match_returns_200_with_none_category(self, mocks):
        _, tc, _, _ = mocks
        tc.classify.return_value = {
            "predicted_category": None,
            "scores": [{"label": "Physics", "score": 0.0}],
        }
        response = client.post(self.URL, json={"abstract": ABSTRACT})
        assert response.status_code == 200
        assert response.json()["predicted_category"] is None

    def test_get_top_matching_terms_not_called_when_no_match(self, mocks):
        _, tc, _, _ = mocks
        tc.classify.return_value = {
            "predicted_category": None,
            "scores": [{"label": "Physics", "score": 0.0}],
        }
        client.post(self.URL, json={"abstract": ABSTRACT})
        tc.get_top_matching_terms.assert_not_called()

    def test_get_top_matching_terms_called_with_correct_args(self, mocks):
        _, tc, _, _ = mocks
        tc.classify.return_value = tfidf_result("Physics", 0.75)
        tc.get_top_matching_terms.return_value = []
        client.post(self.URL, json={"abstract": ABSTRACT})
        tc.get_top_matching_terms.assert_called_once_with(
            abstract=ABSTRACT,
            predicted_category="Physics",
        )

    def test_empty_abstract_returns_422(self, mocks):
        assert client.post(self.URL, json={"abstract": ""}).status_code == 422

    def test_classifier_value_error_returns_400(self, mocks):
        _, tc, _, _ = mocks
        tc.classify.side_effect = ValueError("tfidf error")
        assert client.post(self.URL, json={"abstract": ABSTRACT}).status_code == 400

    def test_classifier_called_with_abstract(self, mocks):
        _, tc, _, _ = mocks
        tc.classify.return_value = tfidf_result("Physics", 0.75)
        tc.get_top_matching_terms.return_value = []
        client.post(self.URL, json={"abstract": ABSTRACT})
        tc.classify.assert_called_once_with(ABSTRACT)


# ---------------------------------------------------------------------------
# POST /classify/embedding
# ---------------------------------------------------------------------------


class TestEmbeddingEndpoint:
    URL = "/classify/embedding"

    def test_success_returns_200(self, mocks):
        _, _, ec, _ = mocks
        ec.classify.return_value = emb_result("AI", 0.92)
        ec.get_top_supporting_sentences.return_value = []
        assert client.post(self.URL, json={"abstract": ABSTRACT}).status_code == 200

    def test_response_method_field(self, mocks):
        _, _, ec, _ = mocks
        ec.classify.return_value = emb_result("AI", 0.92)
        ec.get_top_supporting_sentences.return_value = []
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert body["method"] == "sentence_embedding"

    def test_predicted_category_in_response(self, mocks):
        _, _, ec, _ = mocks
        ec.classify.return_value = emb_result("AI", 0.92)
        ec.get_top_supporting_sentences.return_value = []
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert body["predicted_category"] == "AI"

    def test_reasoning_has_model_name(self, mocks):
        _, _, ec, _ = mocks
        ec.classify.return_value = emb_result("AI", 0.92)
        ec.get_top_supporting_sentences.return_value = []
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert body["reasoning"]["model"] == "mock-model"

    def test_reasoning_has_supporting_sentences(self, mocks):
        _, _, ec, _ = mocks
        ec.classify.return_value = emb_result("AI", 0.92)
        ec.get_top_supporting_sentences.return_value = [
            {"sentence": "Neural networks are powerful.", "score": 0.85}
        ]
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert body["reasoning"]["top_supporting_sentences"] == [
            {"sentence": "Neural networks are powerful.", "score": 0.85}
        ]

    def test_no_match_returns_200_with_none_category(self, mocks):
        _, _, ec, _ = mocks
        ec.classify.return_value = {
            "predicted_category": None,
            "scores": [{"label": "AI", "score": 0.0}],
        }
        response = client.post(self.URL, json={"abstract": ABSTRACT})
        assert response.status_code == 200
        assert response.json()["predicted_category"] is None

    def test_get_top_supporting_sentences_not_called_when_no_match(self, mocks):
        _, _, ec, _ = mocks
        ec.classify.return_value = {
            "predicted_category": None,
            "scores": [{"label": "AI", "score": 0.0}],
        }
        client.post(self.URL, json={"abstract": ABSTRACT})
        ec.get_top_supporting_sentences.assert_not_called()

    def test_get_top_supporting_sentences_called_with_correct_args(self, mocks):
        _, _, ec, _ = mocks
        ec.classify.return_value = emb_result("AI", 0.92)
        ec.get_top_supporting_sentences.return_value = []
        client.post(self.URL, json={"abstract": ABSTRACT})
        ec.get_top_supporting_sentences.assert_called_once_with(
            abstract=ABSTRACT,
            predicted_category="AI",
        )

    def test_empty_abstract_returns_422(self, mocks):
        assert client.post(self.URL, json={"abstract": ""}).status_code == 422

    def test_classifier_value_error_returns_400(self, mocks):
        _, _, ec, _ = mocks
        ec.classify.side_effect = ValueError("embedding error")
        assert client.post(self.URL, json={"abstract": ABSTRACT}).status_code == 400

    def test_classifier_called_with_abstract(self, mocks):
        _, _, ec, _ = mocks
        ec.classify.return_value = emb_result("AI", 0.92)
        ec.get_top_supporting_sentences.return_value = []
        client.post(self.URL, json={"abstract": ABSTRACT})
        ec.classify.assert_called_once_with(ABSTRACT)


# ---------------------------------------------------------------------------
# POST /classify/compare
# ---------------------------------------------------------------------------


class TestCompareEndpoint:
    URL = "/classify/compare"

    def test_success_returns_200(self, mocks):
        _, _, _, cs = mocks
        cs.compare.return_value = cmp_result("AI")
        assert client.post(self.URL, json={"abstract": ABSTRACT}).status_code == 200

    def test_response_method_is_compare(self, mocks):
        _, _, _, cs = mocks
        cs.compare.return_value = cmp_result("AI")
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert body["method"] == "compare"

    def test_results_has_all_three_classifiers(self, mocks):
        _, _, _, cs = mocks
        cs.compare.return_value = cmp_result("AI")
        results = client.post(self.URL, json={"abstract": ABSTRACT}).json()["results"]
        assert "keyword_matching" in results
        assert "tfidf" in results
        assert "embedding" in results

    def test_each_result_has_predicted_category_and_score(self, mocks):
        _, _, _, cs = mocks
        cs.compare.return_value = cmp_result("AI")
        results = client.post(self.URL, json={"abstract": ABSTRACT}).json()["results"]
        for key in ("keyword_matching", "tfidf", "embedding"):
            assert "predicted_category" in results[key]
            assert "score" in results[key]

    def test_final_decision_is_present(self, mocks):
        _, _, _, cs = mocks
        cs.compare.return_value = cmp_result("AI")
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert "final_decision" in body

    def test_final_decision_selected_method_is_embedding(self, mocks):
        _, _, _, cs = mocks
        cs.compare.return_value = cmp_result("AI")
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert body["final_decision"]["selected_method"] == "embedding"

    def test_final_decision_predicted_category_taken_from_embedding(self, mocks):
        _, _, _, cs = mocks
        cs.compare.return_value = cmp_result("Physics")
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert body["final_decision"]["predicted_category"] == "Physics"

    def test_final_decision_has_non_empty_reason(self, mocks):
        _, _, _, cs = mocks
        cs.compare.return_value = cmp_result("AI")
        body = client.post(self.URL, json={"abstract": ABSTRACT}).json()
        assert len(body["final_decision"]["reason"]) > 0

    def test_empty_abstract_returns_422(self, mocks):
        assert client.post(self.URL, json={"abstract": ""}).status_code == 422

    def test_compare_value_error_returns_400(self, mocks):
        _, _, _, cs = mocks
        cs.compare.side_effect = ValueError("compare error")
        assert client.post(self.URL, json={"abstract": ABSTRACT}).status_code == 400

    def test_compare_service_called_with_abstract(self, mocks):
        _, _, _, cs = mocks
        cs.compare.return_value = cmp_result("AI")
        client.post(self.URL, json={"abstract": ABSTRACT})
        cs.compare.assert_called_once_with(ABSTRACT)
