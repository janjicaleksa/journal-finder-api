"""
Unit tests for CompareService.

All three classifiers are replaced with MagicMock objects so these tests
cover only CompareService logic (score extraction, None handling, delegation)
and not the classifiers themselves.
"""
import pytest
from unittest.mock import MagicMock
from app.services.compare_service import CompareService


def make_classifier_result(label: str | None, score: float) -> dict:
    """Build the dict shape that each classifier's .classify() returns."""
    scores = [{"label": label, "score": score}] if label is not None else []
    return {
        "predicted_category": label,
        "scores": scores,
    }


def build_service(
    keyword_result: dict | None = None,
    tfidf_result: dict | None = None,
    embedding_result: dict | None = None,
) -> CompareService:
    """Return a CompareService whose classifiers return the given results."""
    keyword_mock = MagicMock()
    tfidf_mock = MagicMock()
    embedding_mock = MagicMock()

    keyword_mock.classify.return_value = keyword_result or make_classifier_result("biology", 0.8)
    tfidf_mock.classify.return_value = tfidf_result or make_classifier_result("biology", 0.75)
    embedding_mock.classify.return_value = embedding_result or make_classifier_result("biology", 0.9)

    return CompareService(keyword_mock, tfidf_mock, embedding_mock)


ABSTRACT = "This paper studies cell biology, genetics and DNA proteins."


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestCompareServiceInit:
    def test_keyword_matcher_stored(self):
        km = MagicMock()
        svc = CompareService(km, MagicMock(), MagicMock())
        assert svc.keyword_matcher is km

    def test_tfidf_classifier_stored(self):
        tc = MagicMock()
        svc = CompareService(MagicMock(), tc, MagicMock())
        assert svc.tfidf_classifier is tc

    def test_embedding_classifier_stored(self):
        ec = MagicMock()
        svc = CompareService(MagicMock(), MagicMock(), ec)
        assert svc.embedding_classifier is ec


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestCompareInputValidation:
    def setup_method(self):
        self.svc = build_service()

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Abstract must not be empty"):
            self.svc.compare("")

    def test_none_raises(self):
        with pytest.raises((ValueError, TypeError)):
            self.svc.compare(None)


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------


class TestCompareReturnStructure:
    def setup_method(self):
        self.svc = build_service()

    def test_returns_dict(self):
        assert isinstance(self.svc.compare(ABSTRACT), dict)

    def test_has_keyword_matching_key(self):
        assert "keyword_matching" in self.svc.compare(ABSTRACT)

    def test_has_tfidf_key(self):
        assert "tfidf" in self.svc.compare(ABSTRACT)

    def test_has_embedding_key(self):
        assert "embedding" in self.svc.compare(ABSTRACT)

    def test_each_section_has_predicted_category(self):
        result = self.svc.compare(ABSTRACT)
        for key in ("keyword_matching", "tfidf", "embedding"):
            assert "predicted_category" in result[key]

    def test_each_section_has_score(self):
        result = self.svc.compare(ABSTRACT)
        for key in ("keyword_matching", "tfidf", "embedding"):
            assert "score" in result[key]


# ---------------------------------------------------------------------------
# Delegation – all classifiers are called exactly once with the abstract
# ---------------------------------------------------------------------------


class TestCompareCallsDelegation:
    def test_all_classifiers_called_once(self):
        km, tc, ec = MagicMock(), MagicMock(), MagicMock()
        for mock in (km, tc, ec):
            mock.classify.return_value = make_classifier_result("biology", 0.5)

        svc = CompareService(km, tc, ec)
        svc.compare(ABSTRACT)

        km.classify.assert_called_once_with(ABSTRACT)
        tc.classify.assert_called_once_with(ABSTRACT)
        ec.classify.assert_called_once_with(ABSTRACT)

    def test_classifiers_receive_exact_abstract(self):
        km, tc, ec = MagicMock(), MagicMock(), MagicMock()
        specific_abstract = "A very specific abstract about quantum mechanics."
        for mock in (km, tc, ec):
            mock.classify.return_value = make_classifier_result("physics", 0.6)

        svc = CompareService(km, tc, ec)
        svc.compare(specific_abstract)

        km.classify.assert_called_once_with(specific_abstract)
        tc.classify.assert_called_once_with(specific_abstract)
        ec.classify.assert_called_once_with(specific_abstract)


# ---------------------------------------------------------------------------
# Predicted category pass-through
# ---------------------------------------------------------------------------


class TestComparePredictedCategory:
    def test_keyword_predicted_category_passed_through(self):
        svc = build_service(keyword_result=make_classifier_result("physics", 0.6))
        result = svc.compare(ABSTRACT)
        assert result["keyword_matching"]["predicted_category"] == "physics"

    def test_tfidf_predicted_category_passed_through(self):
        svc = build_service(tfidf_result=make_classifier_result("chemistry", 0.7))
        result = svc.compare(ABSTRACT)
        assert result["tfidf"]["predicted_category"] == "chemistry"

    def test_embedding_predicted_category_passed_through(self):
        svc = build_service(embedding_result=make_classifier_result("biology", 0.9))
        result = svc.compare(ABSTRACT)
        assert result["embedding"]["predicted_category"] == "biology"

    def test_none_predicted_category_passed_through(self):
        no_match = {"predicted_category": None, "scores": [{"label": "biology", "score": 0.0}]}
        svc = build_service(keyword_result=no_match)
        result = svc.compare(ABSTRACT)
        assert result["keyword_matching"]["predicted_category"] is None


# ---------------------------------------------------------------------------
# Score extraction logic
# ---------------------------------------------------------------------------


class TestCompareScoreExtraction:
    def test_positive_score_extracted_from_keyword(self):
        svc = build_service(keyword_result=make_classifier_result("biology", 0.8))
        result = svc.compare(ABSTRACT)
        assert result["keyword_matching"]["score"] == 0.8

    def test_positive_score_extracted_from_tfidf(self):
        svc = build_service(tfidf_result=make_classifier_result("physics", 0.65))
        result = svc.compare(ABSTRACT)
        assert result["tfidf"]["score"] == 0.65

    def test_positive_score_extracted_from_embedding(self):
        svc = build_service(embedding_result=make_classifier_result("chemistry", 0.92))
        result = svc.compare(ABSTRACT)
        assert result["embedding"]["score"] == 0.92

    def test_zero_score_returns_none_for_keyword(self):
        no_match = {"predicted_category": None, "scores": [{"label": "biology", "score": 0.0}]}
        svc = build_service(keyword_result=no_match)
        result = svc.compare(ABSTRACT)
        assert result["keyword_matching"]["score"] is None

    def test_zero_score_returns_none_for_tfidf(self):
        no_match = {"predicted_category": None, "scores": [{"label": "biology", "score": 0.0}]}
        svc = build_service(tfidf_result=no_match)
        result = svc.compare(ABSTRACT)
        assert result["tfidf"]["score"] is None

    def test_zero_score_returns_none_for_embedding(self):
        no_match = {"predicted_category": None, "scores": [{"label": "biology", "score": 0.0}]}
        svc = build_service(embedding_result=no_match)
        result = svc.compare(ABSTRACT)
        assert result["embedding"]["score"] is None

    def test_empty_scores_list_returns_none_score(self):
        empty_result = {"predicted_category": None, "scores": []}
        svc = build_service(keyword_result=empty_result)
        result = svc.compare(ABSTRACT)
        assert result["keyword_matching"]["score"] is None


# ---------------------------------------------------------------------------
# Each classifier is independent
# ---------------------------------------------------------------------------


class TestCompareClassifiersAreIndependent:
    def test_keyword_match_tfidf_no_match(self):
        no_match = {"predicted_category": None, "scores": [{"label": "x", "score": 0.0}]}
        svc = build_service(
            keyword_result=make_classifier_result("biology", 0.6),
            tfidf_result=no_match,
        )
        result = svc.compare(ABSTRACT)
        assert result["keyword_matching"]["predicted_category"] == "biology"
        assert result["tfidf"]["predicted_category"] is None
        assert result["tfidf"]["score"] is None

    def test_all_no_match_returns_all_none(self):
        no_match = {"predicted_category": None, "scores": [{"label": "x", "score": 0.0}]}
        svc = build_service(
            keyword_result=no_match,
            tfidf_result=no_match,
            embedding_result=no_match,
        )
        result = svc.compare(ABSTRACT)
        for key in ("keyword_matching", "tfidf", "embedding"):
            assert result[key]["predicted_category"] is None
            assert result[key]["score"] is None

    def test_all_different_predictions(self):
        svc = build_service(
            keyword_result=make_classifier_result("biology", 0.5),
            tfidf_result=make_classifier_result("physics", 0.7),
            embedding_result=make_classifier_result("chemistry", 0.9),
        )
        result = svc.compare(ABSTRACT)
        assert result["keyword_matching"]["predicted_category"] == "biology"
        assert result["tfidf"]["predicted_category"] == "physics"
        assert result["embedding"]["predicted_category"] == "chemistry"
