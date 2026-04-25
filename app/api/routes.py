from fastapi import APIRouter, HTTPException

from app.config import JOURNAL_KEYWORDS, JOURNAL_DESCRIPTIONS
from app.schemas import ClassificationRequest, ClassificationResponse, ClassificationScore
from app.services.keyword_classifier import KeywordMatcher
from app.services.tfidf_classifier import TfidfClassifier

router = APIRouter()

keyword_matching_classifier = KeywordMatcher(JOURNAL_KEYWORDS)
tfidf_classifier = TfidfClassifier(JOURNAL_DESCRIPTIONS)

@router.post("/classify/keyword-matching", response_model=ClassificationResponse)
def classify_keyword_matching(request: ClassificationRequest):
    """Classify a paper abstract into a journal category using keyword matching method."""
    try:
        results = keyword_matching_classifier.classify(request.abstract)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    predicted_category = results["predicted_category"]
    scores = results["scores"]

    if predicted_category is None:
        raise HTTPException(status_code=400, detail="No keywords found in the abstract")

    return ClassificationResponse(
        method="keyword_matching",
        predicted_category=predicted_category,
        scores=[ClassificationScore(label=s["label"], score=s["score"]) for s in scores],
        reasoning={
            "matched_keywords": {
                s["label"]: s["matched_keywords"]
                for s in scores
            }
        }
    )

@router.post("/classify/tfidf", response_model=ClassificationResponse)
def classify_tfidf(request: ClassificationRequest):
    """Classify a paper abstract into a journal category using TF-IDF method."""
    try:
        results = tfidf_classifier.classify(request.abstract)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    predicted_category = results["predicted_category"]
    scores = results["scores"]

    top_matching_terms = []
    if predicted_category is not None:
        top_matching_terms = tfidf_classifier.get_top_matching_terms(
            abstract_vector=results["abstract_vector"],
            predicted_category=predicted_category
        )

    return ClassificationResponse(
        method="TF-IDF_with_cosine_similarity",
        predicted_category=predicted_category,
        scores=[ClassificationScore(label=s["label"], score=s["score"]) for s in scores],
        reasoning={
            "ngram_range": [1, 2],
            "top_matching_terms": top_matching_terms,
        }
    )