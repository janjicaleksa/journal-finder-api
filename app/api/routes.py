from fastapi import APIRouter, HTTPException

from app.config import JOURNAL_KEYWORDS
from app.schemas import ClassificationRequest, ClassificationResponse, ClassificationScore
from app.services.keyword_classifier import KeywordMatcher

router = APIRouter()

keyword_matching_classifier = KeywordMatcher(JOURNAL_KEYWORDS)

@router.post("/classify/keyword-matching", response_model=ClassificationResponse)
def classify_keyword_matching(request: ClassificationRequest):
    """Classify a paper abstract into a journal category using keyword matching method."""
    try:
        results = keyword_matching_classifier.classify(request.abstract)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if results and results[0]["score"] <= 0.0:
        raise HTTPException(status_code=400, detail="No keywords found in the abstract")
    
    predicted = results[0]["label"]

    return ClassificationResponse(
        method="keyword_matching",
        predicted_class=predicted,
        scores=[
            ClassificationScore(label=r["label"], score=r["score"])
            for r in results
        ]
    )