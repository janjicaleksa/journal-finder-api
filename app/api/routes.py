from fastapi import APIRouter, HTTPException

from app.config import JOURNAL_KEYWORDS, JOURNAL_DESCRIPTIONS, EMBEDDING_MODEL_NAME
from app.schemas import ClassificationRequest, ClassificationResponse, ClassificationScore
from app.schemas import ClassifierResult, CompareResponse
from app.services.keyword_classifier import KeywordMatcher
from app.services.tfidf_classifier import TfidfClassifier
from app.services.embedding_classifier import EmbeddingClassifier
from app.services.compare_service import CompareService

router = APIRouter()

keyword_matching_classifier = KeywordMatcher(JOURNAL_KEYWORDS)
tfidf_classifier = TfidfClassifier(JOURNAL_DESCRIPTIONS)
embedding_classifier = EmbeddingClassifier(JOURNAL_DESCRIPTIONS, EMBEDDING_MODEL_NAME)
compare_service = CompareService(keyword_matching_classifier, tfidf_classifier, embedding_classifier)

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

@router.post("/classify/embedding", response_model=ClassificationResponse)
def classify_embedding(request: ClassificationRequest):
    """Classify a paper abstract into a journal category using sentence embeddings."""
    try:
        results = embedding_classifier.classify(request.abstract)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    predicted_category = results["predicted_category"]
    scores = results["scores"]

    top_supporting_sentences = []
    if predicted_category is not None:
        top_supporting_sentences = embedding_classifier.get_top_supporting_sentences(
            abstract=request.abstract,
            predicted_category=predicted_category
        )

    return ClassificationResponse(
        method="sentence_embedding",
        predicted_category=predicted_category,
        scores=[ClassificationScore(label=s["label"], score=s["score"]) for s in scores],
        reasoning={
            "model": embedding_classifier.embedding_model_name,
            "top_supporting_sentences": top_supporting_sentences,
        }
    )

@router.post("/classify/compare", response_model=CompareResponse)
def classify_compare(request: ClassificationRequest):
    """Run all three classifiers on the same abstract and return their top prediction and score."""
    try:
        results = compare_service.compare(request.abstract)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "method": "compare",
        "results": results,
        "final_decision": {
            "predicted_category": results["embedding"]["predicted_category"],
            "selected_method": "embedding",
            "reason": "Embedding classifier captures semantic similarity better than lexical matching and TF-IDF methods."
        }
    }