from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    abstract: str = Field(..., min_length=1)


class ClassificationScore(BaseModel):
    label: str
    score: float


class ClassificationResponse(BaseModel):
    method: str
    predicted_category: str | None
    scores: list[ClassificationScore]
    reasoning: dict | None = None


class ClassifierResult(BaseModel):
    predicted_category: str | None
    score: float | None


class FinalDecision(BaseModel):
    predicted_category: str | None
    selected_method: str
    reason: str


class CompareResponse(BaseModel):
    method: str
    results: dict[str, ClassifierResult]
    final_decision: FinalDecision
