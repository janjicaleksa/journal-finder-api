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