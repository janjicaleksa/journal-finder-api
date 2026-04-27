from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Journal Finder API",
    description="API for scientific journal classification",
    version="1.0.0",
)

app.include_router(router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
