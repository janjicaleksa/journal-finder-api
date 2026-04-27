# 🧠 Journal Finder API

FastAPI-based service for recommending the most relevant academic journal based on a given paper abstract.

This project implements and compares multiple text similarity approaches:

* Keyword-based matching
* TF-IDF + cosine similarity
* Embedding-based semantic similarity

---

## 🚀 Problem Statement

Given a paper abstract, the goal is to identify the most relevant academic journal.

This can be framed as:

* a **text classification problem**, or
* a **semantic similarity / retrieval task**

In this implementation, we approach it as a **ranking problem**, where each method scores candidate journals and returns the best match.

---

## 🏗️ Architecture Overview

The system is built as a lightweight API service using:

* FastAPI for serving predictions
* Modular service layer for different similarity approaches
* Precomputed embeddings for efficient inference

### Pipeline

1. Input abstract is received via API
2. Each classifier computes similarity scores:
   * Keyword matching
   * TF-IDF vector similarity
   * Embedding similarity
3. Results are ranked
4. Final prediction is returned

---

## ⚙️ Implemented Methods

### 1. Keyword-Based Matching

* Simple rule-based approach
* Matches predefined keywords per journal
* Fast but brittle

✅ Pros:

* Interpretable
* No dependencies

❌ Cons:

* Poor generalization
* Sensitive to wording

---

### 2. TF-IDF + Cosine Similarity

* Converts text into sparse vectors
* Measures lexical similarity

✅ Pros:

* Lightweight
* Strong baseline

❌ Cons:

* Ignores semantics
* Fails on paraphrasing

---

### 3. Embedding-Based Similarity (Primary Method)

* Uses transformer-based embeddings via sentence-transformers
* Computes semantic similarity between abstract and journal descriptions

✅ Pros:

* Captures semantic meaning
* Robust to paraphrasing
* Best overall performance

❌ Cons:

* Heavier dependencies (e.g. PyTorch)
* Higher latency

---

## 🌐 API Endpoints

### Health Check

```
GET /health
```

---

All endpoints accept a JSON body: `{"abstract": "your paper abstract here"}`

### Classify (Individual Methods)

```
POST /classify/keyword-matching
POST /classify/tfidf
POST /classify/embedding
```

---

### Compare Methods

```
POST /classify/compare
```

Returns predictions from all methods for analysis and comparison.

---

## 📁 Project Structure

```
journal-finder-api/
├── app/
│   ├── api/
│   │   └── routes.py          # API endpoint definitions
│   ├── services/
│   │   ├── keyword_classifier.py   # Keyword-based classifier
│   │   ├── tfidf_classifier.py     # TF-IDF classifier
│   │   ├── embedding_classifier.py # Sentence-embedding classifier
│   │   └── compare_service.py      # Runs all three and compares results
│   ├── config.py              # Journal keywords and descriptions
│   ├── schemas.py             # Pydantic request/response models
│   └── main.py                # FastAPI app entry point
├── tests/
│   ├── conftest.py            # Shared test fixtures and mocks
│   ├── test_api.py            # API integration tests
│   ├── test_keyword_classifier.py
│   ├── test_tfidf_classifier.py
│   ├── test_embedding_classifier.py
│   └── test_compare_service.py
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## 📦 Installation & Setup

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/)

Install dependencies:

```bash
uv sync
```

Run the API:

```bash
uv run uvicorn app.main:app
```

Once running, the interactive API docs are available at:

* Swagger UI: http://localhost:8000/docs

---

## 🔁 Example Usage

Classify an abstract using the TF-IDF method:

```bash
curl -X POST http://localhost:8000/classify/tfidf \
  -H "Content-Type: application/json" \
  -d '{"abstract": "This paper studies neural networks and deep learning for natural language processing."}'
```

Compare all three methods at once:

```bash
curl -X POST http://localhost:8000/classify/compare \
  -H "Content-Type: application/json" \
  -d '{"abstract": "This paper studies neural networks and deep learning for natural language processing."}'
```

---

## 🐳 Docker

Build and run:

```bash
docker compose up --build
```

---

## 🧪 Testing

Run tests with:

```bash
uv run pytest
```

---

## 📊 Design Decisions

* Treated the problem as **ranking instead of strict classification**
* Implemented multiple approaches to compare trade-offs
* Precomputed embeddings for performance optimization
* Kept API stateless and modular

---

## ⚠️ Limitations

* No labeled dataset → no quantitative evaluation (accuracy, F1, etc.)
* Journal definitions are simplified and manually defined
* No confidence thresholding or fallback logic between methods
* Not optimized for large-scale retrieval

---

## 🔮 Future Improvements

* Add evaluation dataset and metrics (Top-K accuracy, MRR)
* Introduce confidence-based routing between methods
* Replace static journal definitions with real-world data
* Optimize embeddings (caching, batching, or external services)
* Add logging, monitoring, and production readiness features

---

## 🧠 Key Takeaways

* Embedding-based approaches outperform lexical methods for semantic tasks
* Simpler methods still provide strong baselines
* Trade-offs between accuracy, latency, and complexity are crucial in real systems

---

## Author's (not AI generated) NOTEL (note + novel)

The goal of this task was to classify scientific paper abstracts into predefined research domains.

I approached this as a text classification problem, but instead of jumping directly to a single model, I wanted to explore multiple levels of NLP complexity — from simple keyword matching to semantic similarity using embeddings.

I started with a simple **keyword-based** approach as a baseline.
This method is fully interpretable and helps understand how well simple lexical matching performs on this problem.

Each journal topic is represented by a predefined set of keywords (AI generated) and the score is computed based on how many of those keywords appear in the abstract.

However, this approach can lead to incorrect scoring in cases where different topics share common terms or when a keyword appears as part of another word without matching the intended context or when the keyword sets are imbalanced. That's the reason why I moved to the implementation of **TF-IDF** based classifier.

I represented both the abstract and journal description (AI generated) in a vector space using TF-IDF. I used both unigrams and bigrams to capture not only individual terms but also phrases such as "deep learning" or "quantum field". Similarity between the abstract and each journal category is computed using cosine similarity.

To improve performance, the TF-IDF vectorizer is fitted once during application startup and reused for all incoming requests.

To make the TF-IDF approach more interpretable, I added a reasoning step that extracts the top contributing terms for the predicted journal category. These are the terms that had the highest impact on the similarity score.

This approach does not fully capture longer expressions (e.g. "natural language processing"), but including higher-order n-grams such as trigrams or fourgrams would significantly increase the feature space and could lead to overfitting, especially given the small number of journal categories.

Finally, I implemented an **embedding**-based classifier using a sentence-transformer model.

This approach compares the abstract with natural-language descriptions of each journal category in a dense vector space, allowing it to capture semantic similarity even when exact keywords are not present.

Since embedding models are not inherently interpretable, I added a lightweight reasoning step by ranking the abstract sentences according to their similarity to the predicted journal category. This is not a full model explanation, but it helps identify which parts of the abstract most strongly support the final prediction.

The project is structured with a clear separation between API routes, business logic (services), configuration and data schemas.

Each classifier is implemented as an independent service making the system easy to extend with additional methods in the future.

The project includes unit and API tests (AI generated) to validate the behavior of each classifier and endpoint.

Docker is used to ensure reproducibility and simplify local setup.

**Last but not least**, no labeled dataset was provided, so I didn't perform quantitative evaluation (e.g. accuracy or F1 score). Instead, the focus was on building a flexible and extensible classification pipeline and comparing different approaches qualitatively.

I hope that the reader will have at least half of the fun I had while working on this assignment.

---

## 👤 Author

Aleksa Janjić