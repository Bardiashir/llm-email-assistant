
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    SearchRequest,
    SearchResponse,
    ReplyRequest,
    ReplyResponse,
    AutocompleteRequest,
    AutocompleteResponse,
    EmailSnippet,
)
from .rag_pipeline import (
    answer_question_about_emails,
    suggest_reply,
    autocomplete_email,
)

app = FastAPI(
    title="LLM Email Assistant",
    description="Demo GenAI project: semantic email search, reply suggestions, and autocomplete.",
    version="0.1.0",
)

# Allow all origins for easy local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search_emails(payload: SearchRequest):
    result = answer_question_about_emails(
        query=payload.query,
        top_k=payload.top_k,
    )
    snippets = [
        EmailSnippet(
            id=e["id"],
            subject=e["subject"],
            body=e["body"],
            score=e["score"],
        )
        for e in result["results"]
    ]
    return SearchResponse(
        answer=result["answer"],
        results=snippets,
    )


@app.post("/suggest-reply", response_model=ReplyResponse)
def suggest_reply_endpoint(payload: ReplyRequest):
    reply_text = suggest_reply(payload.email_text)
    return ReplyResponse(reply=reply_text)


@app.post("/autocomplete", response_model=AutocompleteResponse)
def autocomplete_endpoint(payload: AutocompleteRequest):
    completion_text = autocomplete_email(payload.partial_text)
    return AutocompleteResponse(completion=completion_text)
