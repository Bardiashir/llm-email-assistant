
from typing import List
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class EmailSnippet(BaseModel):
    id: int
    subject: str
    body: str
    score: float


class SearchResponse(BaseModel):
    answer: str
    results: List[EmailSnippet]


class ReplyRequest(BaseModel):
    email_text: str


class ReplyResponse(BaseModel):
    reply: str


class AutocompleteRequest(BaseModel):
    partial_text: str


class AutocompleteResponse(BaseModel):
    completion: str
