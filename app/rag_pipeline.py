
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from .embeddings_store import EmbeddingsStore
from .prompts import (
    SEARCH_SYSTEM_PROMPT,
    REPLY_SYSTEM_PROMPT,
    AUTOCOMPLETE_SYSTEM_PROMPT,
)

load_dotenv()

# Initialize OpenAI client (uses OPENAI_API_KEY from env)
client = OpenAI()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "email_index.faiss"
EMAILS_PATH = PROJECT_ROOT / "data" / "emails.csv"

emb_store = EmbeddingsStore(
    index_path=INDEX_PATH,
    emails_csv_path=EMAILS_PATH,
)


def _format_context(emails: List[Dict[str, Any]]) -> str:
    chunks = []
    for i, e in enumerate(emails, start=1):
        chunks.append(
            f"[Email {i}]\nSubject: {e['subject']}\nBody: {e['body']}\n"
        )
    return "\n".join(chunks)


def answer_question_about_emails(query: str, top_k: int = 5) -> Dict[str, Any]:
    similar_emails = emb_store.search(query, k=top_k)
    context = _format_context(similar_emails)

    messages = [
        {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context emails:\n{context}\n\n"
                f"User question: {query}\n\n"
                "Answer using only the context above in 2–4 sentences."
            ),
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )

    answer_text = response.choices[0].message.content.strip()

    return {
        "answer": answer_text,
        "results": similar_emails,
    }


def suggest_reply(email_text: str) -> str:
    messages = [
        {"role": "system", "content": REPLY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Here is the email I received:\n\n{email_text}\n\nWrite a reply:",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )

    return response.choices[0].message.content.strip()


def autocomplete_email(partial_text: str) -> str:
    messages = [
        {"role": "system", "content": AUTOCOMPLETE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Current email draft:\n\n{partial_text}\n\nContinue the email:",
        },
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )

    return response.choices[0].message.content.strip()
