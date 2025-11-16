
import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import faiss
from dotenv import load_dotenv
from openai import OpenAI


class EmbeddingsStore:
    def __init__(
        self,
        index_path: Path,
        emails_csv_path: Path,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found in environment or .env file.")

        self.client = OpenAI(api_key=api_key)
        self.embedding_model = embedding_model

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        if not emails_csv_path.exists():
            raise FileNotFoundError(
                f"emails.csv not found at {emails_csv_path}")

        self.index = faiss.read_index(str(index_path))
        self.emails_df = pd.read_csv(emails_csv_path)

    def _embed_text(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        emb = np.array(response.data[0].embedding, dtype="float32")
        return emb.reshape(1, -1)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the top-k most similar emails to the query.
        Returns a list of dicts: {id, subject, body, score}.
        """
        query_emb = self._embed_text(query)
        distances, indices = self.index.search(query_emb, k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.emails_df):
                continue
            row = self.emails_df.iloc[idx]
            results.append(
                {
                    "id": int(row["id"]),
                    "subject": str(row["subject"]),
                    "body": str(row["body"]),
                    "score": float(dist),
                }
            )
        return results
