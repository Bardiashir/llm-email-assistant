
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import faiss


def load_api_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found in environment or .env file.")
    return OpenAI(api_key=api_key)


def build_embeddings(df: pd.DataFrame, client: OpenAI, model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Create embeddings for each email (subject + body).
    Returns a numpy array of shape (n_rows, embedding_dim).
    """
    texts = [
        f"Subject: {row['subject']}\nBody: {row['body']}"
        for _, row in df.iterrows()
    ]

    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        emb = response.data[0].embedding
        embeddings.append(emb)

    embeddings_array = np.array(embeddings, dtype="float32")
    return embeddings_array


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "emails.csv"
    index_path = project_root / "data" / "email_index.faiss"

    if not data_path.exists():
        raise FileNotFoundError(f"emails.csv not found at {data_path}")

    print(f"Loading emails from {data_path}...")
    df = pd.read_csv(data_path)

    client = load_api_client()
    print("Building embeddings...")
    embeddings = build_embeddings(df, client)

    print("Creating FAISS index...")
    index = build_faiss_index(embeddings)

    print(f"Saving index to {index_path}...")
    faiss.write_index(index, str(index_path))

    print("Done! ✅ FAISS index created and saved.")


if __name__ == "__main__":
    main()
