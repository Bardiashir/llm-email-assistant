📧 LLM Email Assistant

A simple GenAI project that uses embeddings, FAISS vector search, and an LLM to perform:

Semantic email search

Reply generation

Email autocomplete

Built with Python, FastAPI, OpenAI API, and FAISS.

🚀 Features
🔍 1. Semantic Search (Embeddings + FAISS)

Ask natural language questions like:

“When is the next meeting?”
The system finds relevant emails using vector similarity and summarizes the answer with an LLM.

✉️ 2. Reply Suggestions

Paste any email text and the model generates a short, professional reply.

✏️ 3. Email Autocomplete

Start writing an email and the assistant continues the next 1–3 sentences naturally.

🧠 Tech Used

FastAPI – backend API

OpenAI API – embeddings + LLM generation

FAISS – vector database for semantic search

Python / Pandas / NumPy

Prompt Engineering for reply tone + autocomplete

📂 Project Structure
llm-email-assistant/
│
├─ app/
│  ├─ main.py              # FastAPI endpoints
│  ├─ rag_pipeline.py      # Retrieval + generation logic
│  ├─ embeddings_store.py  # FAISS search + embedding calls
│  ├─ prompts.py           # Prompt templates
│  └─ models.py            # Request/response models
│
├─ scripts/
│  └─ build_index.py       # Builds embeddings + FAISS index
│
├─ data/
│  └─ emails.csv           # Sample dataset (user-provided)
│
├─ requirements.txt
|
└─ README.md

⚙️ Setup
1. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# or
source venv/bin/activate       # Mac/Linux

2. Install dependencies
pip install -r requirements.txt

3. Add your OpenAI API key

Create a .env file in the project root:

OPENAI_API_KEY=your_api_key_here

4. Build the FAISS index
python scripts/build_index.py

5. Run the API
uvicorn app.main:app --reload


Open the interactive docs:
👉 http://127.0.0.1:8000/docs

📝 Endpoints
POST /search

Semantic search + LLM answer based on retrieved emails.

POST /suggest-reply

Generates a reply to a given email.

POST /autocomplete

Continues a partial email draft.

🧩 Future Improvements

Real Gmail/Outlook integration

Front-end UI (React / Streamlit)

Multi-user email stores

Better chunking + hybrid search
