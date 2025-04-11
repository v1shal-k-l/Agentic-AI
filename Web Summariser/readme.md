# Webpage Summarizer using Groq Llama3 and HuggingFace Embeddings

This is a Streamlit web app that summarizes content from a given URL using:

- **Groq's Llama3-8B model** for text generation
- **HuggingFace's MiniLM embeddings** for semantic chunk retrieval
- **FAISS** for vector search and similarity matching
- **LangChain** for building the summarization pipeline

## Features

- Input any webpage URL and get a concise summary.
- Uses semantic search to extract relevant sections.
- Local embedding generation using `sentence-transformers/all-MiniLM-L6-v2`.
- LLM summarization powered by Groq's Llama3-8B model.
- Clean, responsive Streamlit UI.

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/v1shal-k-l/Agentic-AI.git
cd Agentic-AI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root with the following:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Make sure to never expose your API key in public repositories.

### 4. Run the app

```bash
streamlit run app.py
```

## File Structure

- `app.py` – Main Streamlit application
- `requirements.txt` – Python dependencies
- `.env` – Environment file (excluded from version control)

## Tech Stack

- **Streamlit** – Web interface
- **LangChain** – Prompting and chaining
- **FAISS** – Semantic vector search
- **sentence-transformers** – Embedding generation
- **Groq Llama3** – Fast and efficient LLM backend

## License

This project is licensed under the MIT License.
