---

# Webpage Summarizer using Groq Llama3 and HuggingFace Embeddings

This is a Streamlit web app that summarizes content from a given URL using:

- **Groq's Llama3-8B model** for text generation
- **HuggingFace's MiniLM embeddings** for semantic chunk retrieval
- **FAISS** for vector search and similarity matching
- **LangChain** for building the summarization pipeline

## Features

- Input any webpage URL and get a concise summary.
- Uses semantic search to extract relevant sections.
- Fully local embedding generation using `sentence-transformers/all-MiniLM-L6-v2`.
- LLM summarization using Groq's Llama3-8B model via the Groq API.
- Clean and simple Streamlit UI.

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/v1shal-k-l/Agentic-AI.git
cd webpage-summarizer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the app

```bash
streamlit run app.py
```

## File Structure

- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `.env` - Environment file to store Groq API key (not included in version control)

## Tech Stack

- **Streamlit** - Web UI
- **LangChain** - Chaining and prompt handling
- **FAISS** - In-memory vector search
- **sentence-transformers** - Text embedding
- **Groq Llama3** - Fast and powerful LLM via API

## License

This project is open-source and available under the MIT License.

---
