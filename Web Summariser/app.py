import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit setup
st.set_page_config(page_title="Summarising Web Content", layout="wide")
st.title("Webpage Summarizer")
st.markdown("Enter a URL and get a concise summary.")

# Sidebar input
url = st.sidebar.text_input("Enter the webpage URL", placeholder="https://example.com")

# Summarization logic
def summarize_url(url: str) -> str:
    # Load page content
    loader = WebBaseLoader(web_paths=(url,))
    documents = loader.load()
    documents = [doc for doc in documents if doc.page_content.strip()]
    if not documents:
        return "No content found to summarize."

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Embed chunks
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # LLM setup
    llm = ChatGroq(model_name="Llama3-8b-8192", groq_api_key=groq_api_key)

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are a professional summarizer. Summarize the following webpage content clearly and concisely.
    Dont give random summarise be user friendly 
    
    <context>
    {context}
    </context>
    
    Question: Summarize the above content.
    """)

    # Chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Run
    response = retrieval_chain.invoke({"input": "Summarize the entire content."})
    return response["answer"]

# Main action
if st.sidebar.button("📝 Summarize"):
    if not url:
        st.warning("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Summarizing..."):
                summary = summarize_url(url)
            st.success("Summary generated!")
            st.subheader(" Summary:")
            st.write(summary)
        except Exception as e:
            st.error(f" Error: {e}")

# Footer
st.markdown("---\n Powered by HuggingFace Embeddings, FAISS & Groq Llama3")
