import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Define summarization function
def summarize_url(url: str) -> str:
    # Step 1: Load webpage content
    loader = WebBaseLoader(web_paths=(url,))
    documents = loader.load()
    documents = [doc for doc in documents if doc.page_content.strip()]
    print(f"✅ Loaded {len(documents)} documents from URL.")

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"🧩 Split into {len(chunks)} chunks.")

    if not chunks:
        return "❌ No content found to summarize."

    # Step 3: Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 4: Vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # Step 5: LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="Llama3-8b-8192"
    )

    # Step 6: Prompt
    prompt = ChatPromptTemplate.from_template("""
    You are an expert summarizer. Summarize the webpage content below in a clear, concise way.
    
    <context>
    {context}
    </context>
    
    Question: Summarize the above content.
    """)

    # Step 7: Create chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Step 8: Invoke
    response = retrieval_chain.invoke({
        "input": "Summarize the entire content."
    })

    return response["answer"]

# Example usage
if __name__ == "__main__":
    url_input = "https://www.britannica.com/biography/D-B-Cooper"
    summary = summarize_url(url_input)
    print("\n📝 Summary:\n")
    print(summary)
