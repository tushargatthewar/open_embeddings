import sys
import pysqlite3  # Forces use of modern SQLite
sys.modules["sqlite3"] = pysqlite3
import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
#from chromadb.config import Settings
import uuid

# -----------------------------
# Initialize ChromaDB
# -----------------------------
import chromadb
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection("pdf_documents")

# -----------------------------
# Load SentenceTransformer Model
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Utility Functions
# -----------------------------
def read_pdf(file) -> list:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    texts = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            texts.append(text)
    return texts

def chunk_text(text, chunk_size=500):
    """Simple chunking by word count."""
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def embed_and_store(text_chunks, file_name):
    embeddings = model.encode(text_chunks).tolist()
    ids = [f"{file_name}_{uuid.uuid4()}" for _ in range(len(text_chunks))]
    metadatas = [{"source": file_name}] * len(text_chunks)
    collection.add(
        documents=text_chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

def search_query(query_text, top_k=3):
    query_embedding = model.encode([query_text]).tolist()[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PDF Semantic Search", layout="wide")
st.title("📄 PDF Semantic Search with ChromaDB + Sentence Transformers")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    with st.spinner("Reading and processing PDF..."):
        all_texts = read_pdf(uploaded_file)
        full_text = " ".join(all_texts)
        chunks = chunk_text(full_text)
        embed_and_store(chunks, uploaded_file.name)
        st.success(f"Processed and stored {len(chunks)} chunks from '{uploaded_file.name}'.")

# Search
st.markdown("### 🔍 Search in uploaded documents")
query = st.text_input("Enter your question or search term:")

if query:
    with st.spinner("Searching..."):
        results = search_query(query, top_k=5)
        st.markdown("### 📚 Top Matching Results:")
        for doc, score, metadata in zip(results['documents'][0], results['distances'][0], results['metadatas'][0]):
            st.markdown(f"**Score:** {score:.4f}")
            st.markdown(f"**Source:** {metadata['source']}")
            st.markdown(doc)
            st.markdown("---")
