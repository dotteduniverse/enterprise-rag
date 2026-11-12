import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from typing import List, Dict

# Initialize ChromaDB (persistent local storage)
# This will create a folder called 'chroma_db' in your project
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Use a free, fast local embedding model
# (all-MiniLM-L6-v2 maps text to a 384-dimensional vector)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path: str) -> str:
    """Reads a PDF and returns raw text."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of roughly `chunk_size` characters.
    `overlap` ensures we don't cut a sentence in half and lose context.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap # Move forward, but keep some overlap
    return chunks

def ingest_document(file_path: str, collection_name: str = "enterprise_docs"):
    """
    The main pipeline: Read -> Chunk -> Embed -> Store
    """
    print(f"Ingesting {file_path}...")
    
    # 1. Extract
    text = extract_text_from_pdf(file_path)
    if not text.strip():
        print("Error: No text found in PDF.")
        return

    # 2. Chunk
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")

    # 3. Embed & Store
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Prepare data for Chroma
    documents = []
    embeddings = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        embeddings.append(embedder.encode(chunk).tolist())
        metadatas.append({"source": os.path.basename(file_path), "chunk_id": i})
        ids.append(f"{os.path.basename(file_path)}_chunk_{i}")

    # Add to Chroma (upsert means if you run it twice, it updates instead of duplicating)
    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Successfully ingested {len(chunks)} chunks into '{collection_name}'.")

# --- For testing ---
if __name__ == "__main__":
    # We need a dummy PDF to test. 
    # If you don't have one, download any free PDF (e.g., an AI paper) and name it 'test.pdf'
    ingest_document("test.pdf") 
    print("Ingestion module ready.")