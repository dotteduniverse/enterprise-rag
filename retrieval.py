import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from typing import List, Dict
from schemas import Citation

# Initialize the same embedding model from ingestion
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the Re-ranker (Cross-Encoder)
# This model reads the query AND the chunk together to score relevance
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Connect to our local ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def hybrid_search(query: str, collection_name: str = "enterprise_docs", top_k: int = 10) -> List[Dict]:
    """
    Step 1: Fetch candidates using both Dense (Vector) and Sparse (BM25) search.
    """
    collection = chroma_client.get_collection(name=collection_name)
    
    # --- 1. Dense Search (Vector) ---
    query_embedding = embedder.encode(query).tolist()
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Extract documents and metadata from dense results
    candidates = []
    for i in range(len(dense_results['documents'][0])):
        candidates.append({
            "text": dense_results['documents'][0][i],
            "source": dense_results['metadatas'][0][i]['source'],
            "chunk_id": dense_results['metadatas'][0][i]['chunk_id']
        })

    # --- 2. Sparse Search (BM25) ---
    # Get ALL documents from the collection to build the BM25 index
    all_docs = collection.get()
    if all_docs['documents']:
        tokenized_corpus = [doc.lower().split() for doc in all_docs['documents']]
        bm25 = BM25Okapi(tokenized_corpus)
        
        tokenized_query = query.lower().split()
        # Get BM25 scores for all documents
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Get the top_k indices based on BM25 score
        top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        
        for idx in top_bm25_indices:
            if bm25_scores[idx] > 0: # Only add if there's a keyword match
                candidates.append({
                    "text": all_docs['documents'][idx],
                    "source": all_docs['metadatas'][idx]['source'],
                    "chunk_id": all_docs['metadatas'][idx]['chunk_id']
                })

    # Deduplicate candidates (since Dense and Sparse might return the same chunk)
    seen = set()
    unique_candidates = []
    for c in candidates:
        # Create a unique identifier based on source and chunk_id
        uid = f"{c['source']}_{c['chunk_id']}"
        if uid not in seen:
            seen.add(uid)
            unique_candidates.append(c)

    return unique_candidates

def rerank_and_filter(query: str, candidates: List[Dict], top_n: int = 3) -> List[Citation]:
    """
    Step 2: Use a Cross-Encoder to re-rank the top candidates and return the best N.
    """
    if not candidates:
        return []

    # Prepare pairs for the Cross-Encoder: [(query, text1), (query, text2)...]
    pairs = [[query, c["text"]] for c in candidates]
    
    # Get relevance scores
    scores = reranker.predict(pairs)
    
    # Attach scores to candidates
    for i, score in enumerate(scores):
        candidates[i]["score"] = float(score)
    
    # Sort by highest score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    # Take the top N and format as Pydantic Citations
    top_citations = []
    for c in candidates[:top_n]:
        top_citations.append(Citation(
            source_document=c["source"],
            chunk_text=c["text"]
        ))
        
    return top_citations