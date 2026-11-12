from retrieval import hybrid_search, rerank_and_filter
from llm_router import generate_answer
from schemas import Provider

def ask_question(question: str):
    print(f"\n🔍 Question: {question}")
    
    # 1. Retrieve candidates (Hybrid Search)
    candidates = hybrid_search(query=question, top_k=10)
    print(f"Retrieved {len(candidates)} raw candidates.")
    
    # 2. Re-rank to the top 3
    citations = rerank_and_filter(query=question, candidates=candidates, top_n=3)
    
    if not citations:
        print("❌ No relevant context found in the PDF.")
        return

    # 3. Build the prompt with citations
    context_text = "\n\n".join([f"[Source: {c.source_document}]\n{c.chunk_text}" for c in citations])
    
    prompt = f"""
    Answer the question based ONLY on the context below. 
    If the answer is not in the context, say "I don't know."
    
    Context:
    {context_text}
    
    Question: {question}
    """
    
    # 4. Generate Answer using Groq
    print("🤖 Asking LLM...")
    answer = generate_answer(prompt=prompt, provider=Provider.GROQ)
    
    print("\n" + "="*40)
    print(f"✅ ANSWER:\n{answer}")
    print("="*40)
    print("\n📚 SOURCES USED:")
    for c in citations:
        print(f"- {c.source_document}: {c.chunk_text[:100]}...")

if __name__ == "__main__":
    # Replace this with a question relevant to the PDF you ingested!
    ask_question("What is the main topic of this document?")