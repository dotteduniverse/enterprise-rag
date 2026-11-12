from fastapi import FastAPI, HTTPException
from schemas import RAGRequest, RAGResponse, Provider
from retrieval import hybrid_search, rerank_and_filter
from llm_router import generate_answer

app = FastAPI(title="Enterprise RAG API")

@app.post("/query", response_model=RAGResponse)
async def query_rag(request: RAGRequest):
    try:
        # 1. Retrieve and Re-rank
        candidates = hybrid_search(query=request.question, top_k=request.top_k)
        citations = rerank_and_filter(query=request.question, candidates=candidates, top_n=3)
        
        if not citations:
            return RAGResponse(
                answer="I don't know. No relevant context found.",
                citations=[],
                provider_used=request.provider.value
            )

        # 2. Build Prompt
        context_text = "\n\n".join([f"[Source: {c.source_document}]\n{c.chunk_text}" for c in citations])
        prompt = f"Answer the question based ONLY on this context:\n{context_text}\n\nQuestion: {request.question}"
        
        # 3. Generate Answer
        answer = generate_answer(prompt=prompt, provider=request.provider)
        
        return RAGResponse(
            answer=answer,
            citations=citations,
            provider_used=request.provider.value
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)