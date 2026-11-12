from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Provider(str, Enum):
    GROQ = "groq"
    CEREBRAS = "cerebras"

class RAGRequest(BaseModel):
    question: str = Field(..., min_length=5, description="The user's question.")
    provider: Provider = Field(default=Provider.GROQ, description="Which LLM provider to use.")
    top_k: int = Field(default=3, le=10, description="How many chunks to retrieve.")

class Citation(BaseModel):
    source_document: str = Field(..., description="The name of the file the answer came from.")
    chunk_text: str = Field(..., description="The exact text retrieved.")

class RAGResponse(BaseModel):
    answer: str = Field(..., description="The LLM's generated answer.")
    citations: List[Citation] = Field(..., description="The sources used to answer the question.")
    provider_used: str