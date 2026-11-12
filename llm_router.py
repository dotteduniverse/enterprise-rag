import os
from openai import OpenAI
from dotenv import load_dotenv
from schemas import Provider

load_dotenv()

def get_llm_client(provider: Provider) -> tuple[OpenAI, str]:
    """
    Returns a tuple of (OpenAI Client, Model Name)
    """
    if provider == Provider.GROQ:
        api_key = os.getenv("GROQ_API_KEY")
        base_url = "https://api.groq.com/openai/v1"
        model_name = "openai/gpt-oss-120b"
    elif provider == Provider.CEREBRAS:
        api_key = os.getenv("CEREBRAS_API_KEY")
        base_url = "https://api.cerebras.ai/v1"
        model_name = "llama3.3-70b"
    else:
        raise ValueError(f"Unknown provider: {provider}")

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model_name

def generate_answer(prompt: str, provider: Provider) -> str:
    client, model = get_llm_client(provider)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0 # Determinism for RAG
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM Error ({provider}): {str(e)}"