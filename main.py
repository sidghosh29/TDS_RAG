from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import requests
import os
import numpy as np
from numpy.linalg import norm
from typing import Optional, List

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set")

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class LinkResponse(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkResponse]


# Helper to get embedding from OpenAI using requests (manual approach)
def get_openai_embedding(text, api_key):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "text-embedding-3-small",
        "input": text
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def call_openai_api(messages: list, model: str, max_tokens: int = 500):
    """Generic OpenAI API call using requests"""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"OpenAI API error: {response.text}"
        )
    
    return response.json()['choices'][0]['message']['content']



# Load embeddings and metadata from npz file (do this once at startup)
NPZ_PATH = "embeddings.npz"  # Change this to your actual npz file path
npz_data = np.load(NPZ_PATH, allow_pickle=True)
embeddings = npz_data["embeddings"]  # shape: (N, D)
texts = npz_data["texts"]  # shape: (N,)
urls = npz_data["urls"] if "urls" in npz_data else None

@app.post("/api/", response_model=QueryResponse)
async def answer_question(request: QueryRequest):
    print(f"DEBUG: Received question: {request.question!r}")
    try:
        additional_context = ""
        # Handle image input using GPT-4 Vision
        if request.image:
            vision_messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract text from image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{request.image}"
                        }
                    }
                ]
            }]
            additional_context = call_openai_api(
                messages=vision_messages,
                model="gpt-4o",
                max_tokens=1000
            )
        # Combine question and image text
        full_query = f"{request.question}\n{additional_context}".strip()
        # Get embedding for the query
        query_embedding = get_openai_embedding(full_query, OPENAI_API_KEY)
        # Compute cosine similarity with all stored embeddings
        emb_array = np.array(embeddings)
        query_vec = np.array(query_embedding)
        # Normalize for cosine similarity
        emb_norm = emb_array / (norm(emb_array, axis=1, keepdims=True) + 1e-8)
        query_norm = query_vec / (norm(query_vec) + 1e-8)
        similarities = np.dot(emb_norm, query_norm)
        # Get top 3 most similar chunks
        top_k = 3
        top_indices = similarities.argsort()[-top_k:][::-1]
        retrieved_chunks = [str(texts[i]) for i in top_indices]
        retrieved_urls = [str(urls[i]) if urls is not None else "" for i in top_indices]
        # Prepare context for LLM
        context = "\n---\n".join(retrieved_chunks)
        # Compose prompt for LLM
        messages = [
            {"role": "system", "content": """You are a helpful and concise assistant designed to respond to user questions with clarity and precision. Use the **provided context** to generate answers. Your response will be **blindly posted on Discourse**, so **follow these rules strictly**:

### ✅ Response Format

* Use **Markdown** for formatting.
* Include:

  * Code blocks (` ``` `) where applicable.
  * Lists for steps or key points.
  * Headings for structured explanations.
* Be **brief and to the point**. Avoid unnecessary elaboration.

### 🚨 Rules

1. **Only use the context provided** to generate answers.
2. If you don't know the answer or the context is insufficient, **say: `I don't know.`**
3. When the user provides an abbreviation (e.g. **GA = Graded Assignment**), make sure you understand and use it accordingly.
4. Do **not** include any personal opinions, filler text, or unnecessary greetings.
5. Never reference yourself (e.g. "As an AI...") or mention that you are a language model."""},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {full_query}"}
        ]
        answer = call_openai_api(messages, model="gpt-4o", max_tokens=500)
        # Prepare links (if available)
        links = [LinkResponse(url=retrieved_urls[i], text=retrieved_chunks[i]) for i in range(top_k)]
        return QueryResponse(answer=answer, links=links)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

