from fastapi import FastAPI
from pydantic import BaseModel
from pipeline_hf import build_context
from transformers import pipeline

# Initialize FastAPI
app = FastAPI(title="CustomGPT API")

# Define request model
class Query(BaseModel):
    question: str
    category: str = None
    year: str = None
    top_k: int = 5

# Load lightweight HuggingFace model (FLAN-T5)
qa_model = pipeline("text2text-generation", model="google/flan-t5-small")

# API endpoint
@app.post("/query")
def ask_customgpt(query: Query):
    context = build_context(query.question, query.category, query.year, query.top_k)
    if not context:
        return {"answer": "No relevant documents found."}

    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query.question}"

    answer = qa_model(prompt, max_length=512)
    return {"answer": answer[0]['generated_text']}
