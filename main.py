from fastapi import FastAPI
from pydantic import BaseModel
from pipeline_hf import ask, process_folder
from transformers import pipeline

# --- Initialize FastAPI ---
app = FastAPI(title="CustomGPT API")

# --- Request Model ---
class Query(BaseModel):
    question: str
    category: str = None
    year: str = None
    top_k: int = 5

# --- Load lightweight HuggingFace model (FLAN-T5) ---
qa_model = pipeline("text2text-generation", model="google/flan-t5-small")

# --- API endpoint to ask questions ---
@app.post("/query")
def ask_customgpt(query: Query):
    context = ask(query.question)  # Uses supabase search
    if not context:
        return {"answer": "No relevant documents found."}

    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {query.question}"
    answer = qa_model(prompt, max_length=512)
    return {"answer": answer[0]['generated_text']}

# --- API endpoint to process documents ---
@app.post("/process-docs")
def process_docs():
    process_folder("documents")
    return {"status": "All documents processed!"}
