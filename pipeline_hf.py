import os
from dotenv import load_dotenv
from docx import Document
import tiktoken
from sentence_transformers import SentenceTransformer
from supabase import create_client
import numpy as np

# --- Load environment variables ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- Connect to Supabase ---
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Load HuggingFace embedding model ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Split text into chunks ---
def chunk_text(text, max_tokens=500):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield enc.decode(tokens[i:i+max_tokens])

# --- Create embeddings ---
def embed_text(text):
    return model.encode(text).tolist()

# --- Check if file is already uploaded ---
def file_already_uploaded(filepath):
    resp = supabase.table("documents").select("filepath").eq("filepath", filepath).limit(1).execute()
    return bool(resp.data)

# --- Insert a chunk into Supabase ---
def insert_document(category, year, content, filepath):
    embedding = embed_text(content)
    supabase.table("documents").insert({
        "category": category,
        "year": year,
        "content": content,
        "embedding": embedding,
        "filepath": filepath
    }).execute()

# --- Process single .docx file ---
def process_docx(filepath, category, year):
    if file_already_uploaded(filepath):
        print(f"Skipped (already uploaded): {filepath}")
        return
    doc = Document(filepath)
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    for chunk in chunk_text(full_text):
        insert_document(category, year, chunk, filepath)
    print(f"✅ Inserted: {filepath}")

# --- Recursively process all files in folder ---
def process_folder(folder_path="documents"):
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.endswith(".docx"):
                rel_path = os.path.relpath(root, folder_path)
                parts = rel_path.split(os.sep)
                category = parts[0] if len(parts) > 0 else "Unknown"
                year = parts[1] if len(parts) > 1 else "Unknown"
                process_docx(os.path.join(root, f), category, year)
    print("✅ All files processed!")

# --- Search Supabase ---
def search(query, top_k=5):
    query_embedding = embed_text(query)
    resp = supabase.table("documents").select("*").execute()
    results = resp.data

    def similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for r in results:
        r["similarity"] = similarity(query_embedding, r["embedding"])

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]

# --- Ask CustomGPT ---
def ask(query):
    results = search(query)
    context = "\n".join([f"[{r['category']} - {r['year']}] {r['content']}" for r in results])
    return context
