import os
import psycopg2
from dotenv import load_dotenv
from docx import Document
import tiktoken
from sentence_transformers import SentenceTransformer

# --- Load environment variables ---
load_dotenv()

# --- Connect to Supabase ---
conn = psycopg2.connect(
    host=os.getenv("SUPABASE_HOST"),
    dbname=os.getenv("SUPABASE_DB"),
    user=os.getenv("SUPABASE_USER"),
    password=os.getenv("SUPABASE_PASSWORD"),
    port=os.getenv("SUPABASE_PORT")
)
cur = conn.cursor()

# --- HuggingFace embedding model ---
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

# --- Check if file already uploaded ---
def file_already_uploaded(filepath):
    cur.execute("SELECT 1 FROM documents WHERE filepath = %s LIMIT 1;", (filepath,))
    return cur.fetchone() is not None

# --- Insert into DB ---
def insert_document(category, year, content, filepath):
    embedding = embed_text(content)
    cur.execute(
        """
        INSERT INTO documents (category, year, content, embedding, filepath)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (category, year, content, embedding, filepath)
    )
    conn.commit()

# --- Process single docx file ---
def process_docx(filepath, category, year):
    if file_already_uploaded(filepath):
        print(f"⏩ Skipped (already uploaded): {filepath}")
        return

    try:
        doc = Document(filepath)
        full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        for chunk in chunk_text(full_text):
            insert_document(category, year, chunk, filepath)
        print(f"✅ Uploaded: {filepath}")
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")

# --- Process all years/folders under Workshops ---
def process_workshops(base_folder="documents/Workshops"):
    for category in os.listdir(base_folder):
        cat_path = os.path.join(base_folder, category)
        if not os.path.isdir(cat_path):
            continue
        for year in os.listdir(cat_path):
            year_path = os.path.join(cat_path, year)
            if not os.path.isdir(year_path):
                continue
            for root, dirs, files in os.walk(year_path):
                for f in files:
                    if f.endswith(".docx"):
                        process_docx(
                            os.path.join(root, f),
                            category=category,
                            year=year
                        )
    print("🚀 All missing Workshop files uploaded!")

if __name__ == "__main__":
    process_workshops()
