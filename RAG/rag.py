import os
from typing import Literal
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


EMBEDDING_MODEL = "nvidia/nemotron-3.5-embed-1b"
MODEL = "meta/muse-glimmer-30b"
BASE_URL = "https://integrate.api.nvidia.com/v1"

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("API_KEY")

client = OpenAI(base_url=BASE_URL, api_key=api_key)

# Load the PDF file
pdf_path = Path(__file__).resolve().parent / "Knowledge" / "zentora_test_corpus.pdf"
if not pdf_path.exists():
    raise FileNotFoundError(f"Could not find {pdf_path}. Please place 'zentora_test_corpus.pdf' in your folder!")

# Read the PDF file and extract text
reader = PdfReader(pdf_path)
raw_text = ""
for page_num, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += f"\n--- Page {page_num + 1} ---\n" + text

# def chunk_text(text: str) -> list[str]:
#     """Splits raw text into smaller, manageable chunks by lines or paragraphs."""
#     # Split by single or double newlines to catch smaller sections
#     lines = text.split("\n")
#     chunks = []
#     current_chunk = ""

#     for line in lines:
#         if line.strip().startswith("--- Page") or not line.strip():
#             continue  # Skip page headers and empty lines for a moment
        
#         # Group lines together until they form a decent paragraph
#         if len(current_chunk) < 300:  # Adjust character limit per chunk
#             current_chunk += " " + line.strip()
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = line.strip()
            
#     if current_chunk:
#         chunks.append(current_chunk.strip())
        
#     return [c for c in chunks if c]



# Chunking function to split text into overlapping segments
def chunk_text(text: str, chunk_size: int = 350, overlap: int = 75) -> list[str]:
    """Splits text into sliding chunks with a defined character size and overlap."""
    # 1. Clean up page headers and empty lines from the raw text first
    lines = [line.strip() for line in text.split("\n") if line.strip() and not line.strip().startswith("--- Page")]
    full_clean_text = " ".join(lines)
    
    chunks = []
    start = 0
    text_length = len(full_clean_text)
    
    while start < text_length:
        # Define the end point of the current chunk
        end = start + chunk_size
        chunk = full_clean_text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
            
        # Move the starting point forward, but step back by 'overlap' characters
        start += chunk_size - overlap
        
    return chunks

raw_chunks = chunk_text(raw_text)
print(f"Extracted and split PDF into {len(raw_chunks)} chunks.")

# Create lessons from the chunks
lessons = []
for i, chunk in enumerate(raw_chunks,start=1):
    lessons.append({
        "title": f"PDF Chunk {i}",
        "url": "zentora_test_corpus.pdf",
        "content": chunk,
    })


def get_embedding(text: str, input_type: Literal["passage", "query"]) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
        encoding_format="float",
        extra_body={"input_type": input_type},
    )
    return response.data[0].embedding


def cosine_similarity(a, b) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Pre-compute embeddings for all your PDF chunks when the script starts
print("Generating embeddings for PDF dataset...")
for lesson in lessons:
    lesson["embedding"] = get_embedding(lesson["content"], input_type="passage")
print("All chunk embeddings ready!\n")


# Search function to find the most relevant chunks based on a query
def search_lessons(query: str, top_k: int = 3) -> list[dict]:
    """Searches through the PDF chunk embeddings using cosine similarity."""
    query_embedding = get_embedding(query, input_type="query")

    scored_results = []
    for lesson in lessons:
        score = cosine_similarity(query_embedding, lesson["embedding"])
        scored_results.append({
            "title": lesson["title"],
            "url": lesson["url"],
            "content": lesson["content"],
            "score": score
        })
    # Sort by score in descending order (highest similarity first)
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results[:top_k]

# Generate a grounded answer using the retrieved chunks
def generate_rag_answer(query: str, top_k: int = 2) -> tuple[str, list[dict]]:
    # 1. Retrieve relevant chunks using vector search
    retrieved_chunks = search_lessons(query, top_k=top_k)

    # 2. Format the retrieved context into a single text block
    context_text = "\n\n".join([f"Source ({res['title']}):\n{res['content']}" for res in retrieved_chunks])

    # 3. Construct a clear system prompt enforcing groundedness
    system_prompt = (
        "You are a helpful AI assistant. Answer the user's question accurately using **only** "
        "the provided context below. If the answer cannot be found in the context, state "
        "that you do not know based on the provided documents."
    )

    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

    # 4. Call the Chat Completion model using your MODEL variable
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=512,
    )

    answer = completion.choices[0].message.content
    return answer, retrieved_chunks

# --- Interactive RAG Loop ---
if __name__ == "__main__":
    while True:
        query = input("\n🔍 Ask a question about Zentora (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break
        if not query.strip():
            continue
            
        print(f"\n⏳ Thinking and retrieving context for: '{query}'...")
        answer, sources = generate_rag_answer(query, top_k=2)
        
        print("\n" + "="*60)
        print("🤖 AI ANSWER:")
        print(answer)
        print("="*60)
        
        print("\n📚 RETRIEVED SOURCES:")
        for i, res in enumerate(sources, 1):
            # Clean up newlines in the snippet preview so it prints neatly on one line
            clean_snippet = res['content'].replace('\n', ' ')
            print(f"  [{i}] {res['title']} | Similarity Score: {res['score']:.4f}")
            print(f"      Snippet: {clean_snippet[:120]}...\n")