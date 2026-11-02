import json
import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

EMBEDDING_MODEL = "nvidia/nemotron-3.5-embed-1b"
BASE_URL = "https://integrate.api.nvidia.com/v1"

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("API_KEY")

client = OpenAI(base_url=BASE_URL, api_key=api_key)

doc_path = Path(__file__).with_name("source_document.txt")
if not doc_path.exists():
  raise FileNotFoundError(
      f"Could not find {doc_path}. Please create a 'source_document.txt' file!"
  )

with open(doc_path, "r", encoding="utf-8") as f:
  raw_text = f.read()


def chunk_text(text: str) -> list[str]:
    """Splits raw text into natural chunks based on paragraphs (double newlines)."""
    # Split by double newline and clean up empty spaces
    chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    return chunks

# 1. Automatically chunk the raw document using your function
raw_chunks = chunk_text(raw_text)
print(f"Split raw document into {len(raw_chunks)} chunks.")

# 2. Format chunks into the structure your code already uses
lessons = []
for i, chunk in enumerate(raw_chunks):
  lessons.append({
      "title": f"Document Chunk {i + 1}",
      "url": "Local Source File",
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


def cosine_similarity(a,b) -> float:
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 2. Pre-compute embeddings for all your lessons when the script starts
print("Generating embeddings for lesson dataset...")
for lesson in lessons:
    lesson["embedding"] = get_embedding(lesson["content"], input_type="passage")
print("All lesson embeddings ready!\n")

def search_lessons(query:str, top_k:int=3) -> list[dict]:
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
    # Sort by score in descending order
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results[:top_k]

if __name__ == "__main__":
    while True:
        query = input("Search Query (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break
        if not query.strip():
            continue
            
        results = search_lessons(query, top_k=2)
        
        print(f"\nTop Results for: '{query}'\n" + "-"*40)
        for i, res in enumerate(results, 1):
            print(f"{i}. {res['title']} (Score: {res['score']:.4f})")
            print(f"   Link: {res['url']}")
            print(f"   Snippet: {res['content'][:120]}...\n")