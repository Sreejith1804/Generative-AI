import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


MODEL = "meta/muse-glimmer-30b"
BASE_URL = "https://integrate.api.nvidia.com/v1"

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("API_KEY")
if not api_key:
    raise RuntimeError(
        "NVIDIA API key not found. Set NVIDIA_API_KEY in .env or your environment."
    )

client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=120.0, max_retries=2)


def generate_text(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        top_p=0.95,
        max_tokens=4096,
    )
    if not response.choices or not response.choices[0].message.content:
        raise RuntimeError("The NVIDIA API returned an empty response.")
    return response.choices[0].message.content


if __name__ == "__main__":
    no_recipes = input("No of recipes (for example, 5): ")
    ingredients = input("List of ingredients (for example, chicken, potatoes, and carrots): ")
    diet = input("Ingredients to avoid (e.g., vegetables, onion): ")

    prompt = (
    f"Give me {no_recipes} South Indian recipe(s) using: {ingredients}. "
    f"Do not include these as ingredients: {diet}. "
    "Onion, garlic, ginger, tomato, curry leaves and green chilies are allowed for flavor. "
    "For each recipe, list all ingredients with quantities, then short numbered steps. "
    "Be concise."
    )
    if not prompt:
        raise SystemExit("Enter a non-empty prompt.")
    print("Generating recipes (waiting for NVIDIA API)...", flush=True)
    print(generate_text(prompt))