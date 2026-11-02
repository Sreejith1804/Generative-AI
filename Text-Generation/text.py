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

client = OpenAI(base_url=BASE_URL, api_key=api_key)


def generate_text(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        top_p=0.95,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    user_prompt = input("Prompt: ").strip()
    if not user_prompt:
        raise SystemExit("Enter a non-empty prompt.")
    print(generate_text(user_prompt))