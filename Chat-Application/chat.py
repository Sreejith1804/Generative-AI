import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


MODEL = "meta/muse-glimmer-30b"
BASE_URL = "https://integrate.api.nvidia.com/v1"

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

api_key = os.getenv("API_KEY")
if not api_key:
    raise RuntimeError(
        "NVIDIA API key not found. Set NVIDIA_API_KEY in .env or your environment."
    )

client = OpenAI(base_url=BASE_URL, api_key=api_key)


def generate_text(message: list[dict], temperature: float = 0.7) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=message,
        temperature=temperature,
        top_p=0.95,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":

    chat_history = [
        {
        "role": "system",
        "content": "You are a helpful and concise AI assistant.",
        }
    ]

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Exiting chat session.")
            break
        
        if user_input.lower() == "/clear":
            chat_history = [
                {
                    "role": "system",
                    "content": "You are a helpful and concise AI assistant.",
                }
                ]
            print("\n--- Chat history cleared. Starting fresh! ---\n")
            continue  
        if not user_input.strip():
            continue

        chat_history.append({"role": "user", "content": user_input})
        ai_response = generate_text(chat_history)
        print(f"AI: {ai_response}\n")
        chat_history.append({"role": "assistant", "content": ai_response})

    