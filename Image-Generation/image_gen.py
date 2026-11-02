import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load your token from the .env file in the parent folder
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Initialize the Hugging Face client using the built-in free serverless provider
client = InferenceClient(api_key=os.getenv("HF_TOKEN"))

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

def generate_image(prompt: str, output_filename: str = "output.png"):
    print(f"Sending prompt to Hugging Face: '{prompt}'...")
    
    # Call the Hugging Face text-to-image API
    image = client.text_to_image(
        prompt=prompt,
        model=MODEL_ID
    )
    
    # Save the resulting image locally
    output_path = Path(__file__).with_name(output_filename)
    image.save(output_path)
    print(f"Success! Image saved to: {output_path.absolute()}")

if __name__ == "__main__":
    user_prompt = input("Enter a prompt for image generation: ")
    if user_prompt.strip():
        generate_image(user_prompt)