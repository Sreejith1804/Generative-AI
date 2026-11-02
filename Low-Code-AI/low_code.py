import os
from pathlib import Path
import json
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


def extract_dynamic_data(raw_text: str, target_fields: list[str]) -> str:
    """Dynamically extracts specified fields from raw text into JSON."""
    
    # Format the list of fields into a readable string for the prompt
    fields_str = ", ".join(target_fields)
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an automated data extraction engine. "
                "Read the user's text and extract the requested details. "
                f"You MUST output your response strictly as a valid JSON object containing ONLY these keys: [{fields_str}]. "
                "Do not include any conversational filler, markdown formatting blocks like ```json, or extra text."
            ),
        },
        {
            "role": "user", 
            "content": raw_text
        }
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.1,
            top_p=0.95,
            max_tokens=512,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"Error during API call: {e}")
        return "{}"

def parse_json_response(ai_response: str) -> dict:
    cleaned_text = ai_response.strip()
    
    # 1. Strip markdown code blocks if the model added them
    if cleaned_text.startswith("```json"):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.endswith("```"):
        cleaned_text = cleaned_text[:-3]
        
    cleaned_text = cleaned_text.strip()
    
    # 2. Safely convert the cleaned string into a Python dictionary
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse: {e}")
        return {}

def run_generic_automation_workflow(raw_text: str, target_fields: list[str], task_name: str):
    """Simulates a generic low-code workflow pipeline for any type of data."""
    
    print(f"--- Running Workflow: {task_name} ---")
    print("1. [Trigger] Input data received...")
    
    # Step A: Extract structured data dynamically
    raw_result = extract_dynamic_data(raw_text, target_fields)
    
    # Step B: Parse the string into a clean Python dictionary
    parsed_data = parse_json_response(raw_result)
    
    if not parsed_data:
        print("Workflow Failed: Could not parse data.")
        return

    print(f"2. [AI Extraction Success] Extracted Fields:")
    for key, value in parsed_data.items():
        print(f"   - {key}: {value}")

    print(f"3. [Workflow Action] Data successfully processed and routed for '{task_name}'.\n")


if __name__ == "__main__":
    # --- Scenario 1: Processing a Customer Support Ticket ---
    support_text = "Hi, my internet has been down for 2 hours in apartment 4B. My name is Alex."
    support_fields = ["CustomerName", "Issue", "Location", "Urgency"]
    
    run_generic_automation_workflow(support_text, support_fields, "Support Ticket Triage")


    # --- Scenario 2: Processing a Student Assignment Submission ---
    assignment_text = "Assignment submitted by Priya on Python Data Structures. Status: Completed."
    assignment_fields = ["StudentName", "Topic", "Status"]
    
    run_generic_automation_workflow(assignment_text, assignment_fields, "Assignment Tracker")