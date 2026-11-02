import os
from pathlib import Path
import json
from dotenv import load_dotenv
from openai import OpenAI

# Import our tools and registry from tools.py
from tools import student_tools_schemas, TOOL_REGISTRY

MODEL = "meta/muse-glimmer-30b"
BASE_URL = "https://integrate.api.nvidia.com/v1"

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

api_key = os.getenv("API_KEY")
if not api_key:
    raise RuntimeError(
        "NVIDIA API key not found. Set NVIDIA_API_KEY in .env or your environment."
    )

client = OpenAI(base_url=BASE_URL, api_key=api_key)

def ask_ai_with_tools(messages: list[dict]):
    """Sends a user prompt and available tools to the LLM."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=student_tools_schemas,  
        tool_choice="auto",        # Let the model decide whether to use the tool or just chat
    )
    
    return response.choices[0].message

def process_tool_execution(response_message, messages, max_rounds: int = 5):
    """Executes requested tools repeatedly until the LLM gives a final text answer."""

    rounds = 0 

    while response_message.tool_calls and rounds < max_rounds:
        rounds += 1  # 🆕 NEW
        print("\n[🤖 AI Action: Tool requested by model]")

        # Append the assistant's request to message history
        messages.append(response_message)

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name

            try:
                function_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                function_args = {}

            print(f" ⚙️  Executing function: {function_name}")
            print(f" 📦 Arguments: {function_args}")

            if function_name in TOOL_REGISTRY:
                tool_function = TOOL_REGISTRY[function_name]
                try:
                    function_result = tool_function(**function_args)
                except TypeError as e:
                    function_result = f"Error: bad arguments for {function_name}: {e}"
            else:
                function_result = f"Error: Function {function_name} not found."

            print(f" ✅ Output: {function_result}")

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(function_result),  
            })

        # Now we call ask_ai_with_tools, so the model can call MORE tools.
        response_message = ask_ai_with_tools(messages)

    final_text = response_message.content or ""
    if response_message.tool_calls:  # hit max_rounds and the model still wanted more
        final_text = "Sorry, I couldn't finish that request (too many tool calls)."

    # Save the final answer so follow-ups like "he" have context
    messages.append({"role": "assistant", "content": final_text})
    return final_text

# --- The Application Entry Point ---
if __name__ == "__main__":
    print("Type your questions (e.g., 'What is Sreejith's grade in Java?')")
    print("Type 'exit' to quit.\n")

    # 1. Initialize persistent chat history OUTSIDE the loop
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a helpful student record assistant. "
                "Use your tools when asked about grades, deadlines, assignments, or instructors. "
                "If a question needs several tools, call all of them before answering. "
                "If the user says he/she/they, use the student name from earlier in the conversation."
            )
        },
    ]
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            print("Exiting assistant session.")
            break
            
        if not user_input.strip():
            continue

        # 2. Append the new user input to the ongoing history
        messages.append(
            {
                "role": "user", 
                "content": user_input
            }
        )

        # 2. Call your first function to send prompt + tools to the LLM
        response_message = ask_ai_with_tools(messages)

        # 3. Call your second function to handle tool execution & final answer
        final_answer = process_tool_execution(response_message, messages)

        print(f"\nAssistant: {final_answer}\n" + "-" * 50)