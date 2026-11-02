# Generative AI Projects

A collection of small Python projects exploring text generation, chat, function calling, image generation, data extraction, and document search/RAG.

## Projects

| Folder | Description | Main script |
|---|---|---|
| `Chat-Application` | Interactive chat with an OpenAI-compatible NVIDIA model | `chat.py` |
| `Function-Calling` | Chat assistant that can call local student-record tools | `function_call.py` |
| `Image-Generation` | Generates an image from a prompt using Hugging Face Inference | `image_gen.py` |
| `Low-Code-AI` | Extracts requested fields from text and parses the model response as JSON | `low_code.py` |
| `RAG` | Answers questions using retrieved chunks from a local PDF | `rag.py` |
| `Search-App` | Finds relevant chunks in a local text document | `search.py` |
| `Text-Generation` | General text generation and South Indian recipe generation | `text.py`, `chef_bot.py` |

## Requirements

- Python 3.10 or later
- API credentials for the services used by a project
- Internet access for model API requests

## Setup on Windows

Run these commands in PowerShell from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create a `.env` file in the repository root and add your own credentials:

```dotenv
API_KEY=your_nvidia_api_key
HF_TOKEN=your_huggingface_token
```

The NVIDIA-backed scripts use the OpenAI-compatible NVIDIA endpoint. Most accept `NVIDIA_API_KEY` or `API_KEY`; the chat, function-calling, and low-code scripts specifically read `API_KEY`. The image-generation script reads `HF_TOKEN`. Set only the credentials needed for the project you want to run. Do not commit `.env` or paste real keys into source files.

## Run a project

With the virtual environment activated, run a script from the repository root. For example:

```powershell
python .\Chat-Application\chat.py
python .\Function-Calling\function_call.py
python .\Image-Generation\image_gen.py
python .\Low-Code-AI\low_code.py
python .\RAG\rag.py
python .\Search-App\search.py
python .\Text-Generation\text.py
python .\Text-Generation\chef_bot.py
```

The RAG and search projects use the included `RAG\Knowledge\zentora_test_corpus.pdf` and `Search-App\source_document.txt` files. The RAG and search scripts generate embeddings through the NVIDIA API when they start.

## Dependencies

The root `requirements.txt` contains the shared dependencies for all projects. `Text-Generation\requirements.txt` is retained for installing that folder's smaller dependency set separately.
