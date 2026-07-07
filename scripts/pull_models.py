"""Pull all required Ollama models for production."""
import subprocess
import sys

MODELS = [
    "llama3.1:8b",    # Chatbot, JD Generator, Recruiter, Career, Interview, Resume Builder
    "qwen3:8b",       # Job Parser, Resume Parser, ATS Scanner, Job Matching, Skill Assessment
    "mistral-small",  # General purpose
    "gemma3",         # General purpose alternative
    "nomic-embed-text", # Embeddings (already installed)
]

def pull_model(name: str):
    print(f"Pulling {name}...")
    result = subprocess.run(["ollama", "pull", name], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}: {result.stderr.strip()}")
    return result.returncode == 0

if __name__ == "__main__":
    print("Pulling Ollama models for ZyncJobs AI Platform")
    print(f"RAM recommended: 32GB | Current: ~{__import__('psutil').virtual_memory().total // (1024**3)}GB")
    print()

    for model in MODELS:
        pull_model(model)

    print("\nDone. Run `ollama list` to verify.")
