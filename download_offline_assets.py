import os
import time
import subprocess
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

def download_with_retry(name, download_func, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"--- Downloading {name} (Attempt {attempt}/{max_retries}) ---")
            download_func()
            print(f"{name} download complete.\n")
            return
        except Exception as e:
            print(f"Failed to download {name}: {e}")
            if attempt < max_retries:
                print("Retrying in 5 seconds...\n")
                time.sleep(5)
            else:
                print(f"Max retries reached for {name}. Exiting.")
                raise e

def download_onnx():
    onnx_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_onnx_model")
    if not os.path.exists(onnx_dir):
        model = ORTModelForFeatureExtraction.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", export=True)
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model.save_pretrained(onnx_dir)
        tokenizer.save_pretrained(onnx_dir)
    else:
        print("ONNX model already exists locally. Skipping download.")

def download_spacy():
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)

if __name__ == "__main__":
    print("Starting offline assets download with retry logic...\n")
    download_with_retry("ONNX Semantic Model", download_onnx)
    download_with_retry("spaCy NER Model", download_spacy)
    print("All offline assets successfully prepared!")
