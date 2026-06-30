import os
import subprocess
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

print("--- Downloading and Exporting ONNX Model ---")
onnx_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_onnx_model")
if not os.path.exists(onnx_dir):
    try:
        model = ORTModelForFeatureExtraction.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", export=True)
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model.save_pretrained(onnx_dir)
        tokenizer.save_pretrained(onnx_dir)
        print("ONNX export complete.")
    except Exception as e:
        print(f"ONNX export failed: {e}")
else:
    print("ONNX model already exists locally.")

print("\n--- Downloading spaCy NER Model ---")
try:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
    print("spaCy model download complete.")
except Exception as e:
    print(f"spaCy download failed: {e}")

print("\nAll offline assets prepared!")
