import gradio as gr
from src.pipeline import rank_candidates
import pandas as pd
import subprocess

# Ensure offline assets (like the ONNX model) are downloaded when the Hugging Face space boots up!
print("Checking for offline assets...")
subprocess.run(["python", "download_offline_assets.py"], check=True)

def process_file(file_obj):
    if file_obj is None:
        return None, None
    
    # Handle both string paths and Gradio File objects
    input_path = file_obj if isinstance(file_obj, str) else file_obj.name
    output_path = "submission.csv"
    
    try:
        rank_candidates(input_path, output_path)
        df = pd.read_csv(output_path)
        return df, output_path
    except Exception as e:
        raise gr.Error(f"Error during ranking: {str(e)}")

demo = gr.Interface(
    fn=process_file,
    inputs=gr.File(label="Upload candidates.jsonl (or .json)", file_types=[".jsonl", ".json", ".gz"]),
    outputs=[
        gr.Dataframe(label="Top Candidates Preview"),
        gr.File(label="Download Ranked CSV")
    ],
    title="Bug Hunters - Redrob Ranker v6.0",
    description="Upload a sample of candidates to rank them. The system will process them through the Two-Pass Filter Engine and output a `submission.csv` file."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
