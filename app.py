import gradio as gr
from src.pipeline import rank_candidates
import os

def process_file(file_obj):
    if file_obj is None:
        return None
    
    # Handle both string paths and Gradio File objects
    input_path = file_obj if isinstance(file_obj, str) else file_obj.name
    output_path = "submission.csv"
    
    try:
        rank_candidates(input_path, output_path)
        return output_path
    except Exception as e:
        raise gr.Error(f"Error during ranking: {str(e)}")

demo = gr.Interface(
    fn=process_file,
    inputs=gr.File(label="Upload candidates.jsonl (or .json)", file_types=[".jsonl", ".json", ".gz"]),
    outputs=gr.File(label="Download Ranked CSV"),
    title="Bug Hunters - Redrob Ranker v6.0",
    description="Upload a sample of candidates to rank them. The system will process them through the Two-Pass Filter Engine and output a `submission.csv` file."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
