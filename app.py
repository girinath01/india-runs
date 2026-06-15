import streamlit as st
import pandas as pd
import time
import os
from rank import rank_candidates
import tempfile

st.set_page_config(page_title="Redrob Ranker - Bug Hunters", layout="wide")

st.title("🏆 Redrob Hackathon: Bug Hunters")
st.subheader("Senior AI Engineer - Candidate Ranker v2")

st.markdown("""
This is the sandbox environment for the **Bug Hunters** submission.
Our ranking system uses a 5-component weighted hybrid scorer (Skill Fit, Product Fit, Behavioral, Experience, Location/Notice) with strict hard penalties for JD mismatches.

**Features:**
- **Tiered Skill Scoring**: Prioritizes Retrieval/Ranking/Vector DBs over trendy LLM tooling.
- **Product Fit**: Penalizes IT Services consulting careers and rewards production shipping evidence.
- **Two-Pass Pipeline**: Fast-filters 100K candidates down to 3,000 before deep scoring.
""")

st.divider()

uploaded_file = st.file_uploader("Upload Candidates JSONL file", type=["jsonl", "json"])

if uploaded_file is not None:
    if st.button("Run Ranker"):
        with st.spinner("Processing candidates... this may take up to a minute for 100K records."):
            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp_in:
                tmp_in.write(uploaded_file.getvalue())
                tmp_in_path = tmp_in.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_out:
                tmp_out_path = tmp_out.name

            start_time = time.time()
            try:
                # Run the ranking pipeline
                rank_candidates(tmp_in_path, tmp_out_path)
                elapsed = time.time() - start_time
                
                st.success(f"Ranking complete in {elapsed:.1f} seconds!")
                
                # Read and display the results
                results_df = pd.read_csv(tmp_out_path)
                st.dataframe(results_df, use_container_width=True)
                
                # Provide download button
                with open(tmp_out_path, "rb") as f:
                    csv_bytes = f.read()
                
                st.download_button(
                    label="📥 Download Top 100 CSV",
                    data=csv_bytes,
                    file_name="bug_hunters_submission.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error during ranking: {str(e)}")
            finally:
                os.unlink(tmp_in_path)
                os.unlink(tmp_out_path)
else:
    st.info("Upload a candidate JSONL file to test the ranking algorithm.")
    
    st.markdown("### Pre-computed Results (Top 10)")
    try:
        df = pd.read_csv("bug_hunters.csv")
        st.dataframe(df.head(10), use_container_width=True)
    except FileNotFoundError:
        st.write("Run the ranker locally to generate `bug_hunters.csv` for the preview.")
