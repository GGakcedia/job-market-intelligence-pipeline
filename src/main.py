import pandas as pd
from src.io import load_postings, load_job_skills, load_skills_map
from src.cleaning import clean_postings
from src.features import extract_role_taxonomy, build_job_functions_list, extract_tech_skills
from pathlib import Path

def run_pipeline():
    """Run the end-to-end data processing pipeline."""
    print("Loading skills data...")
    job_skills_df = load_job_skills()
    skills_map_df = load_skills_map()
    
    print("Building job functions list per job...")
    job_skills_agg = build_job_functions_list(job_skills_df, skills_map_df)

    print("Loading and cleaning postings data (in memory-safe chunks if needed)...")
    # For a full pipeline run, we load raw and apply operations
    # If the file is very large we can process it chunk by chunk, 
    # but to build a final parquet we should eventually write it out.
    # To keep this implementation simple and fulfill parquet requirement:
    processed_chunks = []
    
    # Process in chunks to save memory
    postings_iter = load_postings(chunksize=50000)
    for i, chunk in enumerate(postings_iter):
        print(f"Processing chunk {i+1}...")
        
        # Clean
        cleaned = clean_postings(chunk)
        
        # Features: role taxonomy
        if 'title' in cleaned.columns:
            cleaned['role_category'] = extract_role_taxonomy(cleaned['title'])
            
        # Features: join job functions
        # Left join with job_skills_agg
        final_chunk = pd.merge(cleaned, job_skills_agg, on='job_id', how='left')
        
        # The 'job_functions' column will be NaN for joins without skills, convert to empty list
        if 'job_functions' in final_chunk.columns:
            # We must be careful because successful joins might return numpy arrays or lists
            final_chunk['job_functions'] = final_chunk['job_functions'].apply(lambda x: list(x) if isinstance(x, (list, tuple)) or type(x).__name__ == 'ndarray' else [])
            
        # Extract tech skills from description
        if 'description' in final_chunk.columns:
            final_chunk['tech_skills'] = extract_tech_skills(final_chunk['description'])
        else:
            final_chunk['tech_skills'] = [[] for _ in range(len(final_chunk))]
            
        processed_chunks.append(final_chunk)

    print("Concatenating processed chunks...")
    final_df = pd.concat(processed_chunks, ignore_index=True)
    
    # Save to parquet
    out_path = Path(__file__).parent.parent / "data" / "processed" / "jobs_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving to {out_path}...")
    final_df.to_parquet(out_path, engine='pyarrow', index=False)
    print("Pipeline completed.")

if __name__ == "__main__":
    run_pipeline()
