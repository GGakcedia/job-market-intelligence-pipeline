import pandas as pd
from pathlib import Path
from typing import Optional, Union, Iterator

DATA_DIR = Path(__file__).parent.parent / "data"

def load_postings(
    chunksize: Optional[int] = 10000, 
    nrows: Optional[int] = None,
    usecols: Optional[list] = None
) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """
    Load raw postings data. Memory safe by default using chunksize.
    Rename 'At scraping time' to 'scraped_at' if not using explicit usecols that drop it.
    """
    file_path = DATA_DIR / "raw" / "postings.csv"
    
    # Define optimal dtypes for memory savings where applicable
    dtype_map = {
        'job_id': 'Int64',
        'company_name': 'category',
        'title': 'string',
        'max_salary': 'float32',
        'med_salary': 'float32',
        'pay_period': 'category',
        'location': 'category',
        'company_id': 'Int64',
        'views': 'Int32'
    }
    
    # If chunks are requested
    if chunksize:
        iterator = pd.read_csv(
            file_path, 
            chunksize=chunksize, 
            nrows=nrows, 
            usecols=usecols,
            dtype=dtype_map
        )
        # We need a generator to yield renamed chunks
        def renamed_chunks(gen):
            for chunk in gen:
                if 'At scraping time' in chunk.columns:
                    chunk = chunk.rename(columns={'At scraping time': 'scraped_at'})
                yield chunk
        return renamed_chunks(iterator)

    # Standard read
    df = pd.read_csv(file_path, nrows=nrows, usecols=usecols, dtype=dtype_map)
    if 'At scraping time' in df.columns:
        df = df.rename(columns={'At scraping time': 'scraped_at'})
    return df

def load_job_skills() -> pd.DataFrame:
    """Load job_skills mapping."""
    file_path = DATA_DIR / "raw" / "jobs" / "job_skills.csv"
    return pd.read_csv(file_path, dtype={'job_id': 'Int64', 'skill_abr': 'category'})

def load_skills_map() -> pd.DataFrame:
    """Load skills descriptions map."""
    file_path = DATA_DIR / "raw" / "mappings" / "skills.csv"
    return pd.read_csv(file_path, dtype={'skill_abr': 'category', 'skill_name': 'string'})