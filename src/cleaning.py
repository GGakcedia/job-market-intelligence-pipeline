import pandas as pd
import numpy as np

def clean_postings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the postings dataframe:
    - deduplicates by job_id
    - parses scraped_at datetime
    - cleans continuous numerical fields (salaries)
    - standardizes pay_period
    """
    if df.empty:
        return df
        
    # Deduplicate and clean job_id
    if 'job_id' in df.columns:
        df['job_id'] = df['job_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df = df.drop_duplicates(subset=['job_id'], keep='first').copy()
    
    # Parse scraped_at
    if 'scraped_at' in df.columns:
        # Convert to datetime, coercing errors to NaT
        df['scraped_at'] = pd.to_datetime(df['scraped_at'], unit='ms', errors='coerce')
        # fallback if it's a string timestamp
        if df['scraped_at'].isna().all() and df['scraped_at'].dtype != 'datetime64[ns]':
            df['scraped_at'] = pd.to_datetime(df['scraped_at'], errors='coerce')
        
    # Clean numeric fields
    for col in ['max_salary', 'med_salary']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Standardize pay_period
    if 'pay_period' in df.columns:
        # Assuming pay period values could be variations of 'HOURLY', 'YEARLY', 'MONTHLY', etc.
        df['pay_period'] = df['pay_period'].astype(str).str.upper().str.strip()
        df.loc[df['pay_period'] == 'NAN', 'pay_period'] = np.nan
        df['pay_period'] = df['pay_period'].astype('category')
        
    return df
