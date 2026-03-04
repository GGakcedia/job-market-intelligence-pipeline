import pandas as pd
import re

def extract_role_taxonomy(title_series: pd.Series) -> pd.Series:
    """
    Given a series of job titles, extracts a standardized role taxonomy.
    Categories: Data Analyst, Data Scientist, ML Engineer, Data Engineer, BI Analyst, Software Engineer, Other
    """
    def map_title(title):
        if not isinstance(title, str):
            return 'Other'
        t = title.lower()
        if re.search(r'\bdata\s+scient', t) or re.search(r'\bdata\s+science\b', t) or re.search(r'\bapplied\s+scient', t):
            return 'Data Scientist'
        if re.search(r'\bdata\s+engin', t):
            return 'Data Engineer'
        if re.search(r'\bml\b|\bmachine\s+learning\b', t) and re.search(r'engin', t):
            return 'ML Engineer'
        if re.search(r'\bbi\b|\bbusiness\s+intelligence\b', t):
            return 'BI Analyst'
        if re.search(r'\bdata\s+analyst\b', t):
            return 'Data Analyst'
        if re.search(r'\bsoftware\b|\bfront\s*end\b|\bback\s*end\b|\bfull\s*stack\b', t) and re.search(r'engin|develop|program', t):
            return 'Software Engineer'
        return 'Other'
        
    return title_series.apply(map_title).astype('category')

def build_job_functions_list(job_skills: pd.DataFrame, skills_map: pd.DataFrame) -> pd.DataFrame:
    """
    Given job_skills and skills_map dataframes, returns a dataframe with 
    job_id and a list of job_function names (broad industries).
    Columns returned: job_id, job_functions (list)
    """
    if job_skills.empty or skills_map.empty:
        return pd.DataFrame(columns=['job_id', 'job_functions'])
        
    # Clean keys
    if 'job_id' in job_skills.columns:
        job_skills['job_id'] = job_skills['job_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    if 'skill_abr' in job_skills.columns:
        job_skills['skill_abr'] = job_skills['skill_abr'].astype(str).str.strip().str.lower()
        
    if 'skill_abr' in skills_map.columns:
        skills_map['skill_abr'] = skills_map['skill_abr'].astype(str).str.strip().str.lower()
        
    # Join the map
    merged = pd.merge(job_skills, skills_map, on='skill_abr', how='left')
    
    # Drop rows where skill_name couldn't be found just in case
    merged = merged.dropna(subset=['skill_name'])
    
    # Group by job_id and aggregate skill_name into a list
    grouped = merged.groupby('job_id')['skill_name'].agg(list).reset_index()
    grouped = grouped.rename(columns={'skill_name': 'job_functions'})
    
    return grouped

def extract_tech_skills(description_series: pd.Series) -> pd.Series:
    """
    Given a series of job descriptions, extracts specific technical skills
    using transparent keyword matching. Returns a series of lists of skills.
    
    Curated keyword list:
    Python, SQL, R, Java, C++, C#, JavaScript, TypeScript, Go, Rust,
    TensorFlow, PyTorch, scikit-learn, Pandas, NumPy, Spark, Hadoop,
    AWS, Azure, GCP, Docker, Kubernetes, Terraform,
    React, Angular, Node.js, FastAPI, Django, Flask,
    PostgreSQL, MySQL, MongoDB, Redis, Snowflake.
    """
    # Define keywords to search for
    keywords = {
        'Python': r'\bpython\b',
        'SQL': r'\bsql\b',
        'R': r'\br\b',
        'Java': r'\bjava\b',
        'C++': r'\bc\+\+(?!\w)',
        'C#': r'\bc#(?!\w)',
        'JavaScript': r'\bjavascript\b|(?<!\.)\bjs\b',
        'TypeScript': r'\btypescript\b|\bts\b',
        'Go': r'\bgolang\b|\bgo\b',
        'Rust': r'\brust\b',
        'TensorFlow': r'\btensorflow\b|\btf\b',
        'PyTorch': r'\bpy\s*torch\b',
        'scikit-learn': r'\bscikit-learn\b|\bsklearn\b',
        'Pandas': r'\bpandas\b',
        'NumPy': r'\bnumpy\b',
        'Spark': r'\bspark\b|\bpyspark\b|\bapache\s+spark\b',
        'Hadoop': r'\bhadoop\b',
        'AWS': r'\baws\b|\bamazon\s+web\s+services\b',
        'Azure': r'\bazure\b',
        'GCP': r'\bgcp\b|\bgoogle\s+cloud\b',
        'Docker': r'\bdocker\b',
        'Kubernetes': r'\bkubernetes\b|\bk8s\b',
        'Terraform': r'\bterraform\b|\btf\b',
        'React': r'\breact\b|\breactjs\b|\breact\.js\b',
        'Angular': r'\bangular\b|\bangularjs\b',
        'Node.js': r'\bnode\.?js\b|\bnode\b',
        'FastAPI': r'\bfastapi\b',
        'Django': r'\bdjango\b',
        'Flask': r'\bflask\b',
        'PostgreSQL': r'\bpostgresql\b|\bpostgres\b',
        'MySQL': r'\bmysql\b',
        'MongoDB': r'\bmongo\s*db\b|\bmongo\b',
        'Redis': r'\bredis\b',
        'Snowflake': r'\bsnowflake\b'
    }
    
    # Compile regex patterns
    compiled_patterns = {skill: re.compile(pattern, re.IGNORECASE) for skill, pattern in keywords.items()}
    
    def get_skills(desc):
        if not isinstance(desc, str):
            return []
        # Return unique list per job_id
        found = set()
        for skill, pattern in compiled_patterns.items():
            if pattern.search(desc):
                found.add(skill)
        return list(found)
        
    return description_series.apply(get_skills)
