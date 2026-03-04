import pandas as pd
import pytest
from src.features import extract_role_taxonomy, build_job_functions_list, extract_tech_skills
from src.io import load_postings, load_job_skills
from src.cleaning import clean_postings

def test_extract_role_taxonomy():
    titles = pd.Series([
        "Senior Data Scientist",
        "Data Engineer II",
        "Machine Learning Engineer",
        "Business Intelligence Analyst",
        "Data Analyst - Remote",
        "Full Stack Software Engineer",
        "Marketing Manager"
    ])
    
    expected = pd.Series([
        "Data Scientist",
        "Data Engineer",
        "ML Engineer",
        "BI Analyst",
        "Data Analyst",
        "Software Engineer",
        "Other"
    ], dtype='category')
    
    result = extract_role_taxonomy(titles)
    
    pd.testing.assert_series_equal(result, expected, check_names=False)

def test_build_job_functions_list():
    job_skills = pd.DataFrame({
        'job_id': [1, 1, 2, 3],
        'skill_abr': ['PY', 'SQ', 'PY', 'unknown_skill']
    })
    
    skills_map = pd.DataFrame({
        'skill_abr': ['PY', 'SQ', 'AWS'],
        'skill_name': ['Python', 'SQL', 'Amazon Web Services']
    })
    
    # We do not strictly order the output dictionary, so let's just assert on values
    result = build_job_functions_list(job_skills, skills_map)
    result = result.sort_values('job_id').reset_index(drop=True)
    
    assert len(result) == 2
    assert result.loc[0, 'job_id'] == '1'
    assert set(result.loc[0, 'job_functions']) == {'Python', 'SQL'}
    
    assert result.loc[1, 'job_id'] == '2'
    assert set(result.loc[1, 'job_functions']) == {'Python'}

def test_extract_tech_skills():
    descriptions = pd.Series([
        "Looking for someone with strong Python and SQL experience on AWS cloud. Also python again.",
        "Must know Java and C++.",
        "Non-technical role in sales.",
        "Need experience in React.js, nodejs, and k8s",
        "Data infra with apache spark, postgresql, and mongo"
    ])
    
    result = extract_tech_skills(descriptions)
    
    assert set(result[0]) == {'Python', 'SQL', 'AWS'}
    assert set(result[1]) == {'Java', 'C++'}
    assert len(result[2]) == 0
    assert set(result[3]) == {'React', 'Node.js', 'Kubernetes'}
    assert set(result[4]) == {'Spark', 'PostgreSQL', 'MongoDB'}

def test_postings_job_skills_intersection():
    # Load a small sample to test intersection
    postings_iter = load_postings(chunksize=1000)
    postings = next(postings_iter)
    postings = clean_postings(postings)
    
    job_skills = load_job_skills()
    if 'job_id' in job_skills.columns:
        job_skills['job_id'] = job_skills['job_id'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        
    intersection = set(postings['job_id']).intersection(set(job_skills['job_id']))
    assert len(intersection) > 0, "Expected non-zero intersection of job_ids between postings and job_skills"

