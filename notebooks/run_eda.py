import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless + no white png issues

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from collections import Counter
from itertools import combinations


def _is_nonempty_list(x) -> bool:
    return isinstance(x, (list, np.ndarray)) and len(x) > 0


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "jobs_features.parquet"
    fig_dir = base_dir / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path, engine="pyarrow")

    # --- basic sanity
    if "role_category" not in df.columns:
        raise KeyError("Expected 'role_category' in dataframe columns.")

    # Chart 1: Role Distribution
    print("Generating Role Distribution...")
    plt.figure(figsize=(10, 6))
    role_counts = df["role_category"].value_counts()
    role_counts.plot(kind="bar")
    plt.title("Distribution of Job Roles (role_category)")
    plt.xlabel("Role Category")
    plt.ylabel("Number of Postings")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "role_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Chart 2: Salary Distribution (YEARLY only)
    print("Generating Salary Distribution (YEARLY med_salary)...")
    if "pay_period" in df.columns and "med_salary" in df.columns:
        yearly = df[(df["pay_period"] == "YEARLY") & (df["med_salary"].notna())].copy()
        # light outlier trim to avoid useless plots
        yearly = yearly[(yearly["med_salary"] > 10_000) & (yearly["med_salary"] < 1_000_000)]
        if not yearly.empty:
            plt.figure(figsize=(10, 6))
            plt.hist(yearly["med_salary"], bins=30, edgecolor="black")
            plt.title("Distribution of YEARLY Median Salaries")
            plt.xlabel("Median Salary (USD)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(fig_dir / "salary_distribution.png", dpi=200, bbox_inches="tight")
            plt.close()
        else:
            print("No YEARLY med_salary rows after filtering; skipping salary_distribution.png")
    else:
        print("Missing pay_period/med_salary; skipping salary_distribution.png")

    # Chart 3: Top Job Functions (your mapped job functions)
    print("Generating Top Job Functions...")
    if "job_functions" in df.columns:
        all_jf = df["job_functions"].explode().dropna()
        all_jf = all_jf[all_jf != ""]
        top_jf = all_jf.value_counts().head(15)
        if not top_jf.empty:
            plt.figure(figsize=(10, 6))
            top_jf.sort_values().plot(kind="barh")
            plt.title("Top 15 Job Functions Across All Postings")
            plt.xlabel("Frequency")
            plt.ylabel("Job Function")
            plt.tight_layout()
            plt.savefig(fig_dir / "top_job_functions.png", dpi=200, bbox_inches="tight")
            plt.close()
        else:
            print("Job functions empty; skipping top_job_functions.png")
    else:
        print("job_functions column missing; skipping top_job_functions.png")

    # Chart 4: Top Tech Skills
    print("Generating Top Tech Skills...")
    if "tech_skills" in df.columns:
        all_tech = df["tech_skills"].explode().dropna()
        all_tech = all_tech[all_tech != ""]
        top_tech = all_tech.value_counts().head(20)
        if not top_tech.empty:
            plt.figure(figsize=(10, 6))
            top_tech.sort_values().plot(kind="barh")
            plt.title("Top Technical Skills (keyword extracted)")
            plt.xlabel("Frequency")
            plt.ylabel("Tech Skill")
            plt.tight_layout()
            plt.savefig(fig_dir / "top_tech_skills.png", dpi=200, bbox_inches="tight")
            plt.close()
        else:
            print("No tech skills found; skipping top_tech_skills.png")
    else:
        print("tech_skills column missing; skipping top_tech_skills.png")

    # Chart 5: Role x Tech Skill pairs (this was missing in your file)
    print("Generating Role x Tech Skill Pairs...")
    if "tech_skills" in df.columns:
        nonempty = df["tech_skills"].apply(_is_nonempty_list)
        tmp = df.loc[nonempty, ["role_category", "tech_skills"]].explode("tech_skills")
        tmp = tmp[tmp["tech_skills"].notna() & (tmp["tech_skills"] != "")]
        if not tmp.empty:
            pivot = (
                tmp.groupby(["role_category", "tech_skills"])
                   .size()
                   .reset_index(name="count")
                   .sort_values("count", ascending=False)
                   .head(15)
            )
            plt.figure(figsize=(10, 6))
            labels = pivot["tech_skills"].astype(str) + " | " + pivot["role_category"].astype(str)
            plt.barh(labels, pivot["count"])
            plt.title("Most Common Tech Skill + Role Combinations")
            plt.xlabel("Count")
            plt.tight_layout()
            plt.savefig(fig_dir / "role_skill_pairs.png", dpi=200, bbox_inches="tight")
            plt.close()
        else:
            print("No non-empty tech_skills rows; skipping role_skill_pairs.png")
    else:
        print("tech_skills column missing; skipping role_skill_pairs.png")

    # Chart 6: Salary by Role (this was missing in your file)
    print("Generating Salary by Role (YEARLY max_salary)...")
    if all(c in df.columns for c in ["pay_period", "max_salary", "role_category"]):
        salary_df = df[(df["pay_period"] == "YEARLY") & (df["max_salary"].notna())].copy()
        salary_df = salary_df[(salary_df["max_salary"] > 10_000) & (salary_df["max_salary"] < 1_000_000)]
        if not salary_df.empty:
            role_salary = salary_df.groupby("role_category")["max_salary"].median().sort_values()
            plt.figure(figsize=(10, 6))
            role_salary.plot(kind="barh")
            plt.title("Median Max Salary by Role (YEARLY only)")
            plt.xlabel("Median max_salary")
            plt.tight_layout()
            plt.savefig(fig_dir / "salary_by_role.png", dpi=200, bbox_inches="tight")
            plt.close()
        else:
            print("No YEARLY max_salary rows after filtering; skipping salary_by_role.png")
    else:
        print("Missing salary columns; skipping salary_by_role.png")

    # Chart 7: Top Tech Skills by Role (simpler, no subplots -> fewer white image issues)
    print("Generating Top Tech Skills by Role...")
    if "tech_skills" in df.columns:
        top_roles = df["role_category"].value_counts().head(4).index.tolist()
        for role in top_roles:
            role_skills = df.loc[df["role_category"] == role, "tech_skills"].explode().dropna()
            role_skills = role_skills[role_skills != ""]
            top_role_skills = role_skills.value_counts().head(12)

            if top_role_skills.empty:
                continue

            safe_name = role.replace("/", "_").replace(" ", "_")
            plt.figure(figsize=(10, 6))
            top_role_skills.sort_values().plot(kind="barh")
            plt.title(f"Top Tech Skills for {role}")
            plt.xlabel("Frequency")
            plt.tight_layout()
            plt.savefig(fig_dir / f"top_tech_skills_{safe_name}.png", dpi=200, bbox_inches="tight")
            plt.close()

    # Chart 8: Tech Skill Co-occurrence Heatmap (top 20)
    print("Generating Tech Skill Co-occurrence Heatmap...")
    if "tech_skills" in df.columns:
        co_counts = Counter()
        for skills_list in df["tech_skills"].dropna():
            if not _is_nonempty_list(skills_list) or len(skills_list) < 2:
                continue
            clean = sorted([s for s in skills_list if s])
            co_counts.update(combinations(clean, 2))

        if co_counts:
            all_skills = df["tech_skills"].explode().dropna()
            all_skills = all_skills[all_skills != ""]
            top_n = all_skills.value_counts().head(20).index.tolist()

            co_matrix = pd.DataFrame(0, index=top_n, columns=top_n)
            for (s1, s2), count in co_counts.items():
                if s1 in top_n and s2 in top_n:
                    co_matrix.loc[s1, s2] = count
                    co_matrix.loc[s2, s1] = count

            plt.figure(figsize=(12, 10))
            plt.imshow(co_matrix, interpolation="nearest", aspect="auto")
            plt.colorbar(label="Co-occurrence Count")
            plt.xticks(range(len(top_n)), top_n, rotation=90)
            plt.yticks(range(len(top_n)), top_n)
            plt.title("Top Tech Skills Co-occurrence (Heatmap)")
            plt.tight_layout()
            plt.savefig(fig_dir / "tech_skill_cooccurrence.png", dpi=200, bbox_inches="tight")
            plt.close()
        else:
            print("No co-occurrence pairs found; skipping tech_skill_cooccurrence.png")

    print("All figures generated successfully!")


if __name__ == "__main__":
    main()