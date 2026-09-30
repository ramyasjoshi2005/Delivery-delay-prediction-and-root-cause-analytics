import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import numpy as np
import json

def run_statistics(input_path):
    print("Loading data for statistical analysis...")
    df = pd.read_csv(input_path)
    
    # We will compare 'Standard Class' vs 'First Class'
    group1 = df[df['Shipping Mode'] == 'Standard Class']
    group2 = df[df['Shipping Mode'] == 'First Class']
    
    successes = np.array([group1['Late'].sum(), group2['Late'].sum()])
    nobs = np.array([len(group1), len(group2)])
    
    p1 = successes[0] / nobs[0]
    p2 = successes[1] / nobs[1]
    
    # Two-proportion z-test
    stat, pval = proportions_ztest(successes, nobs)
    
    # Confidence Interval
    se = np.sqrt(p1 * (1 - p1) / nobs[0] + p2 * (1 - p2) / nobs[1])
    diff = p1 - p2
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se
    
    risk_ratio = p1 / p2
    
    results = {
        "Research Question": "Is the late-delivery rate significantly different between Standard Class and First Class shipping?",
        "Null Hypothesis": "There is no difference in the late-delivery rate between Standard Class and First Class.",
        "Alternative Hypothesis": "There is a significant difference in the late-delivery rate between Standard Class and First Class.",
        "Test": "Two-Proportion Z-Test",
        "Group 1 (Standard Class) Rate": f"{p1:.2%}",
        "Group 2 (First Class) Rate": f"{p2:.2%}",
        "Difference in proportions": f"{diff:.2%}",
        "Z-statistic": round(stat, 4),
        "p-value": pval,
        "95% Confidence Interval": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
        "Risk Ratio": round(risk_ratio, 4),
        "Interpretation": f"Since the p-value is {'less' if pval < 0.05 else 'greater'} than 0.05, we {'reject' if pval < 0.05 else 'fail to reject'} the null hypothesis. Standard Class has a significantly {'higher' if diff > 0 else 'lower'} late-delivery rate compared to First Class."
    }
    
    print("\n--- Statistical Analysis Report ---")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("-----------------------------------\n")

    with open('data/processed/stats_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    run_statistics('data/processed/clean_data.csv')
