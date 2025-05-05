import numpy as np
import pandas as pd
import random

def generate_synthetic_tax_data(num_records=10000, fraud_ratio=0.1):
    np.random.seed(42)
    random.seed(42)
    
    data = []
    for _ in range(num_records):
        is_fraud = np.random.rand() < fraud_ratio

        income = np.random.lognormal(mean=10, sigma=0.5)
        deductions = income * np.random.uniform(0.05, 0.3) if not is_fraud else income * np.random.uniform(0.5, 2.0)
        credits = income * np.random.uniform(0.01, 0.1) if not is_fraud else income * np.random.uniform(0.1, 0.3)
        num_dependents = np.random.randint(0, 5)
        filing_status = random.choice(["single", "married", "head_of_household"])
        occupation = random.choice(["teacher", "engineer", "artist", "doctor", "retired", "student"])
        days_to_deadline = np.random.randint(0, 100) if not is_fraud else np.random.randint(0, 20)
        year_over_year_income_change = np.random.uniform(-0.2, 0.2) if not is_fraud else np.random.uniform(-0.5, 1.0)

        data.append({
            "income": income,
            "deductions": deductions,
            "credits": credits,
            "num_dependents": num_dependents,
            "filing_status": filing_status,
            "occupation": occupation,
            "days_to_deadline": days_to_deadline,
            "yearly_income_change": year_over_year_income_change,
            "is_fraud": int(is_fraud)
        })

    df = pd.DataFrame(data)
    df["deduction_to_income"] = df["deductions"] / df["income"]
    df["credit_to_income"] = df["credits"] / df["income"]
    df["expense_per_dependent"] = df["deductions"] / (df["num_dependents"] + 1)
    return df

if __name__ == "__main__":
    df = generate_synthetic_tax_data()
    df.to_csv("data/synthetic_tax_fraud_dataset.csv", index=False)
    print("Synthetic dataset saved to data/synthetic_tax_fraud_dataset.csv")
