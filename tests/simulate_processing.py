import os
import pandas as pd

input_path ="D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/data"
if not os.path.exists(input_path):
    dummy_df = pd.DataFrame({
        "return_id": [1001, 1002, 1003],
        "pre-filing_fraud_prob": [0.1, 0.2, 0.3],
        "initial_fraud_prob": [0.4, 0.5, 0.6],
        "post-processing_fraud_prob": [0.7, 0.8, 0.9],
        "revenue_impact": [1000, 2000, 3000]
    })
    dummy_df.to_csv(input_path, index=False)
    print(f"Dummy input file created at {input_path}")
