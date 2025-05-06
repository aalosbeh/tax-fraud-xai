import os
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Paths
DATA_PATH = "D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/data/synthetic_tax_fraud_dataset.csv"
MODEL_PATH = "D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/models/xgboost_model.json"
FIGURES_DIR = "D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/figures"

# Ensure figures directory exists
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
print("✅ Data loaded.")

# Encode categorical features (correct columns)
for col in ["filing_status", "occupation_category"]:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Correct feature list based on your dataset
features = [
    "income_reported",
    "deductions_claimed",
    "tax_credits_claimed",
    "num_dependents",
    "filing_status",
    "occupation_category",
    "days_to_deadline",
    "deduction_to_income_ratio",
    "credit_to_income_ratio",
    "expense_per_dependent",
    "income_per_dependent"
]
X = df[features]

# Load model
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
print("✅ Model loaded.")

# Create SHAP explainer
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
print("✅ SHAP values computed.")

# --- Plot 1: SHAP Summary Plot (Beeswarm)
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_summary_plot.png"))
print("📊 Saved SHAP summary beeswarm plot.")

# --- Plot 2: SHAP Feature Importance (Bar)
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "shap_bar_plot.png"))
print("📊 Saved SHAP bar plot.")

# --- Plot 3: SHAP Force Plot (1st row, HTML)
shap.initjs()
force_html = shap.plots.force(shap_values[0], matplotlib=False)
force_html_path = os.path.join(FIGURES_DIR, "force_plot.html")

html_str = shap.save_html(force_html)  # Correct usage
with open(force_html_path, "w") as f:
    f.write(html_str)

print("📈 Saved SHAP force plot (first instance) as HTML.")
print("✅ All SHAP visualizations successfully generated.")
