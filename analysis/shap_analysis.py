import os
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ========== CONFIGURATION ==========
DATA_PATH = "D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/data/synthetic_tax_fraud_dataset.csv"
MODEL_PATH = "D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/models/xgboost_model.json"
FIGURE_DIR = "D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# ========== LOAD DATA ==========
try:
    df = pd.read_csv(DATA_PATH)
    print(" Data loaded.")
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found at path: {DATA_PATH}")

# ========== ENCODE CATEGORICAL ==========
categorical_cols = ["filing_status", "occupation_category"]
for col in categorical_cols:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        print(f" Column '{col}' not found in dataset and skipped.")

# ========== SELECT FEATURES ==========
features = [
    "income_reported", "deductions_claimed", "tax_credits_claimed", "num_dependents",
    "days_to_deadline", "deduction_to_income_ratio", "credit_to_income_ratio",
    "expense_per_dependent", "income_per_dependent"
]
features += [col for col in categorical_cols if col in df.columns]
features = list(dict.fromkeys(features))  # remove duplicates

X = df[features]

# ========== LOAD MODEL ==========
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
print(" Model loaded.")

# ========== COMPUTE SHAP VALUES ==========
explainer = shap.Explainer(model)
shap_values = explainer(X)
print(" SHAP values computed.")

# ========== PLOT 1: BEESWARM ==========
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
beeswarm_path = os.path.join(FIGURE_DIR, "shap_summary_beeswarm.png")
plt.savefig(beeswarm_path)
print(f" Saved SHAP summary beeswarm plot to {beeswarm_path}")

# ========== PLOT 2: BAR ==========
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
bar_path = os.path.join(FIGURE_DIR, "shap_bar_plot.png")
plt.savefig(bar_path)
print(f" Saved SHAP bar plot to {bar_path}")

# ========== PLOT 3: FORCE PLOT ==========
shap.initjs()
force_plot = shap.force_plot(
    base_value=shap_values.base_values[0],
    shap_values=shap_values.values[0],
    features=X.iloc[0],
    feature_names=X.columns
)

force_path = os.path.join(FIGURE_DIR, "shap_force_plot.html")
with open(force_path, "w", encoding="utf-8") as f:
    f.write(shap.getjs())  # SHAP JS runtime
    f.write(force_plot.html())

print(f" Saved SHAP force plot (instance 0) to {force_path}")
