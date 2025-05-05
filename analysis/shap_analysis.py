import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
df = pd.read_csv("data/synthetic_tax_fraud_dataset.csv")

# Encode categorical features
for col in ["filing_status", "occupation"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

features = ["income", "deductions", "credits", "num_dependents", "filing_status", "occupation",
            "days_to_deadline", "yearly_income_change", "deduction_to_income", "credit_to_income", "expense_per_dependent"]
X = df[features]

# Load model
model = xgb.XGBClassifier()
model.load_model("models/xgboost_model.json")

# SHAP analysis
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Summary plot (beeswarm)
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("figures/shap_summary_plot.png")
print("Saved SHAP summary plot to figures/shap_summary_plot.png")

# Force plot for first example (HTML)
shap.initjs()
force_plot = shap.plots.force(shap_values[0], matplotlib=False)
with open("figures/force_plot.html", "w") as f:
    f.write(shap.save_html(force_plot))
print("Saved SHAP force plot to figures/force_plot.html")
