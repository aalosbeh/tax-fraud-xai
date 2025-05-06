import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import os

# === Load Data ===
df = pd.read_csv("D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/data/synthetic_tax_fraud_dataset.csv")

# Remove unnamed column if present
df = df.drop(columns=["Unnamed: 30"], errors="ignore")

# === Debug: Print available columns ===
print("Available columns:")
print(df.columns.tolist())

# === Encode Categorical Features ===
categorical_features = ["filing_status", "occupation_category"]
for col in categorical_features:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    else:
        print(f"Warning: Column '{col}' not found and skipped.")

# === Define Features ===
features = [
    "income_reported",
    "deductions_claimed",
    "tax_credits_claimed",
    "num_dependents",
    "days_to_deadline",
    "deduction_to_income_ratio",
    "credit_to_income_ratio",
    "expense_per_dependent",
    "income_per_dependent",
]
# Add encoded categorical features
features += [col for col in categorical_features if col in df.columns]
features = list(dict.fromkeys(features))  # Ensure uniqueness

# === Prepare Data ===
X = df[features]
y = df["fraud_flag"]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Train XGBoost Model ===
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# === Evaluate Model ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.4f}")

# === Save Model ===
model_dir = "D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "xgboost_model.json")
model.save_model(model_path)
print(f"✅ Model saved to {model_path}")
