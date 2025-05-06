import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

# === Load Dataset ===
df = pd.read_csv("D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/data/synthetic_tax_fraud_dataset.csv")
df = df.drop(columns=["Unnamed: 30"], errors="ignore")

print("Available columns in dataset:")
print(df.columns.tolist())

# === Encode Categorical Columns ===
categorical_features = ["filing_status", "occupation_category"]
for col in categorical_features:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    else:
        print(f"Warning: '{col}' not found in dataset!")

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
    "income_per_dependent"
]
features += [col for col in categorical_features if col in df.columns]

X = df[features].values
y = df["fraud_flag"].values

# === Scale Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split Dataset ===
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# === Load Pretrained Models ===
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/models/xgboost_model.json")

dnn_model = load_model("D:/myProjects/Proactive-Tax-Fraud-Detection_v2/pythonProject2/models/dnn_attention_model.h5")

# === Predict from Base Models ===
xgb_train_preds = xgb_model.predict_proba(X_train)[:, 1]
xgb_test_preds = xgb_model.predict_proba(X_test)[:, 1]

dnn_train_preds = dnn_model.predict(X_train).ravel()
dnn_test_preds = dnn_model.predict(X_test).ravel()

# === Build Meta-Features for Hybrid Model ===
meta_X_train = np.vstack((xgb_train_preds, dnn_train_preds)).T
meta_X_test = np.vstack((xgb_test_preds, dnn_test_preds)).T

# === Train Meta-Classifier ===
meta_model = LogisticRegression()
meta_model.fit(meta_X_train, y_train)

# === Predict and Evaluate ===
meta_preds = meta_model.predict(meta_X_test)
meta_probs = meta_model.predict_proba(meta_X_test)[:, 1]

print("=== Classification Report (Hybrid) ===")
print(classification_report(y_test, meta_preds))
print(f"AUC-ROC Score (Hybrid): {roc_auc_score(y_test, meta_probs):.4f}")