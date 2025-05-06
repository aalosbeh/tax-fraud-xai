import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model

# Load dataset
df = pd.read_csv("data/synthetic_tax_fraud_dataset.csv")

# First, check available columns
print("Available columns in dataset:")
print(df.columns.tolist())

# Encode categorical features - USE ACTUAL COLUMNS FROM YOUR DATASET
categorical_features = ["filing_status"]  # Remove "occupation" if not present
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Use only existing numerical features
features = ["income", "deductions", "credits", "num_dependents",
            "days_to_deadline", "yearly_income_change",
            "deduction_to_income", "credit_to_income",
            "expense_per_dependent"]

# Add categorical features if they exist
features += [f for f in categorical_features if f in df.columns]

X = df[features].values
y = df["is_fraud"].values

# Preprocess for DNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load models
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("models/xgboost_model.json")

dnn_model = load_model("models/dnn_attention_model.h5")

# Get model outputs
xgb_train_preds = xgb_model.predict_proba(X_train)[:, 1]
xgb_test_preds = xgb_model.predict_proba(X_test)[:, 1]

dnn_train_preds = dnn_model.predict(X_train_scaled).ravel()
dnn_test_preds = dnn_model.predict(X_test_scaled).ravel()

# Meta-features
meta_X_train = np.vstack([xgb_train_preds, dnn_train_preds]).T
meta_X_test = np.vstack([xgb_test_preds, dnn_test_preds]).T

# Meta-learner
meta_model = LogisticRegression()
meta_model.fit(meta_X_train, y_train)
meta_preds = meta_model.predict(meta_X_test)
meta_probs = meta_model.predict_proba(meta_X_test)[:, 1]

# Evaluation
print("Classification Report (Hybrid):")
print(classification_report(y_test, meta_preds))
print(f"AUC-ROC Score (Hybrid): {roc_auc_score(y_test, meta_probs):.4f}")
