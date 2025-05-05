import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
df = pd.read_csv("data/synthetic_tax_fraud_dataset.csv")

# Encode categorical variables
for col in ["filing_status", "occupation"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Features and target
features = ["income", "deductions", "credits", "num_dependents", "filing_status", "occupation",
            "days_to_deadline", "yearly_income_change", "deduction_to_income", "credit_to_income", "expense_per_dependent"]
X = df[features]
y = df["is_fraud"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
                          use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Save the model
model.save_model("models/xgboost_model.json")
print("Model saved to models/xgboost_model.json")
