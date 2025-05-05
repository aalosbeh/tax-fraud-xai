import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, roc_auc_score

# Load dataset
df = pd.read_csv("data/synthetic_tax_fraud_dataset.csv")

# Encode categorical features
for col in ["filing_status", "occupation"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Define features and target
features = ["income", "deductions", "credits", "num_dependents", "filing_status", "occupation",
            "days_to_deadline", "yearly_income_change", "deduction_to_income", "credit_to_income", "expense_per_dependent"]
X = df[features].values
y = df["is_fraud"].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Attention-based DNN
input_layer = Input(shape=(X.shape[1],))
dense_proj = Dense(128, activation='relu')(input_layer)
dense_proj = Dropout(0.3)(dense_proj)

# Reshape for attention: (batch, sequence_length, features)
reshaped = Dense(128)(input_layer)
reshaped = LayerNormalization()(reshaped)
reshaped = Dropout(0.3)(reshaped)
reshaped = np.reshape(X_train, (-1, 1, X.shape[1]))

# Multi-Head Attention Layer
attention_output = MultiHeadAttention(num_heads=2, key_dim=2)(reshaped, reshaped)
attention_output = Flatten()(attention_output)

# Combine with dense layers
x = Dense(64, activation='relu')(attention_output)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

# Build model
model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size=32, callbacks=[early_stop])

# Evaluate
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob > 0.5).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Save model
model.save("models/dnn_attention_model.h5")
print("DNN model saved to models/dnn_attention_model.h5")
