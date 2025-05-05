# 🧠 Proactive Tax Fraud Detection Using Explainable AI  
**Hybrid GBDT + Attention-Based Neural Network with SHAP & Attention Visualization**

This repository contains the official implementation of our IACIS 2025 research paper on proactive tax fraud detection. Our system combines the strengths of Gradient Boosted Decision Trees (XGBoost), Attention-based Deep Neural Networks (DNN), and explainability methods (SHAP and attention heatmaps) to provide an interpretable, high-performing hybrid fraud detection framework for synthetic U.S. tax return data.

---

## 🚀 Project Highlights

- ✅ **Synthetic Dataset**: Mimics real IRS tax return features and fraud patterns (10% fraud rate)
- 📈 **Hybrid Model**: XGBoost + Attention-based DNN + Logistic Regression meta-learner
- 🔍 **Explainable AI**: SHAP values + attention weights for both global and local interpretability
- ⏱️ **Proactive Detection**: Works with partially completed tax filings for early fraud identification
- 🧾 **IRS Law Alignment**: Complies with U.S. tax code 26 U.S.C. §7201 & §7206
- 💡 **Best Paper Submission**: Targeted for IACIS 2025 and Issues in Information Systems journal

---

## 📂 Repository Structure

```
📁 tax_fraud_detection_project/
│
├── data/
│   ├── generate_data.py          # Script to generate synthetic tax return dataset
│   └── synthetic_tax_fraud_dataset.csv (output)
│
├── models/
│   ├── train_xgboost.py          # GBDT model training (XGBoost)
│   ├── train_dnn.py              # Attention-enhanced DNN training (Keras/TensorFlow)
│   └── train_hybrid.py           # Combines GBDT + DNN using logistic regression
│
├── analysis/
│   ├── shap_analysis.py          # SHAP summary + force plot
│   └── attention_heatmap.py      # Visualization of attention weights
│
├── notebooks/                    # (Optional) Jupyter notebooks for exploration
│
├── figures/                      # Auto-generated: SHAP plots, heatmaps
│
├── README.md
├── LICENSE                       # MIT License
└── requirements.txt              # Python dependencies
```

---

## 🧠 Models Overview

| Model            | Accuracy | Recall | Precision | F1 Score |
|------------------|----------|--------|-----------|----------|
| Rule-based       | 0.78     | 0.39   | 0.72      | 0.50     |
| XGBoost (GBDT)   | 0.90     | 0.84   | 0.80      | 0.82     |
| DNN (Attention)  | 0.89     | 0.80   | 0.82      | 0.81     |
| **Hybrid (Ours)**| **0.92** | **0.88** | **0.83** | **0.85** |

✅ Our hybrid model outperforms all baselines in fraud detection, with interpretable results and minimal false positives.

---

## 📊 Explainability Tools

- **SHAP Summary Plot**: Displays global feature importance based on Shapley values  
- **SHAP Force Plot**: Shows contribution of each feature to individual predictions  
- **Attention Heatmaps**: Highlights feature-level attention from the neural network for interpretability

All visual outputs are stored in `figures/`.

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/tax-fraud-xai.git
cd tax-fraud-xai
pip install -r requirements.txt
```

---

## 🧪 Running the Project

```bash
# Step 1: Generate synthetic dataset
python data/generate_data.py

# Step 2: Train GBDT (XGBoost) model
python models/train_xgboost.py

# Step 3: Train DNN with attention
python models/train_dnn.py

# Step 4: Train hybrid meta-learner
python models/train_hybrid.py

# Step 5: Run SHAP analysis
python analysis/shap_analysis.py

# Step 6: Generate attention heatmap
python analysis/attention_heatmap.py
```

---

## ⚖️ Legal Compliance & Ethics

- Data is 100% synthetic and anonymized
- Designed in accordance with U.S. IRS tax code [§7201](https://www.law.cornell.edu/uscode/text/26/7201) (evasion) and [§7206](https://www.law.cornell.edu/uscode/text/26/7206) (false returns)
- Promotes transparency, fairness, and audit-justifiable AI in public finance

---

## 📚 Citation

If you use this repository or paper in your research, please cite:

```bibtex
@article{alsobeh2025taxfraudxai,
  title={Proactive Tax Fraud Detection Using Explainable AI Techniques: A Hybrid Approach},
  author={Alsobeh, Anas and Abo El Rob, Mustafa and Rouibah, Kamel and Shatnawi, Amani},
  journal={Issues in Information Systems (IIS)},
  year={2025},
  note={Submitted to IACIS 2025},
}
```

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙋‍♂️ Contributors

- **Dr. Anas AlSobeh** — Southern Illinois University  
- **Dr. Mustafa Abo El Rob** — University of Denver  
- **Dr. Kamel Rouibah** — Kuwait University  
- **Dr. Amani Shatnawi** — Weber State University  

---

## 📬 Contact

For questions or collaborations, contact [anas.alsobeh@siu.edu](mailto:anas.alsobeh@siu.edu).
