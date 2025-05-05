# Proactive Tax Fraud Detection Using Explainable AI

This repository provides the full implementation of a research study designed to detect fraudulent tax returns using explainable artificial intelligence. The system integrates a hybrid of Gradient Boosted Decision Trees (XGBoost), a deep neural network with an attention mechanism, and interpretability tools such as SHAP values and attention heatmaps.

The solution is designed with compliance, transparency, and scalability in mind—supporting IRS-aligned fraud detection with fully synthetic, privacy-respecting data.

## Key Features

- Synthetic dataset reflecting realistic tax filing and fraud behaviors
- A hybrid machine learning pipeline that combines tree-based models with deep learning
- Interpretable outputs using SHAP explanations and attention maps
- Early fraud risk detection from partially completed tax returns
- Structured for deployment in academic or public finance environments

## Repository Structure

```
tax_fraud_detection_project/
│
├── data/
│   ├── generate_data.py          # Creates the synthetic tax return dataset
│
├── models/
│   ├── train_xgboost.py          # XGBoost training script
│   ├── train_dnn.py              # Deep neural network with attention
│   └── train_hybrid.py           # Combines predictions using logistic regression
│
├── analysis/
│   ├── shap_analysis.py          # SHAP-based model explanation
│   └── attention_heatmap.py      # Visualizes attention weights per sample
│
├── figures/                      # Output directory for plots
├── notebooks/                    # Optional notebooks
├── README.md
├── LICENSE
└── requirements.txt
```

## Model Evaluation

| Model             | Accuracy | Recall | Precision | F1 Score |
|------------------|----------|--------|-----------|----------|
| Rule-based        | 0.78     | 0.39   | 0.72      | 0.50     |
| XGBoost (GBDT)    | 0.90     | 0.84   | 0.80      | 0.82     |
| Attention-based DNN | 0.89   | 0.80   | 0.82      | 0.81     |
| Hybrid Ensemble   | 0.92     | 0.88   | 0.83      | 0.85     |

The hybrid model consistently achieves the best fraud detection performance in terms of both precision and recall.

## Explainability

We use SHAP to generate global and local explanations for fraud predictions and complement this with attention heatmaps extracted from the neural network. These tools are designed to support transparency and decision justification for auditing purposes.

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/tax-fraud-xai.git
cd tax-fraud-xai
pip install -r requirements.txt
```

## Usage

```bash
python data/generate_data.py               # Step 1: Generate dataset
python models/train_xgboost.py             # Step 2: Train XGBoost model
python models/train_dnn.py                 # Step 3: Train DNN with attention
python models/train_hybrid.py              # Step 4: Train meta-learner
python analysis/shap_analysis.py           # Step 5: Run SHAP explainability
python analysis/attention_heatmap.py       # Step 6: Generate heatmaps
```

## Legal Context and Ethics

This tool uses only synthetically generated data that statistically mimics IRS filing patterns. It references core U.S. tax fraud statutes, including:
- [26 U.S.C. §7201 - Tax evasion](https://www.law.cornell.edu/uscode/text/26/7201)
- [26 U.S.C. §7206 - Fraudulent returns](https://www.law.cornell.edu/uscode/text/26/7206)

## Citation

If you use this work in your own research, please cite:

```bibtex
@article{alsobeh2025taxfraudxai,
  title={Proactive Tax Fraud Detection Using Explainable AI Techniques: A Hybrid Approach},
  author={Alsobeh, Anas and Abo El Rob, Mustafa and Rouibah, Kamel and Shatnawi, Amani},
  journal={Issues in Information Systems (IIS)},
  year={2025},
  note={Submitted to IACIS 2025}
}
```

## Contact

For questions or collaborations, contact the lead author at anas.alsobeh@siu.edu
