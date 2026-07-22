# CFPB Consumer Complaint Classifier

## Overview
NLP project that classifies consumer financial complaints into product categories using multiple ML models.

## Models Compared
| Model | Accuracy | F1 |
|---|---|---|
| Logistic Regression | 86.16% | 86.15% |
| RNN | 76.68% | 76.56% |
| BiLSTM | 86.08% | 86.07% |
| BERT (Fine-tuned) | 87.47% | 87.47% |

## Categories
- Credit Card
- Credit Reporting
- Debt Collection
- Mortgages and Loans
- Retail Banking

## Tech Stack
- Python, TensorFlow, PyTorch
- HuggingFace Transformers
- FastAPI, Streamlit
- Optuna, Scikit-learn

## How to Run
```bash
pip install -r deployment/requirements.txt
cd deployment
python api.py
streamlit run app.py
```

## Dataset
CFPB Consumer Complaint Database
- 50,000 samples (balanced)
- 5 product categories
- Source: consumerfinance.gov