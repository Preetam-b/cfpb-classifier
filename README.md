# 🏦 CFPB Consumer Complaint Classifier

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-orange)](https://huggingface.co/spaces/P-r-e-e-t-a-m/cfpb-classifier)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20HuggingFace-Model-blue)](https://huggingface.co/P-r-e-e-t-a-m/cfpb-bert-classifier)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/Preetam-b/cfpb-classifier)

## 🚀 Live Demo
👉 **[Try the app on HuggingFace Spaces](https://huggingface.co/spaces/P-r-e-e-t-a-m/cfpb-classifier)**

---

## 📌 Overview
The **Consumer Financial Protection Bureau (CFPB)** receives millions of consumer complaints every year about financial products. Each complaint needs to be manually routed to the right team. This project **automates that process** using NLP.

Given a consumer complaint narrative, the model predicts which financial product category it belongs to — instantly and accurately.

---

## 🎯 Problem Statement
- CFPB receives **3+ million complaints per year**
- Manual classification is **slow, inconsistent and expensive**
- Wrong routing means **consumers wait longer** for resolution
- An automated system can classify complaints **instantly**

---

## 📊 Models Compared

| Model | Accuracy | F1 Score | Notes |
|---|---|---|---|
| Logistic Regression | 86.16% | 86.15% | Strong baseline |
| RNN | 76.68% | 76.56% | Struggles with long sequences |
| BiLSTM | 86.08% | 86.07% | Matches baseline |
| **BERT (Fine-tuned)** | **87.47%** | **87.47%** | **Best model ✅** |

> **Key Finding:** Vanilla RNN performed worse than Logistic Regression — proving that deep learning isn't automatically better for text classification. BiLSTM improved significantly but BERT's contextual understanding gave the final accuracy boost.

---

## 🏷️ Categories

| Label | Description |
|---|---|
| Credit Card | Issues with credit card fees, billing, disputes |
| Credit Reporting | Incorrect information on credit reports |
| Debt Collection | Harassment, incorrect debt claims |
| Mortgages and Loans | Payment issues, loan servicing problems |
| Retail Banking | Account freezes, unauthorized transactions |

---

## 🛠️ Tech Stack

**Machine Learning:**
- Scikit-learn (Logistic Regression, TF-IDF)
- TensorFlow / Keras (RNN, BiLSTM)
- PyTorch (BERT fine-tuning)
- HuggingFace Transformers
- Optuna (Hyperparameter tuning)
- Imbalanced-learn (Class balancing)
- GloVe Embeddings

**Deployment:**
- FastAPI (REST API backend)
- Streamlit (Local UI)
- Gradio (HuggingFace Spaces)
- HuggingFace Hub (Model hosting)
- AWS S3 (Model storage)
- AWS SageMaker (Real-time inference endpoint)
cfpb-classifier/
│
├── data/
│ ├── preprocess.ipynb ← Data cleaning + balancing
│ ├── baseline.ipynb ← TF-IDF + Logistic Regression
│ ├── rnn.ipynb ← RNN model + Optuna tuning
│ ├── BiLSTM.ipynb ← BiLSTM model + fine tuning
│ └── bert.ipynb ← BERT fine tuning (Google Colab)
│
├── deployment/
│ ├── api.py ← FastAPI REST API
│ ├── app.py ← Streamlit UI
│ └── requirements.txt
│
├── sagemaker/
│ ├── deploy.py ← SageMaker deployment script
│ └── requirements.txt
│
├── .gitignore
└── README.md


---

## 🔧 How to Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/Preetam-b/cfpb-classifier.git
cd cfpb-classifier
```

**2. Install dependencies:**
```bash
pip install -r deployment/requirements.txt
```

**3. Start FastAPI backend:**
```bash
cd deployment
python api.py
```

**4. Start Streamlit frontend:**
```bash
streamlit run app.py
```

**5. Open in browser:**

Streamlit UI → http://localhost:8501
FastAPI docs → http://localhost:8000/docs


---

## ☁️ AWS SageMaker Deployment

**1. Upload model to S3:**

Bucket: cfpb-classifier-model
Path : s3://cfpb-classifier-model/bert_model/


**2. Deploy endpoint:**
```bash
pip install -r sagemaker/requirements.txt
python sagemaker/deploy.py
```

**3. Test endpoint:**
```python
result = predictor.predict({
    "inputs": "My credit card company charged me incorrect fees."
})
# Output: [{'label': 'Credit Card', 'score': 0.87}]
```

**4. Delete endpoint after use:**
```python
predictor.delete_endpoint()
```

---

## 📈 Dataset
- **Source:** [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- **Size:** 50,000 samples (balanced using under/oversampling)
- **Classes:** 5 financial product categories
- **Split:** 72% train | 8% val | 20% test

---

## 🔍 Key Findings
- **Class balancing** was critical — original dataset had 83K credit reporting vs 12K retail banking
- **RNN underperformed** baseline (76% vs 86%) due to vanishing gradient on long complaint texts
- **BiLSTM matched** Logistic Regression — showing traditional ML is competitive on structured text
- **BERT outperformed** all models — contextual understanding handles financial language overlap best
- **Most confused classes:** Credit Card vs Retail Banking — overlapping language in complaints

---

## 🌐 Links
- 🤗 **Live Demo:** [HuggingFace Spaces](https://huggingface.co/spaces/P-r-e-e-t-a-m/cfpb-classifier)
- 🤗 **Model:** [HuggingFace Hub](https://huggingface.co/P-r-e-e-t-a-m/cfpb-bert-classifier)
- 💻 **GitHub:** [cfpb-classifier](https://github.com/Preetam-b/cfpb-classifier)
- ☁️ **S3 Bucket:** cfpb-classifier-model
- 🚀 **SageMaker:** Real-time inference endpoint (deploy on demand)

---

## 👨‍💻 Author
**Preetam** — [GitHub](https://github.com/Preetam-b)

---

## 📁 Project Structure
