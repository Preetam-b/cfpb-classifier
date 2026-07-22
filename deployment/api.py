from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import uvicorn

app = FastAPI(
    title="CFPB Complaint Classifier API",
    description="Classifies consumer complaints using fine-tuned BERT",
    version="1.0.0"
)

# Load model on startup
print("Loading model...")
tokenizer = BertTokenizerFast.from_pretrained('../saved_models/bert_model')
model     = BertForSequenceClassification.from_pretrained('../saved_models/bert_model')
model.eval()
print("Model loaded!")

labels = [
    'Credit Card',
    'Credit Reporting',
    'Debt Collection',
    'Mortgages and Loans',
    'Retail Banking'
]

# Request schema
class ComplaintRequest(BaseModel):
    text: str

# Response schema
class ComplaintResponse(BaseModel):
    predicted_label: str
    confidence:      float
    probabilities:   dict

@app.get("/")
def home():
    return {
        "message": "CFPB Complaint Classifier API",
        "version": "1.0.0",
        "status":  "running"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=ComplaintResponse)
def predict(request: ComplaintRequest):
    inputs = tokenizer(
        request.text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]

    predicted_idx   = probs.argmax().item()
    predicted_label = labels[predicted_idx]
    confidence      = probs[predicted_idx].item()
    probabilities   = {labels[i]: round(probs[i].item(), 4)
                      for i in range(len(labels))}

    return ComplaintResponse(
        predicted_label=predicted_label,
        confidence=confidence,
        probabilities=probabilities
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)