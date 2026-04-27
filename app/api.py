import os
import joblib
import torch
import torch.nn as nn
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "distilbert-base-uncased"
MODEL_DIR = "models/hybrid_distilbert_tabular"

NUMERIC_COLS = [
    "customer_age",
    "customer_tenure_months",
    "previous_tickets",
    "issue_complexity_score",
]

CATEGORICAL_COLS = [
    "product",
    "category",
    "channel",
    "region",
    "customer_gender",
    "subscription_type",
    "operating_system",
    "browser",
    "payment_method",
    "language",
    "preferred_contact_time",
    "customer_segment",
]


class HybridDistilBERT(nn.Module):
    def __init__(self, num_tabular_features, num_classes):
        super().__init__()

        self.bert = AutoModel.from_pretrained(MODEL_NAME)

        self.tabular_net = nn.Sequential(
            nn.Linear(num_tabular_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(768 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, tabular):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        text_embedding = bert_output.last_hidden_state[:, 0, :]
        tabular_embedding = self.tabular_net(tabular)

        fused = torch.cat([text_embedding, tabular_embedding], dim=1)

        return self.classifier(fused)


class TicketInput(BaseModel):
    issue_description: str
    customer_age: int
    customer_tenure_months: int
    previous_tickets: int
    issue_complexity_score: int

    product: str
    category: str
    channel: str
    region: str
    customer_gender: str
    subscription_type: str
    operating_system: str
    browser: str = "Unknown"
    payment_method: str
    language: str
    preferred_contact_time: str
    customer_segment: str


def get_recommended_action(risk: str) -> str:
    if risk == "Critical":
        return (
            "Escalate immediately to a human support specialist. "
            "Recommended SLA: respond within 5 minutes."
        )
    elif risk == "Medium":
        return (
            "Prioritize for same-day resolution. "
            "Recommended SLA: respond within 1 hour."
        )
    else:
        return (
            "Handle through the standard support queue. "
            "Recommended SLA: respond within 24 hours."
        )


def generate_explanation(ticket: TicketInput, risk: str) -> str:
    reasons = []

    if ticket.issue_complexity_score >= 7:
        reasons.append("high issue complexity")

    if ticket.previous_tickets >= 5:
        reasons.append("many previous tickets")

    if ticket.customer_tenure_months < 6:
        reasons.append("new customer with low tenure")

    if ticket.customer_age >= 60:
        reasons.append("older customer profile may require extra support attention")

    urgent_words = [
        "urgent",
        "immediately",
        "failed",
        "crash",
        "error",
        "cannot access",
        "payment",
        "blocked",
        "cancelled",
        "not working",
    ]

    text_lower = ticket.issue_description.lower()

    matched_words = [
        word for word in urgent_words if word in text_lower
    ]

    if matched_words:
        reasons.append(
            "urgent language detected: " + ", ".join(matched_words[:3])
        )

    if not reasons:
        reasons.append("general ticket characteristics")

    return f"{', '.join(reasons)} → {risk.lower()} risk"


def generate_shap_style_explanation(ticket: TicketInput, risk: str):
    """
    Lightweight SHAP-style explanation.
    This is not full SHAP. It gives interpretable directional drivers
    based on the same business features used by the risk system.
    """

    drivers = []

    drivers.append({
        "feature": "issue_complexity_score",
        "value": ticket.issue_complexity_score,
        "impact": "high" if ticket.issue_complexity_score >= 7 else "moderate" if ticket.issue_complexity_score >= 4 else "low",
        "direction": "increases risk" if ticket.issue_complexity_score >= 7 else "neutral/moderate risk contribution",
    })

    drivers.append({
        "feature": "previous_tickets",
        "value": ticket.previous_tickets,
        "impact": "high" if ticket.previous_tickets >= 5 else "moderate" if ticket.previous_tickets >= 2 else "low",
        "direction": "increases risk" if ticket.previous_tickets >= 5 else "limited risk contribution",
    })

    drivers.append({
        "feature": "customer_tenure_months",
        "value": ticket.customer_tenure_months,
        "impact": "high" if ticket.customer_tenure_months < 6 else "moderate" if ticket.customer_tenure_months < 18 else "low",
        "direction": "increases risk for newer customers" if ticket.customer_tenure_months < 6 else "reduces/normalizes risk",
    })

    text_lower = ticket.issue_description.lower()

    text_risk_terms = [
        "urgent",
        "failed",
        "crash",
        "blocked",
        "payment",
        "cancelled",
        "error",
        "not working",
    ]

    matched_terms = [term for term in text_risk_terms if term in text_lower]

    drivers.append({
        "feature": "issue_description",
        "value": matched_terms if matched_terms else "no major urgent terms detected",
        "impact": "high" if matched_terms else "low",
        "direction": "increases risk" if matched_terms else "limited text-based risk signal",
    })

    return {
        "method": "rule-guided SHAP-style explanation",
        "note": "This is a lightweight interpretable explanation, not full SHAP computation.",
        "top_drivers": drivers,
        "summary": generate_explanation(ticket, risk),
    }


app = FastAPI(
    title="AI Customer Support Risk Analyzer",
    description="Hybrid DistilBERT + tabular model for customer support ticket risk prediction.",
    version="1.1.0",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

sample_tabular = pd.DataFrame([{
    "customer_age": 30,
    "customer_tenure_months": 12,
    "previous_tickets": 2,
    "issue_complexity_score": 5,
    "product": "Web Portal",
    "category": "Technical Issue",
    "channel": "Email",
    "region": "North America",
    "customer_gender": "Male",
    "subscription_type": "Premium",
    "operating_system": "MacOS",
    "browser": "Chrome",
    "payment_method": "Credit Card",
    "language": "English",
    "preferred_contact_time": "Morning",
    "customer_segment": "Small Business",
}])

num_tabular_features = preprocessor.transform(sample_tabular).shape[1]

model = HybridDistilBERT(
    num_tabular_features=num_tabular_features,
    num_classes=len(label_encoder.classes_),
)

model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "model.pt"),
        map_location=device,
    )
)

model.to(device)
model.eval()


@app.get("/")
def home():
    return {
        "message": "AI Customer Support Risk Analyzer API is running.",
        "version": "1.1.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device),
    }


@app.post("/predict")
def predict(ticket: TicketInput):
    tabular_data = pd.DataFrame([{
        "customer_age": ticket.customer_age,
        "customer_tenure_months": ticket.customer_tenure_months,
        "previous_tickets": ticket.previous_tickets,
        "issue_complexity_score": ticket.issue_complexity_score,
        "product": ticket.product,
        "category": ticket.category,
        "channel": ticket.channel,
        "region": ticket.region,
        "customer_gender": ticket.customer_gender,
        "subscription_type": ticket.subscription_type,
        "operating_system": ticket.operating_system,
        "browser": ticket.browser if ticket.browser else "Unknown",
        "payment_method": ticket.payment_method,
        "language": ticket.language,
        "preferred_contact_time": ticket.preferred_contact_time,
        "customer_segment": ticket.customer_segment,
    }])

    tabular_features = preprocessor.transform(tabular_data)

    if hasattr(tabular_features, "toarray"):
        tabular_features = tabular_features.toarray()

    tabular_tensor = torch.tensor(
        tabular_features,
        dtype=torch.float32,
    ).to(device)

    encoding = tokenizer(
        ticket.issue_description,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tabular=tabular_tensor,
        )

        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    risk = label_encoder.inverse_transform(
        [predicted_class.cpu().item()]
    )[0]

    explanation = generate_explanation(ticket, risk)
    shap_style_explanation = generate_shap_style_explanation(ticket, risk)

    return {
        "risk": risk,
        "confidence": round(confidence.cpu().item(), 4),
        "recommended_action": get_recommended_action(risk),
        "explanation": explanation,
        "shap_style_explanation": shap_style_explanation,
    }