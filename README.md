# 🤖 AI Customer Support Risk Analyzer

> End-to-end AI system for predicting and prioritizing customer support tickets using Hybrid NLP + Tabular Deep Learning

---

## 🚀 Overview

This project builds a production-style AI system that classifies customer support tickets into:

- 🔴 Critical → Immediate escalation (5 min SLA)
- 🟡 Medium → Same-day handling (1 hour SLA)
- 🟢 Low → Standard queue (24-hour SLA)

The system combines:
- DistilBERT (text understanding)
- Tabular features (customer + ticket metadata)
- FastAPI backend
- Streamlit dashboard

---

## ✨ Features

- Real-time ticket risk prediction  
- Batch CSV processing  
- Explainable AI (human + SHAP-style)  
- SLA-based recommended actions  
- Dashboard analytics  
- Critical ticket alerts  
- API key system (SaaS-ready)  
- Downloadable prediction reports  

---

## 🧠 Key Insight

The original dataset label (priority) performed like random guessing (~25% accuracy).

### Solution: Label Engineering

risk_score =
2.0 * issue_complexity_score +
0.5 * previous_tickets +
0.01 * customer_tenure_months

Converted into:
Low / Medium / Critical

---

## 🤖 Model Architecture

Text → DistilBERT → Embedding  
Tabular → Neural Network → Embedding  
→ Fusion → Risk Prediction  

---

## 📈 Performance

Accuracy: 92.45%  
Macro F1: 92.39%  

---

## 🔍 Explainability

Human explanation example:  
High complexity + many previous tickets + urgent language → critical risk  

SHAP-style insights:
- issue_complexity_score ↑  
- previous_tickets ↑  
- issue_description ↑  

---

## ⚡ API

Endpoint:
POST /predict

Returns:
- risk  
- confidence  
- recommended_action  
- explanation  
- plan  

---

## 🖥️ UI Capabilities

- Single ticket analyzer  
- Batch upload  
- Dashboard analytics  
- Alert system  

---

## 🚀 Production & SaaS Readiness

This system is designed with real-world deployment in mind:

- API key authentication  
- Scalable FastAPI backend  
- Batch processing  
- Dashboard analytics  
- Explainable predictions  

Future Extensions:
- JWT authentication  
- Stripe payments  
- Usage tracking  
- PostgreSQL logging  
- Slack/Zendesk integration  

---

## 🧩 Tech Stack

Python  
PyTorch  
Transformers (HuggingFace)  
FastAPI  
Streamlit  
Scikit-learn  
Pandas  

---

## 🛠️ How to Run

git clone <your-repo>  
pip install -r requirements.txt  

Run API:
uvicorn app.api:app --reload  

Run UI:
streamlit run app/ui.py  

---

## 🚧 Deployment

- Fully working locally  
- Render free tier memory limitation  

Recommended:
- Render (paid)
- Railway
- AWS / GCP  

---

## 👤 Author

Prince Appiah  
PhD Data Science  
 

---

## ⭐ Final Note

This is not just a model.

It is a production-style AI system designed for real-world deployment and SaaS applications.
