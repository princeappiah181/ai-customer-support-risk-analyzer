# AI Customer Support Risk Analyzer

End-to-End AI System for Predicting and Prioritizing Customer Support Tickets
Hybrid NLP + Tabular Deep Learning • FastAPI • Streamlit • Explainable AI

------------------------------------------------------------

OVERVIEW

This project is a production-style AI system that predicts the risk level of customer support tickets in real time using a hybrid deep learning architecture (DistilBERT + tabular features).

It goes beyond prediction by providing:
- Actionable recommendations (SLA-based)
- Explainability (human + SHAP-style)
- Batch processing
- Dashboard analytics
- Alert system
- SaaS-ready API design

------------------------------------------------------------

FEATURES

- Single Ticket Prediction
- Batch CSV Processing
- Explainable AI Outputs
- Recommended Actions (SLA-based)
- Dashboard Analytics
- Critical Ticket Alerts
- API Key System (Monetization-ready)
- Downloadable Prediction Reports

------------------------------------------------------------

PROBLEM STATEMENT

Customer support teams must quickly decide:
Which tickets need immediate attention vs standard handling?

This system classifies tickets into:

Critical → Escalate immediately (5 min SLA)
Medium → Same-day handling (1 hour SLA)
Low → Standard queue (24-hour SLA)

------------------------------------------------------------

SYSTEM ARCHITECTURE

Ticket Text + Metadata
→ Preprocessing
→ Hybrid Model (DistilBERT + Tabular NN)
→ FastAPI Backend
→ Streamlit UI
→ Dashboard + Alerts + Batch Processing

------------------------------------------------------------

DATASET

- 200,000 customer support tickets
- 23 features (text + structured)

Key features include:
issue_description, customer_age, tenure, previous_tickets, complexity, etc.

------------------------------------------------------------

KEY INSIGHT

Original label (priority) performed like random guessing (~25% accuracy).

Solution:
Engineered a business-driven risk score:

risk_score =
2.0 * issue_complexity_score +
0.5 * previous_tickets +
0.01 * customer_tenure_months

Converted into:
Low, Medium, Critical

------------------------------------------------------------

MODEL

Hybrid architecture:
- DistilBERT for text
- Neural network for tabular data
- Fusion layer for final prediction

------------------------------------------------------------

PERFORMANCE

Accuracy: 92.45%
Macro F1: 92.39%

------------------------------------------------------------

EXPLAINABILITY

Human Explanation:
High complexity + many tickets + urgent language → critical risk

SHAP-style Explanation:
- issue_complexity_score → high impact
- previous_tickets → high impact
- issue_description → high impact

------------------------------------------------------------

API

POST /predict

Returns:
- risk
- confidence
- recommended_action
- explanation
- plan

------------------------------------------------------------

API KEY SYSTEM

free-demo-key → free plan
pro-demo-key → pro plan

------------------------------------------------------------

STREAMLIT UI

- Single ticket analyzer
- Batch upload
- Dashboard analytics
- Alerts system

------------------------------------------------------------

##  Production & SaaS Readiness

This system is designed with real-world deployment and monetization in mind.

Key capabilities include:
- API key authentication for controlled access
- Scalable FastAPI backend
- Batch processing for enterprise use
- Dashboard analytics for operational insights
- Explainability for decision transparency

Potential extensions:
- User authentication (JWT)
- Rate limiting & usage tracking
- Stripe-based subscription billing
- Database logging (PostgreSQL)
- Integration with support platforms (Zendesk, Slack)
------------------------------------------------------------

DEPLOYMENT

- Works locally
- Render failed due to memory limits
- Recommended: upgrade or use AWS/Railway

------------------------------------------------------------

TECH STACK

Python
PyTorch
Transformers (HuggingFace)
FastAPI
Streamlit
Scikit-learn
Pandas


------------------------------------------------------------

AUTHOR

Prince Appiah
PhD Data Science 

------------------------------------------------------------

FINAL NOTE

This is not just a model.
It is a production-style AI product prototype ready for real-world use.
