# AI Customer Support Triage & Priority Analyzer

Production-ready hybrid NLP + tabular machine learning system for customer support ticket prioritization.

## Goal
Predict customer support ticket priority using:
- DistilBERT text embeddings from `issue_description`
- Tabular customer/ticket metadata
- FastAPI deployment
- Docker-ready structure

## Main Dataset Column Plan

### Text input
- `issue_description`

### Target label
- `priority`

### Recommended tabular features
- `product`
- `category`
- `channel`
- `region`
- `customer_age`
- `customer_gender`
- `subscription_type`
- `customer_tenure_months`
- `previous_tickets`
- `customer_satisfaction_score`
- `first_response_time_hours`
- `resolution_time_hours`
- `escalated`
- `sla_breached`
- `operating_system`
- `browser`
- `payment_method`
- `language`
- `preferred_contact_time`
- `issue_complexity_score`
- `customer_segment`

## First Build Steps

1. Put the dataset inside `data/customer_support_tickets_200k.csv`
2. Run data audit:

```bash
python src/01_data_audit.py
```

3. Train the hybrid model:

```bash
python src/02_train_hybrid_distilbert.py
```

4. Run the API:

```bash
uvicorn app.api:app --reload
```

## API Endpoints

- `GET /health`
- `POST /predict`

## Project Roadmap

### Phase 1: Baseline
- Dataset audit
- Label mapping
- DistilBERT-only model
- Basic metrics

### Phase 2: Hybrid model
- DistilBERT + tabular metadata
- Fusion dense layers
- Classification head

### Phase 3: Production
- FastAPI
- Docker
- Prediction logging
- API key support

### Phase 4: Monetization
- Usage tiers
- Dashboard
- Slack/Zendesk/Gmail integration
