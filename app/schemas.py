from pydantic import BaseModel

class TicketInput(BaseModel):
    issue_description: str
    product: str
    category: str
    channel: str
    region: str
    customer_age: float
    customer_gender: str
    subscription_type: str
    customer_tenure_months: float
    previous_tickets: float
    customer_satisfaction_score: float
    first_response_time_hours: float
    resolution_time_hours: float
    escalated: str
    sla_breached: str
    operating_system: str
    browser: str
    payment_method: str
    language: str
    preferred_contact_time: str
    issue_complexity_score: float
    customer_segment: str
