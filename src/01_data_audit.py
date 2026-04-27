import pandas as pd

DATA_PATH = "data/customer_support_tickets_200k.csv"

# Read only needed columns first for speed
needed_cols = [
    "issue_description", "priority", "product", "category", "channel", "region",
    "customer_age", "customer_gender", "subscription_type", "customer_tenure_months",
    "previous_tickets", "customer_satisfaction_score", "first_response_time_hours",
    "resolution_time_hours", "escalated", "sla_breached", "operating_system",
    "browser", "payment_method", "language", "preferred_contact_time",
    "issue_complexity_score", "customer_segment"
]

df = pd.read_csv(DATA_PATH, usecols=needed_cols)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False))

print("\nPriority distribution:")
print(df["priority"].value_counts(dropna=False))

print("\nSample rows:")
print(df.head(5).to_string())
