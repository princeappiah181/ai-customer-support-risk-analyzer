import pandas as pd

DATA_PATH = "data/customer_support_tickets_200k.csv"
OUTPUT_PATH = "data/customer_support_tickets_with_risk_label.csv"

df = pd.read_csv(DATA_PATH)

# Handle missing values
df["browser"] = df["browser"].fillna("Unknown")

# Create risk score using only features available before resolution
# This avoids target leakage.
df["risk_score"] = (
    2.0 * df["issue_complexity_score"]
    + 0.5 * df["previous_tickets"]
    + 0.01 * df["customer_tenure_months"]
)

# Create 3-class risk label directly
df["risk_label_3class"] = pd.qcut(
    df["risk_score"],
    q=3,
    labels=["Low", "Medium", "Critical"]
)

# Also keep risk_label for compatibility
df["risk_label"] = df["risk_label_3class"]

# Save updated dataset
df.to_csv(OUTPUT_PATH, index=False)

print("Risk label creation complete.")

print("\n3-class distribution:")
print(df["risk_label_3class"].value_counts())

print("\nSample:")
print(
    df[
        [
            "risk_score",
            "risk_label",
            "risk_label_3class",
            "issue_complexity_score",
            "previous_tickets",
            "customer_tenure_months",
        ]
    ].head()
)