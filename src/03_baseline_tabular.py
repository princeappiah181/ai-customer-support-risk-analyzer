import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_PATH = "data/customer_support_tickets_200k.csv"

TEXT_COL = "issue_description"
TARGET_COL = "priority"

NUMERIC_COLS = [
    "customer_age",
    "customer_tenure_months",
    "previous_tickets",
    "customer_satisfaction_score",
    "first_response_time_hours",
    "resolution_time_hours",
    "issue_complexity_score",
]

CATEGORICAL_COLS = [
    "product",
    "category",
    "channel",
    "region",
    "customer_gender",
    "subscription_type",
    "escalated",
    "sla_breached",
    "operating_system",
    "browser",
    "payment_method",
    "language",
    "preferred_contact_time",
    "customer_segment",
]

df = pd.read_csv(DATA_PATH)
df["browser"] = df["browser"].fillna("Unknown")
df = df.dropna(subset=[TARGET_COL])

X = df[NUMERIC_COLS + CATEGORICAL_COLS]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )),
    ]
)

model.fit(X_train, y_train)
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print("Macro F1:", f1_score(y_test, preds, average="macro"))
print(classification_report(y_test, preds, zero_division=0))
