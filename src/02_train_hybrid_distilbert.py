import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup


DATA_PATH = "data/customer_support_tickets_with_risk_label.csv"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/hybrid_distilbert_tabular"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXT_COL = "issue_description"
TARGET_COL = "risk_label_3class"

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

USE_COLS = [TEXT_COL, TARGET_COL] + NUMERIC_COLS + CATEGORICAL_COLS

MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 6
LR = 2e-5
SAMPLE_SIZE = 30000


class TicketDataset(Dataset):
    def __init__(self, texts, tabular, labels, tokenizer, max_len):
        self.texts = texts.tolist()
        self.tabular = torch.tensor(tabular, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "tabular": self.tabular[idx],
            "labels": self.labels[idx],
        }


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(DATA_PATH, usecols=USE_COLS)

    df["browser"] = df["browser"].fillna("Unknown")
    df = df.dropna(subset=[TEXT_COL, TARGET_COL])
    df[TEXT_COL] = df[TEXT_COL].astype(str)

    print("Preprocessing complete.")
    print("Dataset shape:", df.shape)
    print("\nTarget label distribution:")
    print(df[TARGET_COL].value_counts())

    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COL])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )

    X_tab = preprocessor.fit_transform(df[NUMERIC_COLS + CATEGORICAL_COLS])

    if hasattr(X_tab, "toarray"):
        X_tab = X_tab.toarray()

    X_text_train, X_text_test, X_tab_train, X_tab_test, y_train, y_test = train_test_split(
        df[TEXT_COL],
        X_tab,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = TicketDataset(
        X_text_train,
        X_tab_train,
        y_train,
        tokenizer,
        MAX_LEN,
    )

    test_dataset = TicketDataset(
        X_text_test,
        X_tab_test,
        y_test,
        tokenizer,
        MAX_LEN,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = HybridDistilBERT(
        num_tabular_features=X_tab.shape[1],
        num_classes=len(label_encoder.classes_),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tabular = batch["tabular"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, tabular)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tabular = batch["tabular"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, tabular)
            predicted_classes = torch.argmax(logits, dim=1)

            preds.extend(predicted_classes.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print("\nAccuracy:", accuracy_score(true_labels, preds))
    print("Macro F1:", f1_score(true_labels, preds, average="macro"))

    print(
        classification_report(
            true_labels,
            preds,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))
    tokenizer.save_pretrained(OUTPUT_DIR)
    joblib.dump(preprocessor, os.path.join(OUTPUT_DIR, "preprocessor.joblib"))
    joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))

    print(f"Saved model artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()