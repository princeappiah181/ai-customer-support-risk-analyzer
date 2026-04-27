import streamlit as st
import requests

# 🔥 UPDATE THIS AFTER DEPLOYMENT
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="AI Support Risk Analyzer", layout="centered")

st.title("🤖 AI Customer Support Risk Analyzer")
st.markdown("Predict ticket risk level and recommended action in real time.")

# -------------------------
# User Inputs
# -------------------------

issue_description = st.text_area("Issue Description")

col1, col2 = st.columns(2)

with col1:
    customer_age = st.number_input("Customer Age", value=30)
    customer_tenure_months = st.number_input("Tenure (months)", value=12)
    previous_tickets = st.number_input("Previous Tickets", value=2)

with col2:
    issue_complexity_score = st.slider("Issue Complexity", 1, 10, 5)

# -------------------------
# Static fields (can upgrade later)
# -------------------------

payload = {
    "issue_description": issue_description,
    "customer_age": customer_age,
    "customer_tenure_months": customer_tenure_months,
    "previous_tickets": previous_tickets,
    "issue_complexity_score": issue_complexity_score,
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
    "customer_segment": "Small Business"
}

# -------------------------
# Prediction Button
# -------------------------

if st.button("🔍 Predict Risk"):

    if issue_description.strip() == "":
        st.warning("Please enter an issue description.")
    else:
        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                result = response.json()

                st.success(f"🚨 Risk Level: {result['risk']}")
                st.write(f"📊 Confidence: {result['confidence']}")
                st.write(f"⚡ Action: {result['recommended_action']}")

                st.markdown("### 🔎 Explanation")
                st.write(result["explanation"])

                st.markdown("### 🧠 SHAP-style Explanation")
                st.json(result["shap_style_explanation"])

            else:
                st.error("API error. Check backend is running.")

        except Exception as e:
            st.error(f"Connection error: {e}")
