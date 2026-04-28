import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="AI Support Risk Analyzer",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI Customer Support Risk Analyzer")
st.markdown(
    """
    Predict customer support ticket risk in real time using a hybrid AI model.
    The system returns a risk level, confidence score, recommended action, and explanation.
    """
)

st.sidebar.header("🔐 API Access")

api_key = st.sidebar.text_input(
    "Enter API Key",
    value="free-demo-key",
    type="password",
)

st.sidebar.markdown("### Demo Keys")
st.sidebar.code("free-demo-key")
st.sidebar.code("pro-demo-key")

st.sidebar.markdown("---")
st.sidebar.markdown("### Pricing Preview")
st.sidebar.markdown(
    """
    **Free Plan**  
    50 predictions/day  

    **Pro Plan**  
    $9/month  
    Higher request limits  

    **Business Plan**  
    $49/month  
    Team dashboard + integrations  
    """
)

st.markdown("---")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Model Type", "Hybrid AI")

with col_b:
    st.metric("Backend", "FastAPI")

with col_c:
    st.metric("Explainability", "Enabled")

st.markdown("## 📝 Ticket Input")

left, right = st.columns([2, 1])

with left:
    issue_description = st.text_area(
        "Issue Description",
        value="My payment failed and I need this resolved urgently.",
        height=160,
    )

with right:
    customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=34)
    customer_tenure_months = st.number_input("Customer Tenure (months)", min_value=0, value=24)
    previous_tickets = st.number_input("Previous Tickets", min_value=0, value=7)
    issue_complexity_score = st.slider("Issue Complexity Score", 1, 10, 8)

st.markdown("## 🧾 Ticket Metadata")

m1, m2, m3 = st.columns(3)

with m1:
    product = st.selectbox("Product", ["Web Portal", "Mobile App", "Payment Gateway"])
    category = st.selectbox("Category", ["Technical Issue", "Billing Issue", "Account Suspension", "Performance Issue"])
    channel = st.selectbox("Channel", ["Email", "Chat", "Social Media", "Phone"])

with m2:
    region = st.selectbox("Region", ["North America", "Europe", "Asia", "South America"])
    customer_gender = st.selectbox("Customer Gender", ["Male", "Female", "Other"])
    subscription_type = st.selectbox("Subscription Type", ["Free", "Premium", "Enterprise"])

with m3:
    operating_system = st.selectbox("Operating System", ["Windows", "MacOS", "Linux", "iOS", "Android"])
    browser = st.selectbox("Browser", ["Chrome", "Safari", "Firefox", "Edge", "Unknown"])
    payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"])

m4, m5 = st.columns(2)

with m4:
    language = st.selectbox("Language", ["English", "Spanish", "French"])
    preferred_contact_time = st.selectbox("Preferred Contact Time", ["Morning", "Afternoon", "Evening"])

with m5:
    customer_segment = st.selectbox("Customer Segment", ["Individual", "Small Business", "Corporate"])

payload = {
    "issue_description": issue_description,
    "customer_age": customer_age,
    "customer_tenure_months": customer_tenure_months,
    "previous_tickets": previous_tickets,
    "issue_complexity_score": issue_complexity_score,
    "product": product,
    "category": category,
    "channel": channel,
    "region": region,
    "customer_gender": customer_gender,
    "subscription_type": subscription_type,
    "operating_system": operating_system,
    "browser": browser,
    "payment_method": payment_method,
    "language": language,
    "preferred_contact_time": preferred_contact_time,
    "customer_segment": customer_segment,
}

st.markdown("---")

if st.button("🔍 Analyze Ticket", use_container_width=True):
    if not issue_description.strip():
        st.warning("Please enter an issue description.")
    else:
        try:
            headers = {
                "x-api-key": api_key
            }

            response = requests.post(
                API_URL,
                json=payload,
                headers=headers,
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()

                risk = result["risk"]
                confidence = result["confidence"]

                st.markdown("## ✅ Prediction Result")

                r1, r2, r3 = st.columns(3)

                with r1:
                    if risk == "Critical":
                        st.error(f"🚨 Risk Level: {risk}")
                    elif risk == "Medium":
                        st.warning(f"⚠️ Risk Level: {risk}")
                    else:
                        st.success(f"✅ Risk Level: {risk}")

                with r2:
                    st.metric("Confidence", f"{confidence * 100:.2f}%")

                with r3:
                    st.metric("Plan", result.get("plan", "unknown").upper())

                st.markdown("### ⚡ Recommended Action")
                st.info(result["recommended_action"])

                st.markdown("### 🔎 Explanation")
                st.write(result["explanation"])

                st.markdown("### 🧠 SHAP-style Explanation")
                shap_data = result["shap_style_explanation"]

                st.write(shap_data["summary"])
                st.caption(shap_data["note"])

                drivers = pd.DataFrame(shap_data["top_drivers"])
                st.dataframe(drivers, use_container_width=True)

            elif response.status_code == 401:
                st.error("Invalid or missing API key. Try `free-demo-key` or `pro-demo-key`.")

            else:
                st.error(f"API error: {response.status_code}")
                st.write(response.text)

        except Exception as e:
            st.error(f"Connection error: {e}")

st.markdown("---")

st.markdown("## 📌 Product Preview")
st.markdown(
    """
    This system can be packaged as a SaaS product for support teams.

    **Core value:**
    - Detect high-risk tickets early
    - Reduce missed urgent requests
    - Recommend SLA-based actions
    - Provide explainable AI decisions
    """
)