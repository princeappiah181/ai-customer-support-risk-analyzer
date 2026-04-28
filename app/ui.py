import streamlit as st
import requests
import pandas as pd
from datetime import datetime

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="AI Support Risk Analyzer",
    page_icon="🤖",
    layout="wide",
)

if "history" not in st.session_state:
    st.session_state["history"] = []

st.title("🤖 AI Customer Support Risk Analyzer")
st.markdown(
    """
    Predict customer support ticket risk in real time using a hybrid AI model.
    The system returns a risk level, confidence score, recommended action, and explanation.
    """
)

# -------------------------
# Sidebar
# -------------------------

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


# -------------------------
# Helper Function
# -------------------------

def call_api(payload, api_key):
    headers = {"x-api-key": api_key}
    response = requests.post(
        API_URL,
        json=payload,
        headers=headers,
        timeout=60,
    )
    return response


def save_to_history(payload, result):
    st.session_state["history"].append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "issue_description": payload["issue_description"],
            "risk": result["risk"],
            "confidence": result["confidence"],
            "recommended_action": result["recommended_action"],
            "explanation": result["explanation"],
            "previous_tickets": payload["previous_tickets"],
            "issue_complexity_score": payload["issue_complexity_score"],
            "customer_segment": payload["customer_segment"],
            "subscription_type": payload["subscription_type"],
            "plan": result.get("plan", "unknown"),
        }
    )


def show_prediction_result(result):
    risk = result["risk"]
    confidence = result["confidence"]

    st.markdown("## ✅ Prediction Result")

    if risk == "Critical":
        st.error("🚨 CRITICAL ALERT: This ticket should be escalated immediately.")
    elif risk == "Medium":
        st.warning("⚠️ Medium-risk ticket. Prioritize this for same-day handling.")
    else:
        st.success("✅ Low-risk ticket. Standard support queue is appropriate.")

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

    if risk == "Critical" and confidence >= 0.60:
        st.toast("🚨 Critical ticket detected!", icon="🚨")

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


# -------------------------
# Tabs
# -------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "🔍 Single Ticket Analyzer",
        "📦 Batch Upload",
        "📊 Dashboard",
    ]
)

# -------------------------
# Tab 1: Single Ticket
# -------------------------

with tab1:
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
        category = st.selectbox(
            "Category",
            ["Technical Issue", "Billing Issue", "Account Suspension", "Performance Issue"],
        )
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
                response = call_api(payload, api_key)

                if response.status_code == 200:
                    result = response.json()
                    save_to_history(payload, result)
                    show_prediction_result(result)

                elif response.status_code == 401:
                    st.error("Invalid or missing API key. Try `free-demo-key` or `pro-demo-key`.")

                else:
                    st.error(f"API error: {response.status_code}")
                    st.write(response.text)

            except Exception as e:
                st.error(f"Connection error: {e}")


# -------------------------
# Tab 2: Batch Upload
# -------------------------

with tab2:
    st.markdown("## 📦 Batch Ticket Upload")
    st.markdown(
        """
        Upload a CSV file containing multiple customer support tickets.
        The CSV should include the same columns used by the single-ticket form.
        """
    )

    required_columns = [
        "issue_description",
        "customer_age",
        "customer_tenure_months",
        "previous_tickets",
        "issue_complexity_score",
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

    with st.expander("Required CSV columns"):
        st.write(required_columns)

    sample_df = pd.DataFrame(
        [
            {
                "issue_description": "My payment failed and I need this resolved urgently.",
                "customer_age": 34,
                "customer_tenure_months": 24,
                "previous_tickets": 7,
                "issue_complexity_score": 8,
                "product": "Payment Gateway",
                "category": "Billing Issue",
                "channel": "Email",
                "region": "North America",
                "customer_gender": "Male",
                "subscription_type": "Premium",
                "operating_system": "MacOS",
                "browser": "Chrome",
                "payment_method": "Credit Card",
                "language": "English",
                "preferred_contact_time": "Morning",
                "customer_segment": "Small Business",
            }
        ]
    )

    csv_template = sample_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download CSV Template",
        data=csv_template,
        file_name="ticket_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.markdown("### Uploaded Data Preview")
        st.dataframe(batch_df.head(), use_container_width=True)

        missing_cols = [col for col in required_columns if col not in batch_df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                results = []

                progress = st.progress(0)

                for idx, row in batch_df.iterrows():
                    payload = {col: row[col] for col in required_columns}

                    try:
                        response = call_api(payload, api_key)

                        if response.status_code == 200:
                            result = response.json()

                            output_row = payload.copy()
                            output_row["risk"] = result["risk"]
                            output_row["confidence"] = result["confidence"]
                            output_row["recommended_action"] = result["recommended_action"]
                            output_row["explanation"] = result["explanation"]
                            output_row["plan"] = result.get("plan", "unknown")

                            results.append(output_row)
                            save_to_history(payload, result)

                        else:
                            output_row = payload.copy()
                            output_row["risk"] = "ERROR"
                            output_row["confidence"] = None
                            output_row["recommended_action"] = response.text
                            output_row["explanation"] = "API error"
                            output_row["plan"] = "unknown"
                            results.append(output_row)

                    except Exception as e:
                        output_row = payload.copy()
                        output_row["risk"] = "ERROR"
                        output_row["confidence"] = None
                        output_row["recommended_action"] = str(e)
                        output_row["explanation"] = "Connection error"
                        output_row["plan"] = "unknown"
                        results.append(output_row)

                    progress.progress((idx + 1) / len(batch_df))

                result_df = pd.DataFrame(results)

                st.success("Batch prediction complete.")
                st.dataframe(result_df, use_container_width=True)

                output_csv = result_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇️ Download Predictions CSV",
                    data=output_csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )


# -------------------------
# Tab 3: Dashboard
# -------------------------

with tab3:
    st.markdown("## 📊 Prediction History Dashboard")

    history = st.session_state["history"]

    if len(history) == 0:
        st.info("No predictions yet. Run a single or batch prediction first.")
    else:
        hist_df = pd.DataFrame(history)

        total_predictions = len(hist_df)
        critical_count = (hist_df["risk"] == "Critical").sum()
        medium_count = (hist_df["risk"] == "Medium").sum()
        low_count = (hist_df["risk"] == "Low").sum()

        d1, d2, d3, d4 = st.columns(4)

        with d1:
            st.metric("Total Predictions", total_predictions)

        with d2:
            st.metric("Critical Tickets", critical_count)

        with d3:
            st.metric("Medium Tickets", medium_count)

        with d4:
            st.metric("Low Tickets", low_count)

        st.markdown("### Risk Distribution")
        risk_counts = hist_df["risk"].value_counts()
        st.bar_chart(risk_counts)

        st.markdown("### Confidence by Risk Level")
        confidence_summary = hist_df.groupby("risk")["confidence"].mean().reset_index()
        st.dataframe(confidence_summary, use_container_width=True)

        st.markdown("### Recent Predictions")
        st.dataframe(hist_df.tail(20), use_container_width=True)

        csv_data = hist_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Prediction History",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv",
        )

        if st.button("🧹 Clear Dashboard History"):
            st.session_state["history"] = []
            st.rerun()


# -------------------------
# Product Preview
# -------------------------

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
    - Support batch ticket triage
    - Provide dashboard analytics for support managers
    """
)