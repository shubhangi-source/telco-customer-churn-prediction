import streamlit as st
import pandas as pd

from utils import (
    calculate_clv,
    calculate_revenue_at_risk,
    calculate_priority_score,
    priority_level,
    smart_retention,
    recommendation_reason,
    calculate_average_monthly_spend,
)


def predict_customer_page(model):

    st.title("🔍 Customer Churn Prediction")

    st.write("Enter customer details below.")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox("Gender", ["Female", "Male"])

        senior = st.selectbox("Senior Citizen", [0, 1])

        partner = st.selectbox("Partner", ["No", "Yes"])

        dependents = st.selectbox("Dependents", ["No", "Yes"])

        tenure = st.slider("Tenure", 0, 72, 24)

        phone = st.selectbox("Phone Service", ["No", "Yes"])

        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )

        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        online_security = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"]
        )
    with col2:

        device_protection = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"]
        )

        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])

        payment = st.selectbox(
            "Payment Method",
            [
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ],
        )

        monthly = st.number_input("Monthly Charges", value=70.0)

        total = st.number_input("Total Charges", value=1500.0)

    if st.button("Predict"):

        avg_spend = calculate_average_monthly_spend(total, tenure)

        customer = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "SeniorCitizen": senior,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone,
                    "MultipleLines": multiple_lines,
                    "InternetService": internet,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless,
                    "PaymentMethod": payment,
                    "MonthlyCharges": monthly,
                    "TotalCharges": total,
                    "AverageMonthlySpend": avg_spend,
                }
            ]
        )

        prediction = model.predict(customer)[0]

        probability = model.predict_proba(customer)[0, 1]

        clv = calculate_clv(monthly, tenure)

        revenue = calculate_revenue_at_risk(monthly, probability)

        priority = calculate_priority_score(clv, probability)

        level_name = priority_level(priority)

        # Create one-hot-like fields only for the # business-rule functions in utils.py. # # These are NOT sent to the ML model.
        business_data = {
            "Contract": contract,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "TechSupport": tech_support,
            "InternetService": internet,
            "PaymentMethod": payment,
            "SeniorCitizen": senior,
            "Churn_Probability": probability,
            "Priority_Level": level_name,
            "CLV": clv,
            "tenure": tenure,
            "MonthlyCharges": monthly,
        }

        retention_action = smart_retention(business_data)

        reason = recommendation_reason(business_data)

        retention_action = smart_retention(business_data)

        reason = recommendation_reason(business_data)

        st.success("Prediction Completed")

        if prediction == 1:
            st.error("⚠ Customer Will Churn")
        else:
            st.success("✅ Customer Will Stay")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Churn Probability", f"{probability:.2%}")
            st.metric("Priority Score", f"{priority:.2f}")

        with col2:
            st.metric("Customer Lifetime Value", f"${clv:.2f}")
            st.metric("Revenue At Risk", f"${revenue:.2f}")

        # Dynamic retention strategy UI based on priority level
        if level_name == "Critical":
            st.error(
                f"🚨 **Priority Level: Critical**\n\n"
                f"**Recommended Action:** {retention_action}\n\n"
                f"**Recommendation Reason:** {reason}"
            )

        elif level_name == "High":
            st.warning(
                f"⚠️ **Priority Level: High**\n\n"
                f"**Recommended Action:** {retention_action}\n\n"
                f"**Recommendation Reason:** {reason}"
            )

        elif level_name == "Medium":
            st.info(
                f"ℹ️ **Priority Level: Medium**\n\n"
                f"**Recommended Action:** {retention_action}\n\n"
                f"**Recommendation Reason:** {reason}"
            )

        else:
            st.success(
                f"✅ **Priority Level: Low**\n\n"
                f"**Recommended Action:** {retention_action}\n\n"
                f"**Recommendation Reason:** {reason}"
            )
