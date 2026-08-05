import streamlit as st
import pandas as pd

from utils import (
    calculate_clv,
    calculate_revenue_at_risk,
    calculate_priority_score,
    priority_level,
    smart_retention,
    calculate_average_monthly_spend,
)


def prepare_input(customer_df):

    customer_df["AverageMonthlySpend"] = calculate_average_monthly_spend(
        customer_df["TotalCharges"], customer_df["tenure"]
    )

    customer_df["Churn"].map({"Yes": 1, "No": 0})
    customer_df = pd.get_dummies(customer_df, drop_first=True, dtype=int)
    return customer_df


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

        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col2:

        online_security = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"]
        )

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

        customer = {
            "SeniorCitizen": senior,
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "AverageMonthlySpend": avg_spend,
            "gender_Male": 0,
            "Partner_Yes": 0,
            "Dependents_Yes": 0,
            "PhoneService_Yes": 0,
            "MultipleLines_No phone service": 0,
            "MultipleLines_Yes": 0,
            "InternetService_Fiber optic": 0,
            "InternetService_No": 0,
            "OnlineSecurity_No internet service": 0,
            "OnlineSecurity_Yes": 0,
            "OnlineBackup_No internet service": 0,
            "OnlineBackup_Yes": 0,
            "DeviceProtection_No internet service": 0,
            "DeviceProtection_Yes": 0,
            "TechSupport_No internet service": 0,
            "TechSupport_Yes": 0,
            "StreamingTV_No internet service": 0,
            "StreamingTV_Yes": 0,
            "StreamingMovies_No internet service": 0,
            "StreamingMovies_Yes": 0,
            "Contract_One year": 0,
            "Contract_Two year": 0,
            "PaperlessBilling_Yes": 0,
            "PaymentMethod_Credit card (automatic)": 0,
            "PaymentMethod_Electronic check": 0,
            "PaymentMethod_Mailed check": 0,
        }

        if gender == "Male":
            customer["gender_Male"] = 1

        if partner == "Yes":
            customer["Partner_Yes"] = 1

        if dependents == "Yes":
            customer["Dependents_Yes"] = 1

        if phone == "Yes":
            customer["PhoneService_Yes"] = 1

        if internet == "Fiber optic":
            customer["InternetService_Fiber optic"] = 1

        elif internet == "No":
            customer["InternetService_No"] = 1

        if online_security == "Yes":
            customer["OnlineSecurity_Yes"] = 1
        elif online_security == "No internet service":
            customer["OnlineSecurity_No internet service"] = 1

        if online_backup == "Yes":
            customer["OnlineBackup_Yes"] = 1
        elif online_backup == "No internet service":
            customer["OnlineBackup_No internet service"] = 1

        if device_protection == "Yes":
            customer["DeviceProtection_Yes"] = 1
        elif device_protection == "No internet service":
            customer["DeviceProtection_No internet service"] = 1

        if tech_support == "Yes":
            customer["TechSupport_Yes"] = 1
        elif tech_support == "No internet service":
            customer["TechSupport_No internet service"] = 1

        if streaming_tv == "Yes":
            customer["StreamingTV_Yes"] = 1
        elif streaming_tv == "No internet service":
            customer["StreamingTV_No internet service"] = 1

        if streaming_movies == "Yes":
            customer["StreamingMovies_Yes"] = 1
        elif streaming_movies == "No internet service":
            customer["StreamingMovies_No internet service"] = 1

        if contract == "One year":
            customer["Contract_One year"] = 1
        elif contract == "Two year":
            customer["Contract_Two year"] = 1

        if paperless == "Yes":
            customer["PaperlessBilling_Yes"] = 1

        if payment == "Credit card (automatic)":
            customer["PaymentMethod_Credit card (automatic)"] = 1

        elif payment == "Electronic check":
            customer["PaymentMethod_Electronic check"] = 1

        elif payment == "Mailed check":
            customer["PaymentMethod_Mailed check"] = 1

        input_df = pd.DataFrame([customer])

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

        clv = calculate_clv(monthly, tenure)

        revenue = calculate_revenue_at_risk(monthly, probability)

        priority = calculate_priority_score(clv, probability)

        level_name = priority_level(priority)

        customer_data = {
            "Priority_Level": level_name,  # e.g., "Critical", "High", "Medium", "Low"
            "CLV": clv,
            "tenure": tenure,
            "MonthlyCharges": monthly,
        }

        # Get recommended action
        retention_action = smart_retention(customer_data)
        st.success("Prediction Completed")

        if prediction == 1:
            st.error("⚠ Customer Will Churn")
        else:
            st.success("✅ Customer Will Stay")

        # Replace vertical st.metric calls with columns
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
                f"🚨 **Priority Level: Critical**\n\n**Recommended Action:** {retention_action}"
            )
        elif level_name == "High":
            st.warning(
                f"⚠️ **Priority Level: High**\n\n**Recommended Action:** {retention_action}"
            )
        elif level_name == "Medium":
            st.info(
                f"ℹ️ **Priority Level: Medium**\n\n**Recommended Action:** {retention_action}"
            )
        else:
            st.success(
                f"✅ **Priority Level: Low**\n\n**Recommended Action:** {retention_action}"
            )
       
