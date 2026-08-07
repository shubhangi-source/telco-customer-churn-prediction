import streamlit as st
import pandas as pd

from utils import (
    calculate_clv,
    calculate_revenue_at_risk,
    calculate_priority_score,
    priority_level,
    smart_retention,
    recommendation_reason,
)


def batch_prediction_page(model):

    st.title("📂 Batch Customer Churn Prediction")

    st.write("Upload a CSV file containing customer information.")

    uploaded_file = st.file_uploader("Choose CSV File", type=["csv"])

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")

        st.dataframe(df.head())

        # -----------------------------
        # Feature Engineering
        # -----------------------------
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

        df.drop("customerID", axis=1, inplace=True)

        df["AverageMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

        df = pd.get_dummies(df, drop_first=True, dtype=int)

        X = df.drop(columns=["Churn"], errors="ignore")
        # -----------------------------
        # Prediction
        # -----------------------------

        prediction = model.predict(X)

        probability = model.predict_proba(X)[:, 1]

        df["predicted_churn"] = prediction

        df["Churn_Probability"] = probability

        # -----------------------------
        # CLV
        # -----------------------------

        df["CLV"] = df.apply(
            lambda x: calculate_clv(x["MonthlyCharges"], x["tenure"]), axis=1
        )

        # -----------------------------
        # Revenue At Risk
        # -----------------------------

        df["Revenue_At_Risk"] = df.apply(
            lambda x: calculate_revenue_at_risk(
                x["MonthlyCharges"], x["Churn_Probability"]
            ),
            axis=1,
        )

        # -----------------------------
        # Priority Score
        # -----------------------------

        df["Priority_Score"] = df.apply(
            lambda x: calculate_priority_score(x["CLV"], x["Churn_Probability"]), axis=1
        )

        # -----------------------------
        # Priority Level
        # -----------------------------

        df["Priority_Level"] = df["Priority_Score"].apply(priority_level)

        # -----------------------------
        # Retention Recommendation
        # -----------------------------

        df["Retention_Action"] = df.apply(smart_retention, axis=1)

        # -----------------------------
        # Recommedation Reason
        # -----------------------------
        df["Recommendation_Reason"] = df.apply(recommendation_reason, axis=1)
        # -----------------------------
        # Results
        # -----------------------------

        st.success("Prediction Completed Successfully")

        st.subheader("Prediction Results")

        st.dataframe(df)

        # -----------------------------
        # Download Button
        # -----------------------------

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Prediction Results",
            data=csv,
            file_name="customer_retention_results.csv",
            mime="text/csv",
        )
