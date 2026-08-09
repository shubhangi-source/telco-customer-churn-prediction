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

        # --------------------------------
        # Load Data
        # --------------------------------

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(df.head(), use_container_width=True)

        # --------------------------------
        # Feature Engineering
        # --------------------------------

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        df["AverageMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

        if "Churn" in df.columns:
            df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

        # --------------------------------
        # Keep Original Data
        # --------------------------------

        original_df = df.copy()
        # --------------------------------
        # Prepare data for model
        # --------------------------------

        X = df.drop(columns=["customerID", "Churn"], errors="ignore")

        # IMPORTANT:
        # No pd.get_dummies()
        # No manual scaling
        # ColumnTransformer Pipeline handles it.
        # Automatically align columns with the model's trained features
        if hasattr(model, "feature_names_in_"):
            X = X.reindex(columns=model.feature_names_in_, fill_value=0)
        # --------------------------------
        # Prediction
        # --------------------------------

        prediction = model.predict(X)

        probability = model.predict_proba(X)[:, 1]

        # --------------------------------
        # Add prediction results
        # --------------------------------

        original_df["predicted_churn"] = prediction

        original_df["Churn_Probability"] = probability

        # --------------------------------
        # CLV
        # --------------------------------

        original_df["CLV"] = original_df["MonthlyCharges"] * original_df["tenure"]

        # --------------------------------
        # Revenue At Risk
        # --------------------------------

        original_df["Revenue_At_Risk"] = (
            original_df["MonthlyCharges"] * original_df["Churn_Probability"]
        )

        # --------------------------------
        # Priority Score
        # --------------------------------
        max_clv = original_df["CLV"].max()

        original_df["Priority_Score"] = original_df.apply(
            lambda row: calculate_priority_score(
                row["CLV"], row["Churn_Probability"], max_clv
            ),
            axis=1,
        )

        # --------------------------------
        # Priority Level
        # --------------------------------

        original_df["Priority_Level"] = original_df["Priority_Score"].apply(
            priority_level
        )

        # --------------------------------
        # Retention Recommendation
        # --------------------------------

        original_df["Retention_Action"] = original_df.apply(smart_retention, axis=1)

        # --------------------------------
        # Recommendation Reason
        # --------------------------------

        original_df["Recommendation_Reason"] = original_df.apply(
            recommendation_reason, axis=1
        )

        # --------------------------------
        # Results
        # --------------------------------

        st.success("Prediction Completed Successfully")

        st.subheader("Prediction Results")

        st.dataframe(original_df, use_container_width=True)

        # --------------------------------
        # Download
        # --------------------------------

        csv = original_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Prediction Results",
            data=csv,
            file_name="customer_retention_results.csv",
            mime="text/csv",
        )
