import streamlit as st
import pandas as pd
import numpy as np

from utils import (
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
        # 1. Load data
        # --------------------------------

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(df.head(), use_container_width=True)

        # --------------------------------
        # 2. Convert numeric columns
        # --------------------------------

        numeric_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # --------------------------------
        # 3. Feature Engineering
        # --------------------------------

        df["AverageMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

        # --------------------------------
        # 4. Remove infinite values
        # --------------------------------

        df = df.replace([np.inf, -np.inf], np.nan)

        # --------------------------------
        # 5. Keep original data
        # --------------------------------

        original_df = df.copy()

        # --------------------------------
        # 6. Prepare model input
        # --------------------------------

        X = df.drop(columns=["customerID", "Churn"], errors="ignore")

        # --------------------------------
        # 7. Make sure columns match model & handle missing values
        # --------------------------------

        if hasattr(model, "feature_names_in_"):
            expected_columns = model.feature_names_in_
            # reindex with fill_value=0 so new/missing columns get 0 instead of NaN
            X = X.reindex(columns=expected_columns, fill_value=0)

        # Fill any remaining NaNs in numeric/categorical features
        X = X.fillna(0)

        # --------------------------------
        # 8. Prediction
        # --------------------------------

        prediction = model.predict(X)

        probability = model.predict_proba(X)[:, 1]

        # --------------------------------
        # 9. Add prediction
        # --------------------------------

        original_df["predicted_churn"] = prediction

        original_df["Churn_Probability"] = probability

        # --------------------------------
        # 10. CLV
        # --------------------------------

        original_df["CLV"] = original_df["MonthlyCharges"] * original_df["tenure"]

        # --------------------------------
        # 11. Revenue At Risk
        # --------------------------------

        original_df["Revenue_At_Risk"] = (
            original_df["MonthlyCharges"] * original_df["Churn_Probability"]
        )

        # --------------------------------
        # 12. Priority Score
        # --------------------------------

        max_clv = original_df["CLV"].max()

        original_df["Priority_Score"] = original_df.apply(
            lambda row: calculate_priority_score(
                row["CLV"], row["Churn_Probability"], max_clv
            ),
            axis=1,
        )

        # --------------------------------
        # 13. Priority Level
        # --------------------------------

        original_df["Priority_Level"] = original_df["Priority_Score"].apply(
            priority_level
        )

        # --------------------------------
        # 14. Retention Action
        # --------------------------------

        original_df["Retention_Action"] = original_df.apply(smart_retention, axis=1)

        # --------------------------------
        # 15. Recommendation Reason
        # --------------------------------

        original_df["Recommendation_Reason"] = original_df.apply(
            recommendation_reason, axis=1
        )

        # --------------------------------
        # 16. Results
        # --------------------------------

        st.success("Prediction Completed Successfully")

        st.subheader("Prediction Results")

        st.dataframe(original_df, use_container_width=True)

        # --------------------------------
        # 17. Download
        # --------------------------------

        csv = original_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download Prediction Results",
            data=csv,
            file_name="customer_retention_results.csv",
            mime="text/csv",
        )
