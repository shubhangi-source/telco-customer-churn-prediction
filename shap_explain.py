import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def shap_page(model):

    st.title("📈 Model Explainability (SHAP)")
    st.write("Understand feature impact on churn predictions.")

    # 1. Load Data
    df = pd.read_csv("data/processed_telco_churn.csv")

    # 2. Separate Features (X) and Target (y)
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    y = df["Churn"]

    # 3. Align columns to match the exact features the model was trained on
    if hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    # Clean up any NaN or infinite values
    X = X.fillna(0)

    # 4. Train-Test Split
    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Extract the underlying classifier model (last step of pipeline)
    if hasattr(model, "steps"):
        classifier = model.steps[-1][1]
    else:
        classifier = model

    # 6. Initialize SHAP Explainer directly on numerical features
    explainer = shap.LinearExplainer(classifier, X_train)
    shap_vals = explainer.shap_values(X_test)

    feature_names = list(X.columns)

    # --------------------------------
    # Global Feature Importance Plot
    # --------------------------------
    st.subheader("📊 Global Feature Importance")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    shap.summary_plot(shap_vals, X_test, feature_names=feature_names, show=False)
    st.pyplot(fig1, clear_figure=True)

    # --------------------------------
    # Local Individual Prediction Explanation
    # --------------------------------
    st.subheader("🔍 Individual Customer Explanation")

    customer_idx = st.number_input(
        "Select Customer Row Index",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1,
    )

    # Extract base value and SHAP values for selected index safely
    base_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )

    val_row = (
        shap_vals[1][customer_idx]
        if isinstance(shap_vals, list)
        else shap_vals[customer_idx]
    )

    exp = shap.Explanation(
        values=val_row,
        base_values=base_value,
        data=X_test.iloc[customer_idx].values,
        feature_names=feature_names,
    )

    # Render Waterfall Plot
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(exp, show=False)
    st.pyplot(fig2, clear_figure=True)
