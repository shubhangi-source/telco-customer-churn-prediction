import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt


def shap_page(best_lr):

    st.title("📈 SHAP Explainability")

    st.write("Understand how the machine learning model makes predictions.")

    # -----------------------------
    # Load Dataset
    # -----------------------------

    df = pd.read_csv("data/revenue_loss_estimator.csv")

    # Remove columns not used for prediction
    drop_cols = [
        "predicted_churn",
        "Churn_Probability",
        "CLV",
        "Priority_Score",
        "Priority_Level",
        "Retention_Action",
        "Recommendation_Reason",
        "Revenue_At_Risk",
    ]

    df = df.drop(columns=drop_cols, errors="ignore")
    from sklearn.model_selection import train_test_split

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logistic_model = best_lr.named_steps["model"]

    scaler = best_lr.named_steps["scaler"]

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler = best_lr.named_steps["scaler"]

    # -----------------------------
    # SHAP Explainer
    # -----------------------------

    explainer = shap.LinearExplainer(logistic_model, X_train_scaled)

    shap_values = explainer.shap_values(X_test_scaled)

    # -----------------------------
    # Summary Plot
    # -----------------------------

    st.subheader("Feature Importance")

    fig = plt.figure(figsize=(8, 4))

    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)

    st.pyplot(fig)

    plt.clf()

    # -----------------------------
    # Bar Plot
    # -----------------------------

    # -----------------------------
    # Individual Prediction
    # -----------------------------

    st.subheader("Explain Individual Customer")

    customer_index = st.number_input(
        "Customer Index", min_value=0, max_value=len(X) - 1, value=0
    )

    fig = plt.figure(figsize=(8, 4))

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[customer_index],
            base_values=explainer.expected_value,
            data=X_test.iloc[customer_index],
            feature_names=X.columns,
        ),
        show=False,
    )

    st.pyplot(fig)

    plt.clf()
