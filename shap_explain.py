import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def shap_page(model):

    st.title("📈 SHAP Explainability")
    st.write("Understand how the model makes predictions.")

    # -----------------------------
    # 1. Load Data
    # -----------------------------

    df = pd.read_csv("data/revenue_loss_estimator.csv")

    # -----------------------------
    # 2. Remove extra columns
    # -----------------------------

    drop_columns = [
        "predicted_churn",
        "Churn_Probability",
        "CLV",
        "Priority_Score",
        "Priority_Level",
        "Retention_Action",
        "Recommendation_Reason",
        "Revenue_At_Risk",
        "customerID",
    ]

    df = df.drop(columns=drop_columns, errors="ignore")

    # -----------------------------
    # 3. X and y
    # -----------------------------

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # -----------------------------
    # 4. Train Test Split
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # 5. Get preprocessing
    # -----------------------------

    preprocessor = model.named_steps["preprocessor"]

    logistic_model = model.named_steps["model"]

    # -----------------------------
    # 6. Transform data
    # -----------------------------

    X_train_processed = preprocessor.transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    # -----------------------------
    # 7. Feature names
    # -----------------------------

    feature_names = preprocessor.get_feature_names_out()

    # -----------------------------
    # 8. SHAP
    # -----------------------------

    explainer = shap.LinearExplainer(logistic_model, X_train_processed)

    shap_values = explainer.shap_values(X_test_processed)

    # -----------------------------
    # 9. Summary Plot
    # -----------------------------

    st.subheader("📊 Feature Importance")

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values, X_test_processed, feature_names=feature_names, show=False
    )

    st.pyplot(plt.gcf(), clear_figure=True)

    # -----------------------------
    # 10. Individual Customer
    # -----------------------------

    st.subheader("🔍 Individual Customer")

    index = st.number_input(
        "Customer Index", min_value=0, max_value=len(X_test_processed) - 1, value=0
    )

    index = int(index)

    # -----------------------------
    # 11. Customer data
    # -----------------------------

    customer_data = X_test_processed[index]

    # Convert sparse row to array
    if hasattr(customer_data, "toarray"):
        customer_data = customer_data.toarray().flatten()

    # -----------------------------
    # 12. Waterfall
    # -----------------------------

    explanation = shap.Explanation(
        values=shap_values[index],
        base_values=explainer.expected_value,
        data=customer_data,
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 6))

    shap.waterfall_plot(explanation, show=False)

    st.pyplot(plt.gcf(), clear_figure=True)
