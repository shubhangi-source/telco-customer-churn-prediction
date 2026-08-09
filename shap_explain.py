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
    # 2. Prepare Data
    # -----------------------------

    df = df.drop(
        columns=[
            "predicted_churn",
            "Churn_Probability",
            "CLV",
            "Priority_Score",
            "Priority_Level",
            "Retention_Action",
            "Recommendation_Reason",
            "Revenue_At_Risk",
            "customerID",
        ],
        errors="ignore",
    )

    X = df.drop("Churn", axis=1)

    y = df["Churn"]

    # Convert Yes/No to 0/1
    if y.dtype == "object":
        y = y.map({"No": 0, "Yes": 1})

    # -----------------------------
    # 3. Train Test Split
    # -----------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # 4. Get Preprocessor
    # -----------------------------

    preprocessor = model.named_steps["preprocessor"]

    logistic_model = model.named_steps["model"]

    # -----------------------------
    # 5. Transform Data
    # -----------------------------

    X_train = preprocessor.transform(X_train)

    X_test = preprocessor.transform(X_test)

    # Get feature names
    feature_names = preprocessor.get_feature_names_out()

    # -----------------------------
    # 6. SHAP
    # -----------------------------

    explainer = shap.LinearExplainer(logistic_model, X_train)

    shap_values = explainer.shap_values(X_test)

    # -----------------------------
    # 7. Feature Importance
    # -----------------------------

    st.subheader("📊 Feature Importance")

    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)

    st.pyplot(plt.gcf(), clear_figure=True)

    # -----------------------------
    # 8. Individual Customer
    # -----------------------------

    st.subheader("🔍 Individual Customer")

    index = st.number_input(
        "Customer Index", min_value=0, max_value=len(X_test) - 1, value=0
    )

    index = int(index)

    # -----------------------------
    # 9. Waterfall Plot
    # -----------------------------

    explanation = shap.Explanation(
        values=shap_values[index],
        base_values=explainer.expected_value,
        data=X_test[index],
        feature_names=feature_names,
    )

    shap.waterfall_plot(explanation, show=False)

    st.pyplot(plt.gcf(), clear_figure=True)
