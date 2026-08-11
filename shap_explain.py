import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


@st.cache_data
def load_data():
    df = pd.read_csv("data/processed_telco_churn.csv")
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    y = df["Churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


@st.cache_resource
def compute_shap(_model, _X_train, _X_test):
    # Pull the preprocessor and classifier out of the pipeline
    preprocessor = _model.named_steps["preprocessor"]
    logistic_model = _model.named_steps["model"]

    X_train_processed = preprocessor.transform(_X_train)
    X_test_processed = preprocessor.transform(_X_test)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.LinearExplainer(logistic_model, X_train_processed)
    shap_values = explainer.shap_values(X_test_processed)

    return explainer, shap_values, X_test_processed, feature_names


def shap_page(model):
    st.title("📈 Model Explainability (SHAP)")
    st.write("Understand which features increase or decrease churn probability.")

    # 1. Load data
    try:
        X_train, X_test, y_train, y_test = load_data()
    except FileNotFoundError:
        st.error("Dataset not found: data/processed_telco_churn.csv")
        return

    # 2. Small samples (SHAP is expensive)
    X_train_sample = X_train.sample(n=min(100, len(X_train)), random_state=42)
    X_test_sample = X_test.sample(n=min(100, len(X_test)), random_state=42).reset_index(
        drop=True
    )

    st.info("Calculating SHAP values. Please wait...")

    try:
        explainer, shap_values, X_test_processed, feature_names = compute_shap(
            model, X_train_sample, X_test_sample
        )
    except Exception as e:
        st.error(f"Unable to calculate SHAP values: {e}")
        st.info(
            "Make sure the model passed in is the fitted pipeline with "
            "'preprocessor' and 'model' steps."
        )
        return

    # 3. Global feature importance (beeswarm)
    st.subheader("📊 Global Feature Importance")
    st.write("Which features have the biggest overall impact on churn predictions.")
    fig1, _ = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_test_processed, feature_names=feature_names, show=False
    )
    st.pyplot(fig1, clear_figure=True)
    plt.close(fig1)

    # 5. Individual customer explanation
    st.subheader("🔍 Individual Customer Explanation")
    st.write("Select a customer to see why the model predicted their churn risk.")

    customer_idx = st.number_input(
        "Select Customer Row Index",
        min_value=0,
        max_value=len(X_test_sample) - 1,
        value=0,
        step=1,
    )

    st.write("### Customer Information")
    st.dataframe(X_test_sample.iloc[customer_idx].to_frame(name="Value"))

    st.write("### Why did the model make this prediction?")
    try:
        row = X_test_processed[customer_idx]
        row_dense = row.toarray().flatten() if hasattr(row, "toarray") else row

        fig3 = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[customer_idx],
                base_values=explainer.expected_value,
                data=row_dense,
                feature_names=feature_names,
            ),
            show=False,
        )
        st.pyplot(fig3, clear_figure=True)
        plt.close(fig3)
    except Exception as e:
        st.error(f"Could not create individual SHAP explanation: {e}")

    # 6. Explanation text
    st.subheader("💡 How to Interpret SHAP")
    st.markdown("""
        - 🔴 **Positive SHAP value** → pushes prediction toward higher churn probability.
        - 🔵 **Negative SHAP value** → pushes prediction toward lower churn probability.
        - 📊 **Larger absolute value** → stronger impact on the prediction.

        Example: a large positive SHAP value for `Contract_Month-to-month` means that
        feature is increasing the customer's predicted churn risk. A negative SHAP
        value for `tenure` means longer tenure is reducing predicted churn risk.
        """)
