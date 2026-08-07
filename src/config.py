import streamlit as st


def set_page_config():
    """
    Configure Streamlit page settings.
    """

    st.set_page_config(
        page_title="Customer Churn Prediction & Retention System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# -------------------------------
# Project Information
# -------------------------------

APP_TITLE = "📊 Customer Churn Prediction & Retention System"

APP_DESCRIPTION = """
This application predicts customer churn using a trained Logistic Regression model
and provides business insights ."""


MODEL_NAME = "Logistic Regression Classifier"

MODEL__ACCURACY = " 74.3%"

MODEL_RECALL = "78.6%"

MODEL_ROC_AUC = 0.845

DATASET_NAME = "Telco Customer Churn Dataset"

DEVELOPER = "Shubhangi Singh"


# -------------------------------
# File Paths
# -------------------------------

MODEL_PATH = "models/customer_churn_model.pkl"

DATA_PATH = "data/revenue_loss_estimator.csv"

LOGO_PATH = "images/logo.png"


# -------------------------------
# Theme Colors
# -------------------------------

SUCCESS_COLOR = "#28a745"

WARNING_COLOR = "#ffc107"

DANGER_COLOR = "#dc3545"

PRIMARY_COLOR = "#1f77b4"
