import streamlit as st
import joblib


@st.cache_resource
def load_model():
    """
    Load the trained Logistic Regression model.
    """

    best_lr = joblib.load("models/customer_churn_model.pkl")

    return best_lr
