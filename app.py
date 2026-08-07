import streamlit as st

# -----------------------------
# Import Configuration
# -----------------------------
from src.config import set_page_config, APP_TITLE, APP_DESCRIPTION

# -----------------------------
# Import Model Loader
# -----------------------------
from src.load_model import load_model

# -----------------------------
# Import Pages
# -----------------------------
from src.dashboard import show_dashboard
from src.predict_customer import predict_customer_page
from src.batch_prediction import batch_prediction_page
from src.shap_explain import shap_page

# -----------------------------
# Configure Streamlit Page
# -----------------------------
set_page_config()

# -----------------------------
# Load Model
# -----------------------------
best_lr = load_model()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📋 Navigation")

page = st.sidebar.radio(
    "Choose a Page",
    (
        "🏠 Home",
        "📊 Dashboard",
        "🔍 Predict Customer",
        "📂 Batch Prediction",
        "📈 SHAP Explainability",
        "ℹ️ About",
    ),
)

# =============================
# HOME PAGE
# =============================

if page == "🏠 Home":

    st.title(APP_TITLE)

    st.write(APP_DESCRIPTION)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📌 Prediction Features")

        st.markdown("""
        ✅ Customer Churn Prediction

        ✅ Churn Probability

        ✅ Customer Lifetime Value (CLV)

        ✅ Revenue at Risk

        ✅ Priority Score
        """)

    with col2:

        st.subheader("💼 Business Intelligence")

        st.markdown("""
        ✅ Priority Level

        ✅ Smart Retention Recommendation

        ✅ Recommedation Reason

        ✅ Batch Prediction

        ✅ SHAP Explainability
        """)

# =============================
# DASHBOARD
# =============================

elif page == "📊 Dashboard":

    show_dashboard()

# =============================
# PREDICT CUSTOMER
# =============================

elif page == "🔍 Predict Customer":

    predict_customer_page(best_lr)

# =============================
# BATCH PREDICTION
# =============================

elif page == "📂 Batch Prediction":

    batch_prediction_page(best_lr)

# =============================
# SHAP EXPLAINABILITY
# =============================

elif page == "📈 SHAP Explainability":

    shap_page(best_lr)

# =============================
# ABOUT
# =============================

elif page == "ℹ️ About":

    st.title("About This Project")

    st.write("""
### AI-Powered Customer Churn Prediction & Retention System

This application predicts customer churn using a Machine Learning model
and provides business insights to help companies reduce customer attrition.

### Features

- Customer Churn Prediction
- Churn Probability
- Revenue At Risk
- Customer Lifetime Value (CLV)
- Priority Score
- Priority Level
- Smart Retention Recommendation
- Recommendation Reason
- SHAP Explainability
- Batch Prediction

### Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Logistic regression
- Matplotlib
- Seaborn
- Plotly
- SHAP
- Joblib

### Developer

Shubhangi Singh
""")
