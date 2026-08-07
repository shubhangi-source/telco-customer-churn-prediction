import streamlit as st
import pandas as pd
import plotly.express as px


def show_dashboard():

    st.title("📊 Business Dashboard")

    # Load data
    df = pd.read_csv("data/03_results/revenue_loss_estimator.csv")

    # ------------------------------
    # KPI Cards
    # ------------------------------

    total_customers = len(df)

    churn_customers = df["predicted_churn"].sum()

    churn_rate = (churn_customers / total_customers) * 100

    revenue_risk = df["Revenue_At_Risk"].sum()

    avg_probability = df["Churn_Probability"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Customers", total_customers)

    col2.metric("Predicted Churn", int(churn_customers))

    col3.metric("Churn Rate", f"{churn_rate:.2f}%")

    col4.metric("Revenue At Risk", f"${revenue_risk:.2f}")

    col5.metric("Average Probability", f"{avg_probability:.2%}")

    st.markdown("---")

    # ------------------------------
    # Churn Distribution
    # ------------------------------

    st.subheader("Churn Distribution")

    churn_fig = px.pie(
        df,
        names="predicted_churn",
        color="predicted_churn",
        color_discrete_map={0: "#34D399", 1: "#F87171"},
        hole=0.4,
        title="Predicted Churn Distribution",
    )

    st.plotly_chart(churn_fig, use_container_width=True)

    st.success(
        "💡 **Business Insight:** Approximately **40% of customers are predicted to churn**. "
        "These customers should be prioritized with targeted retention campaigns to reduce customer loss."
    )

    # ------------------------------
    # Priority Level
    # ------------------------------

    st.subheader("Customer Priority Levels")

    color_map = {"High": "#F87171", "Medium": "#FBBF24", "Low": "#34D399"}
    priority_fig = px.bar(
        df["Priority_Level"].value_counts().reset_index(),
        x="Priority_Level",
        y="count",
        color="Priority_Level",
        color_discrete_map=color_map,
        title="Priority Level Distribution",
    )

    st.plotly_chart(priority_fig, use_container_width=True)

    st.success(
        "💡 **Business Insight:** Most customers fall into the **Low Priority** category, "
        "while **High Priority** customers require immediate retention efforts to maximize business impact."
    )
    # ------------------------------
    # Revenue At Risk
    # ------------------------------

    st.subheader("Revenue At Risk")

    revenue_fig = px.histogram(
        df,
        x="Revenue_At_Risk",
        nbins=30,
        color_discrete_sequence=["#818CF8"],
        title="Revenue At Risk Distribution",
    )

    st.plotly_chart(revenue_fig, use_container_width=True)

    st.success(
        "💡 **Business Insight:** Customers with **higher revenue at risk** represent the greatest potential financial loss. "
        "Prioritizing these customers can significantly reduce revenue leakage."
    )

    # ------------------------------
    # Top 10 High Risk Customers
    # ------------------------------

    top10 = df.sort_values(by="Priority_Score", ascending=False).head(10)

    st.subheader("Top 10 High Risk Customers")
    st.dataframe(
        top10[
            [
                "Churn_Probability",
                "CLV",
                "Priority_Score",
                "Priority_Level",
                "Revenue_At_Risk",
                "Retention_Action",
                "Recommendation_Reason",
            ]
        ],
        use_container_width=True,
    )

    # ------------------------------
    # Download Button
    # ------------------------------

    st.download_button(
        label="⬇ Download Dashboard Data",
        data=df.to_csv(index=False),
        file_name="customer_retention_dashboard.csv",
        mime="text/csv",
    )
