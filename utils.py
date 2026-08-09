import pandas as pd
import numpy as np


def load_data():
    """
    Load revenue loss estimator dataset.
    """

    df = pd.read_csv("../data/revenue_loss_estimator.csv")

    return df


def calculate_average_monthly_spend(total_charges, tenure):
    """
    Calculate Average Monthly Spend.
    """

    return total_charges / (tenure + 1)


def calculate_clv(monthly_charges, tenure):
    """
    Customer Lifetime Value
    """

    return monthly_charges * tenure


def calculate_revenue_at_risk(monthly_charges, probability):
    """
    Expected Revenue Loss
    """

    return monthly_charges * probability


def calculate_priority_score(clv, probability, max_clv=8550):
    """
    Priority Score
    """
    clv_scaled = np.clip(clv / max_clv, 0, 1)
    priority_score = ((probability * 0.7) + (clv_scaled * 0.3)) * 100
    return round(priority_score, 2)


def priority_level(score):
    if score >= 80:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


def smart_retention(row):

    priority = row["Priority_Level"]
    clv = row["CLV"]
    tenure = row["tenure"]
    monthly = row["MonthlyCharges"]
    churn = row["Churn_Probability"]

    # =========================
    # CRITICAL
    # =========================

    if priority == "Critical":

        if churn >= 0.95 and clv >= 5000:
            return (
                "Immediate Manager Call + 30% Discount + Dedicated Relationship Manager"
            )

        elif tenure < 6:
            return "Free 3-Month Subscription + Personal Onboarding"

        elif row["Contract"] == "Month-to-month":
            return "Offer Free Upgrade to One-Year Contract"

        elif monthly >= 100:
            return "25% Discount + Premium Support"

        elif row["OnlineSecurity"] == "No" and row["TechSupport"] == "No":
            return "Free Online Security & Tech Support for 6 Months"

        elif row["OnlineSecurity"] == "No":
            return "Free Online Security for 6 Months"

        elif row["TechSupport"] == "No":
            return "Free Technical Support for 6 Months"

        elif row["OnlineBackup"] == "No":
            return "Free Cloud Backup for 3 Months"

        elif row["InternetService"] == "Fiber optic":
            return "Free Speed Upgrade + Premium Support"

        elif row["PaymentMethod"] == "Electronic check":
            return "Switch to Auto-Pay and Get ₹500 Cashback"

        else:
            return "20% Loyalty Discount"

    # =========================
    # HIGH
    # =========================

    elif priority == "High":

        if clv >= 5000:
            return "Premium Support + Free Service Upgrade"

        elif tenure < 12:
            return "Welcome Back Offer + Free Installation"

        elif monthly >= 100:
            return "25% Discount"

        elif row["Contract"] == "Month-to-month":
            return "Offer One-Year Contract with Discount"

        elif row["InternetService"] == "Fiber optic":
            return "Free Speed Upgrade"

        elif row["OnlineSecurity"] == "No":
            return "Free Online Security Trial"

        elif row["TechSupport"] == "No":
            return "Free Technical Support Trial"

        elif row["OnlineBackup"] == "No":
            return "Free Cloud Backup Trial"

        elif row["PaymentMethod"] == "Electronic check":
            return "Auto-Pay Cashback Offer"

        elif row["SeniorCitizen"] == 1:
            return "Senior Citizen Special Discount"

        else:
            return "15% Discount + Loyalty Points"

    # =========================
    # MEDIUM
    # =========================

    elif priority == "Medium":

        if tenure < 12:
            return "Welcome Offer + Personalized Email"

        elif monthly >= 90:
            return "10% Discount on Monthly Bill"

        elif row["Contract"] == "Month-to-month":
            return "Recommend Annual Contract"

        elif row["OnlineSecurity"] == "No":
            return "Free Online Security Trial"

        elif row["TechSupport"] == "No":
            return "Free Technical Support Trial"

        elif row["OnlineBackup"] == "No":
            return "Free Cloud Backup Trial"

        elif row["PaymentMethod"] == "Electronic check":
            return "Recommend Auto-Pay"

        elif row["SeniorCitizen"] == 1:
            return "Senior Citizen Discount Plan"

        else:
            return "Personalized Marketing Email"

    # =========================
    # LOW
    # =========================

    else:

        if clv >= 5000:
            return "VIP Loyalty Rewards"

        elif tenure >= 60:
            return "Anniversary Reward Coupon"

        elif row["Contract"] == "Two year":
            return "Early Renewal Bonus"

        elif row["InternetService"] == "Fiber optic":
            return "Free Streaming Service Trial"

        elif row["PaymentMethod"] == "Credit card (automatic)":
            return "Cashback Reward"

        else:
            return "Regular Promotional Email"


def recommendation_reason(row):

    priority = row["Priority_Level"]
    clv = row["CLV"]
    tenure = row["tenure"]
    monthly = row["MonthlyCharges"]
    churn = row["Churn_Probability"]

    # =========================
    # CRITICAL
    # =========================

    if priority == "Critical":

        if churn >= 0.95 and clv >= 5000:
            return (
                "Customer has very high churn probability and very high lifetime value."
            )

        elif tenure < 6:
            return "Customer is new and has a high risk of churn."

        elif row["Contract"] == "Month-to-month":
            return (
                "Customer is on a month-to-month contract, which has higher churn risk."
            )

        elif monthly >= 100:
            return "Customer has high monthly charges."

        elif row["OnlineSecurity"] == "No" and row["TechSupport"] == "No":
            return "Customer is not subscribed to Online Security and Tech Support."

        elif row["OnlineSecurity"] == "No":
            return "Customer is not using Online Security."

        elif row["TechSupport"] == "No":
            return "Customer is not using Technical Support."

        elif row["OnlineBackup"] == "No":
            return "Customer is not using Online Backup."

        elif row["InternetService"] == "Fiber optic":
            return "Customer uses Fiber optic internet, which showed higher churn behaviour in the dataset."

        elif row["PaymentMethod"] == "Electronic check":
            return (
                "Customer uses Electronic Check, which showed relatively higher churn."
            )

        else:
            return "Customer has been identified as a critical retention case."

    # =========================
    # HIGH
    # =========================

    elif priority == "High":

        if clv >= 5000:
            return "Customer has high lifetime value and should be retained."

        elif tenure < 12:
            return "Customer is still in the early stage of the relationship."

        elif monthly >= 100:
            return "Customer has high monthly charges."

        elif row["Contract"] == "Month-to-month":
            return "Customer is using a month-to-month contract."

        elif row["InternetService"] == "Fiber optic":
            return "Customer uses Fiber optic internet."

        elif row["OnlineSecurity"] == "No":
            return "Customer has not subscribed to Online Security."

        elif row["TechSupport"] == "No":
            return "Customer has not subscribed to Technical Support."

        elif row["OnlineBackup"] == "No":
            return "Customer has not subscribed to Online Backup."

        elif row["PaymentMethod"] == "Electronic check":
            return "Customer uses Electronic Check payment."

        elif row["SeniorCitizen"] == 1:
            return "Customer is a senior citizen and may benefit from targeted offers."

        else:
            return "Customer has been identified as a high-priority retention case."

    # =========================
    # MEDIUM
    # =========================

    elif priority == "Medium":

        if tenure < 12:
            return "Customer is relatively new."

        elif monthly >= 90:
            return "Customer has relatively high monthly charges."

        elif row["Contract"] == "Month-to-month":
            return "Customer is using a month-to-month contract."

        elif row["OnlineSecurity"] == "No":
            return "Customer is not using Online Security."

        elif row["TechSupport"] == "No":
            return "Customer is not using Technical Support."

        elif row["OnlineBackup"] == "No":
            return "Customer is not using Online Backup."

        elif row["PaymentMethod"] == "Electronic check":
            return "Customer uses Electronic Check payment."

        elif row["SeniorCitizen"] == 1:
            return "Customer may benefit from senior-focused offers."

        else:
            return "Customer has moderate churn risk."

    # =========================
    # LOW
    # =========================

    else:

        if clv >= 5000:
            return "Customer is highly valuable and currently has low churn priority."

        elif tenure >= 60:
            return "Customer has been with the company for a long time."

        elif row["Contract"] == "Two year":
            return "Customer has a long-term contract, which generally indicates stronger retention."

        elif row["InternetService"] == "Fiber optic":
            return "Customer uses Fiber internet but currently has low churn priority."

        elif row["PaymentMethod"] == "Credit card (automatic)":
            return (
                "Customer uses automatic payment and currently has low churn priority."
            )

        else:
            return "Customer currently has low churn risk."
