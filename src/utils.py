import pandas as pd


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
    clv_scaled = min(clv / max_clv, 1.0)
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

    # ===========================
    # CRITICAL PRIORITY
    # ===========================

    if priority == "Critical":

        if churn >= 0.95 and clv >= 5000:
            return (
                "Immediate Manager Call + 30% Discount + Dedicated Relationship Manager"
            )

        elif tenure < 6:
            return "Free 3-Month Subscription + Personal Onboarding"

        elif row["Contract_One year"] == 0 and row["Contract_Two year"] == 0:
            return "Offer Free Upgrade to One-Year Contract"

        elif monthly >= 100:
            return "25% Discount + Premium Support"

        elif row["OnlineSecurity_Yes"] == 0 and row["TechSupport_Yes"] == 0:
            return "Free Online Security & Tech Support for 6 Months"

        elif row["OnlineSecurity_Yes"] == 0:
            return "Free Online Security for 6 Months"

        elif row["TechSupport_Yes"] == 0:
            return "Free Technical Support for 6 Months"

        elif row["OnlineBackup_Yes"] == 0:
            return "Free Cloud Backup for 3 Months"

        elif row["InternetService_Fiber optic"] == 1:
            return "Free Speed Upgrade + Premium Support"

        elif row["PaymentMethod_Electronic check"] == 1:
            return "Switch to Auto-Pay and Get ₹500 Cashback"

        else:
            return "20% Loyalty Discount"

        # ===========================
        # HIGH PRIORITY
        # ===========================

    elif priority == "High":

        if clv >= 5000:
            return "Premium Support + Free Service Upgrade"

        elif tenure < 12:
            return "Welcome Back Offer + Free Installation"

        elif monthly >= 100:
            return "25% Discount"

        elif row["Contract_One year"] == 0 and row["Contract_Two year"] == 0:
            return "Offer One-Year Contract with Discount"

        elif row["InternetService_Fiber optic"] == 1:
            return "Free Speed Upgrade"

        elif row["OnlineSecurity_Yes"] == 0:
            return "Free Online Security Trial"

        elif row["TechSupport_Yes"] == 0:
            return "Free Technical Support Trial"

        elif row["OnlineBackup_Yes"] == 0:
            return "Free Cloud Backup Trial"

        elif row["PaymentMethod_Electronic check"] == 1:
            return "Auto-Pay Cashback Offer"

        elif row["SeniorCitizen"] == 1:
            return "Senior Citizen Special Discount"

        else:
            return "15% Discount + Loyalty Points"

        # ===========================
        # MEDIUM PRIORITY
        # ===========================

    elif priority == "Medium":

        if tenure < 12:
            return "Welcome Offer + Personalized Email"

        elif monthly >= 90:
            return "10% Discount on Monthly Bill"

        elif row["Contract_One year"] == 0 and row["Contract_Two year"] == 0:
            return "Recommend Annual Contract"

        elif row["OnlineSecurity_Yes"] == 0:
            return "Free Online Security Trial"

        elif row["TechSupport_Yes"] == 0:
            return "Free Technical Support Trial"

        elif row["OnlineBackup_Yes"] == 0:
            return "Free Cloud Backup Trial"

        elif row["PaymentMethod_Electronic check"] == 1:
            return "Recommend Auto-Pay"

        elif row["SeniorCitizen"] == 1:
            return "Senior Citizen Discount Plan"

        else:
            return "Personalized Marketing Email"

        # ===========================
        # LOW PRIORITY
        # ===========================

    else:

        if clv >= 5000:
            return "VIP Loyalty Rewards"

        elif tenure >= 60:
            return "Anniversary Reward Coupon"

        elif row["Contract_Two year"] == 1:
            return "Early Renewal Bonus"

        elif row["InternetService_Fiber optic"] == 1:
            return "Free Streaming Service Trial"

        elif row["PaymentMethod_Credit card (automatic)"] == 1:
            return "Cashback Reward"

        else:
            return "Regular Promotional Email"


def recommendation_reason(row):

    priority = row["Priority_Level"]
    clv = row["CLV"]
    tenure = row["tenure"]
    monthly = row["MonthlyCharges"]
    churn = row["Churn_Probability"]

    # ===========================
    # CRITICAL PRIORITY
    # ===========================

    if priority == "Critical":

        if churn >= 0.95 and clv >= 5000:
            return (
                "Customer has very high churn probability and very high lifetime value."
            )

        elif tenure < 6:
            return "Customer is new and has a high risk of churn."

        elif row["Contract_One year"] == 0 and row["Contract_Two year"] == 0:
            return "Customer is on a month-to-month contract, which has a higher churn risk."

        elif monthly >= 100:
            return "Customer pays a high monthly bill, increasing churn risk."

        elif row["OnlineSecurity_Yes"] == 0 and row["TechSupport_Yes"] == 0:
            return "Customer is not subscribed to Online Security and Tech Support."

        elif row["OnlineSecurity_Yes"] == 0:
            return "Customer is not using the Online Security service."

        elif row["TechSupport_Yes"] == 0:
            return "Customer is not using the Technical Support service."

        elif row["OnlineBackup_Yes"] == 0:
            return "Customer is not using the Online Backup service."

        elif row["InternetService_Fiber optic"] == 1:
            return "Fiber optic customers have shown higher churn behaviour."

        elif row["PaymentMethod_Electronic check"] == 1:
            return "Electronic Check payment customers have relatively higher churn."

        else:
            return "Critical customer identified for immediate retention."

    # ===========================
    # HIGH PRIORITY
    # ===========================

    elif priority == "High":

        if clv >= 5000:
            return "Customer has high lifetime value and should be retained."

        elif tenure < 12:
            return "Customer is still in the early stage of the relationship."

        elif monthly >= 100:
            return "Customer has high monthly charges."

        elif row["Contract_One year"] == 0 and row["Contract_Two year"] == 0:
            return "Customer is on a month-to-month contract."

        elif row["InternetService_Fiber optic"] == 1:
            return "Fiber optic customers generally have higher churn."

        elif row["OnlineSecurity_Yes"] == 0:
            return "Customer has not subscribed to Online Security."

        elif row["TechSupport_Yes"] == 0:
            return "Customer has not subscribed to Technical Support."

        elif row["OnlineBackup_Yes"] == 0:
            return "Customer has not subscribed to Online Backup."

        elif row["PaymentMethod_Electronic check"] == 1:
            return "Electronic Check payment is associated with higher churn."

        elif row["SeniorCitizen"] == 1:
            return "Customer is a senior citizen and eligible for special offers."

        else:
            return "High-priority customer identified by the churn model."

    # ===========================
    # MEDIUM PRIORITY
    # ===========================

    elif priority == "Medium":

        if tenure < 12:
            return "Customer is relatively new."

        elif monthly >= 90:
            return "Customer pays above-average monthly charges."

        elif row["Contract_One year"] == 0 and row["Contract_Two year"] == 0:
            return "Customer is using a month-to-month contract."

        elif row["OnlineSecurity_Yes"] == 0:
            return "Customer is not using Online Security."

        elif row["TechSupport_Yes"] == 0:
            return "Customer is not using Technical Support."

        elif row["OnlineBackup_Yes"] == 0:
            return "Customer is not using Online Backup."

        elif row["PaymentMethod_Electronic check"] == 1:
            return "Electronic Check payment method has moderate churn risk."

        elif row["SeniorCitizen"] == 1:
            return "Customer qualifies for senior citizen benefits."

        else:
            return "Customer has moderate churn risk."

    # ===========================
    # LOW PRIORITY
    # ===========================

    else:

        if clv >= 5000:
            return "Customer is highly valuable and already loyal."

        elif tenure >= 60:
            return "Customer has been with the company for a long time."

        elif row["Contract_Two year"] == 1:
            return "Long-term contract significantly reduces churn."

        elif row["InternetService_Fiber optic"] == 1:
            return "Customer uses Fiber Internet but currently has low churn risk."

        elif row["PaymentMethod_Credit card (automatic)"] == 1:
            return "Automatic payment indicates stable customer behaviour."

        else:
            return "Customer currently has low churn risk."
