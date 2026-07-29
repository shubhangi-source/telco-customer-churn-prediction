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

    # Critical Priority
    if priority == "Critical":

        if clv >= 5000:
            return "Immediate Call + 30% Discount + Dedicated Relationship Manager"

        elif tenure < 12:
            return "Free 2-Month Subscription + Onboarding Support"

        elif monthly >= 80:
            return "25% Discount + Premium Support"

        else:
            return "20% Discount + Loyalty Rewards"

    # High Priority
    elif priority == "High":

        if clv >= 5000:
            return "Premium Support + Service Upgrade"

        elif monthly >= 80:
            return "20% Discount"

        else:
            return "15% Discount + Loyalty Points"

    # Medium Priority
    elif priority == "Medium":

        if tenure < 12:
            return "Welcome Offer + Personalized Email"

        else:
            return "10% Discount + Plan Upgrade Recommendation"

    # Low Priority
    else:

        if clv >= 5000:
            return "Loyalty Rewards + Thank You Coupon"

        else:
            return "Regular Promotional Email"
