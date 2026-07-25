import os
import json
import google.generativeai as genai
from backend.schemas import AiStrategyRequest, AiStrategyResponse

def generate_ai_retention_strategy(req: AiStrategyRequest) -> AiStrategyResponse:
    # Check for API key
    api_key = req.gemini_api_key or os.environ.get("GEMINI_API_KEY") or "AIzaSyCMr48MKHcNkx1HCToTRDaLqwSHn34TMho"

    top_drivers_str = ", ".join(req.top_shap_drivers)
    prompt = f"""
    You are an expert Enterprise Customer Retention Strategist and Chief Marketing Officer.
    Analyze the following customer profile and generate a segment-aware retention offer and strategy:

    - Customer Segment Cluster: {req.cluster_label}
    - Churn Probability: {req.churn_probability}% (Risk Tier: {req.risk_tier})
    - Customer Lifetime Value (CLV): ${req.clv_estimate:,.2f}
    - Tenure: {req.tenure} months
    - Contract Type: {req.Contract}
    - Monthly Charges: ${req.MonthlyCharges}
    - Key Churn Risk Drivers (SHAP Top Factors): {top_drivers_str}
    - Custom Context / Notes: {req.custom_notes or 'None'}

    Task Requirements:
    1. Formulate a targeted, segment-aware retention plan specifically crafted for a customer in the '{req.cluster_label}' segment.
    2. Recommend a specific discount or value-add coupon (e.g. 20% off for 6 months, free TechSupport for 1 year, upgrade to Annual contract with $50 bill credit).
    3. Estimate the intervention cost, expected risk reduction percentage, and calculated ROI.
    4. Provide executive summary notes for customer support representatives.

    Output format MUST be valid JSON with keys:
    strategy_title, executive_summary, retention_offer, discount_coupon, estimated_cost, expected_risk_reduction, estimated_roi, strategy_markdown
    """

    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = gemini_model.generate_content(prompt)
        text_content = response.text.strip()
        
        # Try to parse JSON from Markdown block if returned
        if "```json" in text_content:
            text_content = text_content.split("```json")[1].split("```")[0].strip()
        elif "```" in text_content:
            text_content = text_content.split("```")[1].split("```")[0].strip()
            
        data = json.loads(text_content)
        return AiStrategyResponse(**data)
    except Exception as e:
        print(f"Gemini API fallback triggered ({e})")
        # Rule-based fallback strategy
        discount_code = f"SAVE-{req.cluster_label.split()[0].upper()}-2026"
        est_cost = round(req.MonthlyCharges * 0.20 * 6, 2)
        exp_reduction = round(min(35.0, req.churn_probability * 0.45), 1)
        est_roi = round(((req.clv_estimate * (exp_reduction / 100.0)) - est_cost) / max(1, est_cost) * 100, 1)

        summary_markdown = f"""
### 🎯 Segment-Aware Retention Plan: {req.cluster_label}

**Executive Strategy Summary:**
Customer exhibits **{req.churn_probability}% churn risk** ({req.risk_tier} Tier) primarily driven by **{top_drivers_str}**. Because they belong to the **{req.cluster_label}** cluster, a standard discount alone is insufficient; they require a high-touch value stabilization bundle.

#### 💡 Personalized Retention Offer:
- **Primary Offer:** 20% Monthly Charge Discount for 6 Months upon upgrading to 1-Year Contract.
- **Value Add:** Complimentary 12-Month **TechSupport & Security Suite** package.
- **Promo Code:** `{discount_code}`

#### 📊 Financial & ROI Impact:
- **Estimated Retention Cost:** ${est_cost}
- **Expected Risk Reduction:** -{exp_reduction}% Churn Risk
- **Estimated Campaign ROI:** +{est_roi}%
        """

        return AiStrategyResponse(
            strategy_title=f"Custom Retention Strategy for {req.cluster_label}",
            executive_summary=f"Segment-targeted intervention for {req.risk_tier} Risk customer with ${req.clv_estimate:,.2f} CLV.",
            retention_offer=f"20% discount on monthly charges for 6 months + free TechSupport with 1-year contract extension.",
            discount_coupon=discount_code,
            estimated_cost=est_cost,
            expected_risk_reduction=exp_reduction,
            estimated_roi=est_roi,
            strategy_markdown=summary_markdown
        )
