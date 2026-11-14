import os
import re
from typing import List, Optional, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/assistant", tags=["Conversational AI Insurance Assistant"])

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []
    user_context: Optional[Dict] = None

class ChatResponse(BaseModel):
    reply: str
    suggested_questions: List[str]
    context_tags: List[str]

# Domain Knowledge Base for Guaranteed High-Quality Offline Responses
KNOWLEDGE_BASE = {
    "deductible": (
        "A **deductible** is the amount of healthcare expenses you must pay out-of-pocket each calendar year "
        "before your health insurance carrier begins to cover expenses. For example, if your plan has a $1,500 deductible, "
        "you pay 100% of non-preventive eligible medical costs up to $1,500. Once met, you only pay a copay or coinsurance."
    ),
    "copayment": (
        "A **copayment (copay)** is a predetermined flat fee you pay for specific healthcare services at the time of care. "
        "For instance, a standard primary care consultation might require a $25 copay, or a specialist consultation $50. "
        "Copays count toward your total annual out-of-pocket maximum."
    ),
    "coinsurance": (
        "**Coinsurance** is your percentage share of covered medical expenses after your annual deductible has been fully satisfied. "
        "If your plan has an 80/20 coinsurance structure and an eligible hospital stay bills $5,000 post-deductible, your insurer pays $4,000 (80%) "
        "and you pay $1,000 (20%), capped by your out-of-pocket maximum."
    ),
    "out_of_pocket": (
        "The **out-of-pocket maximum (OOPM)** is the absolute financial ceiling you will pay for covered in-network medical services "
        "during a single policy year. It includes your deductible, copayments, and coinsurance. Once you reach this limit, your insurer pays "
        "100% of all eligible covered healthcare services for the remainder of that policy year."
    ),
    "waiting_period": (
        "A **waiting period** is the duration of time that must elapse after your policy starts before coverage applies to specific medical conditions "
        "— typically pre-existing chronic conditions or maternity services. In MedInsight AI's plan comparison, plans range from 0 days (Elite Sovereign Gold) "
        "to 12 months for standard HMO policies."
    ),
    "smoker_impact": (
        "In our machine learning benchmark analysis, **smoking status** is the single highest coefficient actuarial driver. "
        "Actuarially, tobacco usage increases baseline medical charges by 150% to 280%. Furthermore, our Tree SHAP waterfall demonstrates a strong **non-linear interaction**: "
        "when smoking is combined with clinical obesity (BMI ≥ 30), estimated annual medical charges surge by over $30,000 due to compounded cardiovascular and pulmonary risk."
    ),
    "bmi_obesity": (
        "**Body Mass Index (BMI)** measures weight relative to height. Clinical benchmarks: Normal (18.5–24.9), Overweight (25–29.9), Obese (≥ 30). "
        "In our predictive regression models, BMI shows a gentle linear slope for non-smokers (~$300-$500 per unit), but triggers severe exponential compounding "
        "in active tobacco users."
    ),
    "conformal_prediction": (
        "MedInsight AI utilizes **Conformal Prediction** rather than heuristic or fabricated confidence scores. Conformal prediction is a mathematically rigorous, "
        "distribution-free uncertainty framework that guarantees a 90% statistical coverage guarantee based on calibration residuals. This means 90% of actual future "
        "charges will fall within the provided lower and upper bounds."
    ),
    "shap_explanation": (
        "**SHAP (SHapley Additive exPlanations)** is based on cooperative game theory. It decomposes the model's prediction into exact, additive feature contributions. "
        "The baseline represents the average population cost ($13,270). Every feature (e.g. +$23,500 for tobacco use, +$4,200 for age 52, -$1,800 for normal BMI) "
        "pushes the estimate up or down to arrive at the final individual premium prediction."
    )
}

def generate_assistant_response(user_query: str, history: List[ChatMessage], user_context: Optional[Dict] = None) -> ChatResponse:
    query_lower = user_query.lower()

    # Context enrichment if prediction provided
    ctx_str = ""
    if user_context and "predicted_charges" in user_context:
        ctx_str = f" (Referencing your current prediction: ${user_context.get('predicted_charges'):,.2f}, Risk Tier: {user_context.get('risk_tier')})"

    # 1. Matching Knowledge Base
    if any(k in query_lower for k in ["deductible", "deduct"]):
        reply = (
            f"Here is how deductibles work{ctx_str}:\n\n"
            f"{KNOWLEDGE_BASE['deductible']}\n\n"
            "**Actuarial Tip:** If you anticipate frequent specialist consultations or have recurring prescriptions, choosing a plan with a lower deductible (like PPO Plus or Gold) often results in lower total annual expenditure despite a higher monthly premium."
        )
        tags = ["Deductible", "Health Economics", "Out-of-pocket"]
        suggested = [
            "What is the difference between copay and coinsurance?",
            "How does my predicted cost compare to the average deductible?",
            "Which plan best covers pre-existing conditions?"
        ]
    elif any(k in query_lower for k in ["copay", "co-pay", "copayment"]):
        reply = (
            f"**Understanding Copayments (Copays):**\n\n"
            f"{KNOWLEDGE_BASE['copayment']}\n\n"
            "Unlike coinsurance where you bear a percentage risk, copayments provide predictable fixed costs for routine clinic visits and standard pharmacy tiers."
        )
        tags = ["Copay", "Cost Sharing", "Primary Care"]
        suggested = [
            "What is coinsurance?",
            "What is an Out-of-Pocket Maximum?",
            "How does smoking affect my insurance cost?"
        ]
    elif any(k in query_lower for k in ["coinsurance", "co-insurance"]):
        reply = (
            f"**Understanding Coinsurance:**\n\n"
            f"{KNOWLEDGE_BASE['coinsurance']}\n\n"
            "Coinsurance only activates after you pay your full deductible, protecting you from paying total charges out-of-pocket."
        )
        tags = ["Coinsurance", "Cost Sharing", "Hospitalization"]
        suggested = [
            "What is the Out-of-Pocket Maximum?",
            "How do I choose between HMO and PPO?",
            "What is conformal prediction?"
        ]
    elif any(k in query_lower for k in ["pocket", "maximum", "oopm", "oop"]):
        reply = (
            f"**Out-of-Pocket Maximum (OOPM):**\n\n"
            f"{KNOWLEDGE_BASE['out_of_pocket']}\n\n"
            "This is your ultimate safety net against catastrophic health crises."
        )
        tags = ["OOPM", "Financial Protection", "Catastrophic"]
        suggested = [
            "What is a High Deductible Health Plan (HDHP)?",
            "Can you explain the SHAP waterfall chart?",
            "How can I lower my insurance premium?"
        ]
    elif any(k in query_lower for k in ["smok", "tobacco", "cigarette", "vape"]):
        reply = (
            f"**Actuarial Impact of Smoking & Tobacco:**\n\n"
            f"{KNOWLEDGE_BASE['smoker_impact']}\n\n"
            "**Clinical Guidance:** Enrolling in an evidence-based smoking cessation protocol for 12 months allows policyholders in many jurisdictions to qualify for standard non-smoker underwriting rates, potentially saving $15,000–$25,000 annually."
        )
        tags = ["Tobacco Surcharge", "Actuarial Risk", "SHAP Waterfall"]
        suggested = [
            "What happens if I combine smoking with high BMI?",
            "How much does age increase medical insurance prices?",
            "How does the What-If Simulator work?"
        ]
    elif any(k in query_lower for k in ["bmi", "weight", "obese", "obesity"]):
        reply = (
            f"**Body Mass Index (BMI) & Actuarial Charges:**\n\n"
            f"{KNOWLEDGE_BASE['bmi_obesity']}\n\n"
            "In our ML feature importance rankings, BMI is ranked #2 in overall predictive importance behind smoking."
        )
        tags = ["BMI", "Obesity Risk", "Feature Importance"]
        suggested = [
            "What is the interaction between obesity and smoking?",
            "How does my region affect insurance cost?",
            "How are prediction intervals calculated?"
        ]
    elif any(k in query_lower for k in ["shap", "waterfall", "explain", "attribution"]):
        reply = (
            f"**Explainable AI via SHAP (Shapley Values):**\n\n"
            f"{KNOWLEDGE_BASE['shap_explanation']}\n\n"
            "In MEDINSIGHT AI, red bars in your report signify risk factors that increased estimated expenses, whereas green bars reflect protective attributes (such as youth or non-smoking status) that decreased your cost relative to the baseline."
        )
        tags = ["SHAP", "Explainable AI", "Model Interpretability"]
        suggested = [
            "What is Conformal Prediction?",
            "Which ML algorithm achieved the best R² score?",
            "Can I download a certified PDF report?"
        ]
    elif any(k in query_lower for k in ["conformal", "interval", "confidence", "uncertainty"]):
        reply = (
            f"**Conformal Prediction & Statistical Intervals:**\n\n"
            f"{KNOWLEDGE_BASE['conformal_prediction']}\n\n"
            "Rather than giving a single deterministic point estimate (which can be misleading), MEDINSIGHT AI provides an actuarially sound uncertainty range."
        )
        tags = ["Conformal Prediction", "Uncertainty Estimation", "Actuarial Science"]
        suggested = [
            "How can I compare insurance plans side by side?",
            "What is the difference between HMO, PPO, and HDHP?",
            "Explain deductibles and copayments."
        ]
    elif any(k in query_lower for k in ["hmo", "ppo", "hdhp", "compare", "plan"]):
        reply = (
            "**Comparing Health Insurance Plan Categories:**\n\n"
            "1. **HMO (Health Maintenance Organization):** Lower premiums, strict in-network coverage, primary care physician referral required for specialists.\n"
            "2. **PPO (Preferred Provider Organization):** Greater flexibility, direct access to specialists without referrals, partial reimbursement for out-of-network care.\n"
            "3. **HDHP with HSA:** Lowest monthly premiums, high deductible, eligible for tax-free Health Savings Accounts.\n"
            "4. **Gold Comprehensive Tier:** Higher monthly premium, zero waiting periods for pre-existing conditions, low copayments.\n\n"
            "You can use the **Insurance Comparison** tab to inspect all 4 plan tiers side-by-side with exact copays, deductibles, and benefits."
        )
        tags = ["Plan Comparison", "HMO vs PPO", "Benefit Design"]
        suggested = [
            "Which plan is recommended for my risk profile?",
            "What is a waiting period for pre-existing conditions?",
            "How does the What-if Simulator work?"
        ]
    else:
        # Default intelligent response
        reply = (
            f"Hello! I am your **MedInsight AI Actuarial & Insurance Intelligence Assistant**{ctx_str}.\n\n"
            "I can explain any aspect of health insurance economics, model predictions, and risk analytics:\n"
            "- **Insurance Terminology:** Deductibles, copays, coinsurance, waiting periods, out-of-pocket maximums.\n"
            "- **Explainable AI:** How SHAP values break down your predicted charges.\n"
            "- **Risk Factor Analysis:** How age, BMI, tobacco status, and region drive actuarial costs.\n"
            "- **Plan Comparison:** HMO, PPO, HDHP, and Gold Critical Care plans.\n"
            "- **What-if Simulation:** How lifestyle modifications can lower projected insurance expenditures.\n\n"
            "What specific question would you like to explore?"
        )
        tags = ["MedInsight AI", "Assistant", "Insurance Guide"]
        suggested = [
            "Explain how deductibles and copays work.",
            "Why does smoking dramatically increase insurance charges?",
            "How does SHAP explain my prediction?",
            "What is the difference between HMO and PPO plans?"
        ]

    return ChatResponse(reply=reply, suggested_questions=suggested, context_tags=tags)

@router.post("/chat", response_model=ChatResponse)
def chat_with_assistant(req: ChatRequest):
    """Conversational endpoint for explaining insurance policies, SHAP risk drivers, and terms."""
    if not req.message or len(req.message.strip()) == 0:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    return generate_assistant_response(req.message.strip(), req.history or [], req.user_context)
