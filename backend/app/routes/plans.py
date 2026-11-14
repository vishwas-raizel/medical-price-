from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/plans", tags=["Insurance Plan Comparison & Recommendation"])

class InsurancePlan(BaseModel):
    id: str
    name: str
    provider: str
    category: str  # HMO, PPO, HDHP, Comprehensive Gold
    annual_premium: float
    monthly_premium: float
    deductible: float
    copay_primary: str
    copay_specialist: str
    coinsurance: str
    out_of_pocket_max: float
    waiting_period_preexisting: str
    network_size: str
    benefits: List[str]
    exclusions: List[str]
    claim_settlement_ratio: str
    suitability: str

PLANS_DATABASE = [
    InsurancePlan(
        id="hmo_standard",
        name="Essential Care HMO Plan",
        provider="Apex Health Network",
        category="HMO (Health Maintenance Organization)",
        annual_premium=4200.0,
        monthly_premium=350.0,
        deductible=2500.0,
        copay_primary="$20 per visit",
        copay_specialist="$45 with referral",
        coinsurance="10% after deductible",
        out_of_pocket_max=6500.0,
        waiting_period_preexisting="12 months",
        network_size="Regional in-network (Primary Care Gatekeeper)",
        benefits=[
            "100% Preventive care & annual wellness screening",
            "Generic prescription drug coverage ($10 copay)",
            "Telehealth consultations at $0 cost",
            "Emergency room coverage after $250 copay"
        ],
        exclusions=[
            "Out-of-network non-emergency medical services",
            "Elective cosmetic surgical procedures",
            "Experimental medical therapies",
            "Adult dental and routine vision care"
        ],
        claim_settlement_ratio="96.4%",
        suitability="Best for young individuals, budget-conscious families, and low-to-moderate risk profiles."
    ),
    InsurancePlan(
        id="ppo_comprehensive",
        name="Preferred Choice PPO Plus",
        provider="BlueCross Sentinel Assurance",
        category="PPO (Preferred Provider Organization)",
        annual_premium=7800.0,
        monthly_premium=650.0,
        deductible=1200.0,
        copay_primary="$30 (No referral needed)",
        copay_specialist="$50 (Direct specialist access)",
        coinsurance="20% in-network / 40% out-of-network",
        out_of_pocket_max=5000.0,
        waiting_period_preexisting="6 months",
        network_size="Nationwide open network (45,000+ facilities)",
        benefits=[
            "Direct specialist consultations without referral gatekeeping",
            "Out-of-network physician & hospital coverage",
            "Comprehensive brand and generic pharmacy formulary",
            "Maternity, pediatric, and mental health services",
            "Advanced imaging (MRI/CT) after deductible"
        ],
        exclusions=[
            "Cosmetic procedures and non-prescribed therapies",
            "Acupuncture & non-traditional treatments",
            "Over-the-counter medical supplements"
        ],
        claim_settlement_ratio="98.1%",
        suitability="Ideal for active families, individuals with recurring specialist visits, or moderate health risk profiles."
    ),
    InsurancePlan(
        id="hdhp_hsa",
        name="Optima HSA Advantage (HDHP)",
        provider="United Horizon Underwriters",
        category="HDHP (High-Deductible Health Plan)",
        annual_premium=3100.0,
        monthly_premium=258.33,
        deductible=4000.0,
        copay_primary="100% covered after deductible",
        copay_specialist="100% covered after deductible",
        coinsurance="0% after deductible is reached",
        out_of_pocket_max=4000.0,
        waiting_period_preexisting="12 months",
        network_size="National broad network",
        benefits=[
            "Triple-tax-advantaged Health Savings Account (HSA) eligible",
            "100% catastrophic inpatient care covered once deductible met",
            "Preventive immunizations and biometric screens at $0",
            "Lowest monthly premium footprint"
        ],
        exclusions=[
            "Non-preventive services prior to meeting deductible",
            "Elective wellness treatments",
            "Overseas elective treatments"
        ],
        claim_settlement_ratio="95.2%",
        suitability="Perfect for healthy, low-risk applicants seeking catastrophic financial shield with maximal tax savings."
    ),
    InsurancePlan(
        id="gold_critical_care",
        name="Elite Sovereign Critical Care Gold",
        provider="Global Medical Shield Reinsurance",
        category="Comprehensive Gold Tier",
        annual_premium=13500.0,
        monthly_premium=1125.0,
        deductible=500.0,
        copay_primary="$10 copay",
        copay_specialist="$25 copay",
        coinsurance="5% after deductible",
        out_of_pocket_max=2500.0,
        waiting_period_preexisting="0 days (Immediate Day-1 Coverage)",
        network_size="International & premier academic medical centers",
        benefits=[
            "Immediate Day-1 coverage for all pre-existing chronic conditions",
            "Zero waiting period for cardiovascular, metabolic & oncology care",
            "Dedicated clinical care concierge and second medical opinion service",
            "Full private suite hospitalization & unlimited ICU days",
            "Comprehensive outpatient rehabilitation and physical therapy"
        ],
        exclusions=[
            "Experimental unverified clinical trials",
            "Cosmetic surgery not related to trauma reconstructive surgery"
        ],
        claim_settlement_ratio="99.3%",
        suitability="Recommended for high-risk profiles, chronic condition management, smokers, or senior policyholders requiring comprehensive coverage."
    )
]

@router.get("", response_model=List[InsurancePlan])
def list_insurance_plans(category: Optional[str] = None):
    """Retrieve verified illustrative insurance plan comparison database."""
    if category:
        return [p for p in PLANS_DATABASE if category.lower() in p.category.lower()]
    return PLANS_DATABASE

class PlanRecommendationRequest(BaseModel):
    predicted_annual_cost: float
    risk_tier: str
    age: int
    smoker: str

@router.post("/recommend")
def recommend_optimal_plan(req: PlanRecommendationRequest):
    """Actuarial algorithm to recommend the best matching insurance plan."""
    if req.risk_tier in ["High Risk", "Critical Risk"] or (req.smoker == "yes" and req.predicted_annual_cost > 15000):
        recommended = PLANS_DATABASE[3]  # Gold Critical Care
        reason = (
            f"Given your projected annual charges (${req.predicted_annual_cost:,.2f}) and '{req.risk_tier}' status, "
            "the Elite Sovereign Gold plan minimizes high out-of-pocket volatility with zero pre-existing waiting periods."
        )
    elif req.predicted_annual_cost > 6000 or req.age >= 45:
        recommended = PLANS_DATABASE[1]  # PPO Plus
        reason = (
            f"With moderate predicted annual expenses (${req.predicted_annual_cost:,.2f}), the Preferred Choice PPO provides "
            "direct specialist access without referral delays and a balanced $1,200 deductible."
        )
    elif req.smoker == "no" and req.age < 35:
        recommended = PLANS_DATABASE[2]  # HDHP
        reason = (
            f"Your low predicted risk (${req.predicted_annual_cost:,.2f}) makes the Optima HDHP ideal, allowing tax-free HSA "
            "savings while preserving comprehensive protection for major medical events."
        )
    else:
        recommended = PLANS_DATABASE[0]  # HMO
        reason = (
            f"The Essential Care HMO offers predictable low-copay routine medical care with well-balanced premium costs."
        )

    return {
        "recommended_plan": recommended,
        "actuarial_rationale": reason,
        "all_plans": PLANS_DATABASE
    }
