from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict
import numpy as np
from pydantic import BaseModel
from datetime import datetime

from src.models.marketing_formulas import MarketingFormulas

router = APIRouter()
formulas = MarketingFormulas()

class ConversionRequest(BaseModel):
    features: List[List[float]]
    feature_names: List[str]

class OfferRequest(BaseModel):
    offers: List[Dict]
    user_profile: Dict
    weights: Dict[str, float]

class CommissionRequest(BaseModel):
    revenue: List[float]
    cost: List[float]
    commission_rates: List[float]

class SentimentRequest(BaseModel):
    texts: List[str]

class PricingRequest(BaseModel):
    base_price: float
    demand: float
    supply: float
    sensitivity: float = 0.1

class CampaignRequest(BaseModel):
    conversion_values: List[float]
    ad_costs: List[float]

class LeadScoringRequest(BaseModel):
    user_data: Dict[str, float]
    feature_weights: Dict[str, float]

@router.post("/predict_conversion")
async def predict_conversion(request: ConversionRequest):
    """التنبؤ باحتمالية التحويل"""
    try:
        features = np.array(request.features)
        probabilities = formulas.predict_conversion_probability(features)
        
        return {
            "probabilities": probabilities.tolist(),
            "timestamp": datetime.now(),
            "feature_importance": dict(zip(
                request.feature_names,
                formulas.conversion_model.coef_[0].tolist()
            ))
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/optimize_offer")
async def optimize_offer(request: OfferRequest):
    """تحسين العرض للمستخدم"""
    try:
        result = formulas.optimize_offer(
            request.offers,
            request.user_profile,
            request.weights
        )
        return {**result, "timestamp": datetime.now()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/optimize_commission")
async def optimize_commission(request: CommissionRequest):
    """تحسين معدلات العمولة"""
    try:
        result = formulas.optimize_commission(
            np.array(request.revenue),
            np.array(request.cost),
            np.array(request.commission_rates)
        )
        return {**result, "timestamp": datetime.now()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/analyze_sentiment")
async def analyze_sentiment(request: SentimentRequest):
    """تحليل المشاعر في النصوص"""
    try:
        result = formulas.analyze_sentiment(request.texts)
        return {**result, "timestamp": datetime.now()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/calculate_dynamic_price")
async def calculate_dynamic_price(request: PricingRequest):
    """حساب السعر الديناميكي"""
    try:
        result = formulas.calculate_dynamic_price(
            request.base_price,
            request.demand,
            request.supply,
            request.sensitivity
        )
        return {**result, "timestamp": datetime.now()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/calculate_campaign_efficiency")
async def calculate_campaign_efficiency(request: CampaignRequest):
    """حساب كفاءة الحملة"""
    try:
        result = formulas.calculate_campaign_efficiency(
            np.array(request.conversion_values),
            np.array(request.ad_costs)
        )
        return {**result, "timestamp": datetime.now()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/calculate_lead_score")
async def calculate_lead_score(request: LeadScoringRequest):
    """حساب درجة العميل المحتمل"""
    try:
        result = formulas.calculate_lead_score(
            request.user_data,
            request.feature_weights
        )
        return {**result, "timestamp": datetime.now()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
