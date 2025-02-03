import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from fastapi import FastAPI, HTTPException, Security, Depends, Header
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from datetime import datetime
from starlette.status import HTTP_403_FORBIDDEN

from src.models.campaign_optimizer import CampaignOptimizer

# API Key ثابت للجميع
API_KEY = "ai-marketing-x-2025-key"
API_KEY_NAME = "X-API-Key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="API Key header is missing"
        )
    if api_key_header != API_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid API Key"
        )
    return api_key_header

app = FastAPI(
    title="AI-MarketingX API",
    description="API للتحليل الذكي لحملات التسويق",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# إعداد CORS للسماح بالوصول من أي موقع
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

optimizer = CampaignOptimizer()

# تعريف نماذج البيانات
class CampaignData(BaseModel):
    campaign_id: str
    daily_spend: float
    clicks: int
    impressions: int
    conversion_rate: float
    demographics: dict
    location: str
    keywords: List[str]
    avg_session_duration: Optional[float] = 0
    bounce_rate: Optional[float] = 0
    avg_ctr: Optional[float] = 0
    engagement_rate: Optional[float] = 0
    avg_engagement: Optional[float] = 0
    day_of_week: Optional[int] = datetime.now().weekday()
    season: Optional[int] = ((datetime.now().month % 12) // 3) + 1

    class Config:
        schema_extra = {
            "example": {
                "campaign_id": "test_campaign_1",
                "daily_spend": 100,
                "clicks": 500,
                "impressions": 10000,
                "conversion_rate": 0.02,
                "demographics": {
                    "age": "25-34",
                    "gender": "all",
                    "interests": ["technology", "marketing"]
                },
                "location": "US",
                "keywords": ["digital marketing", "ai", "automation"]
            }
        }

class OptimizationResponse(BaseModel):
    campaign_id: str
    predicted_roi: float
    recommendations: List[dict]
    timestamp: datetime
    best_times: Optional[List[dict]]
    confidence_score: Optional[float]

@app.post("/api/v1/optimize", response_model=OptimizationResponse, 
          description="تحليل وتحسين حملة تسويقية")
async def optimize_campaign(
    campaign: CampaignData,
    api_key: APIKey = Depends(get_api_key)
):
    try:
        results = optimizer.get_recommendations({
            'daily_spend': campaign.daily_spend,
            'clicks': campaign.clicks,
            'impressions': campaign.impressions,
            'conversion_rate': campaign.conversion_rate,
            'avg_session_duration': campaign.avg_session_duration,
            'bounce_rate': campaign.bounce_rate,
            'avg_ctr': campaign.avg_ctr,
            'engagement_rate': campaign.engagement_rate,
            'avg_engagement': campaign.avg_engagement,
            'day_of_week': campaign.day_of_week,
            'season': campaign.season
        })
        
        return OptimizationResponse(
            campaign_id=campaign.campaign_id,
            predicted_roi=float(results['predicted_roi']),
            recommendations=results['recommendations'],
            timestamp=datetime.now(),
            best_times=results.get('best_times', []),
            confidence_score=results.get('confidence_score', 0.0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check(api_key: APIKey = Depends(get_api_key)):
    """فحص حالة الخادم"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
