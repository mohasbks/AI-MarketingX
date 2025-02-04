import sys
import os
from pathlib import Path
import logging
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, HTTPException, Security, Depends, Header
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

# إعدادات TensorFlow للعمل على CPU فقط
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # تقليل رسائل TensorFlow

import tensorflow as tf
# تعطيل GPU بشكل كامل
physical_devices = tf.config.list_physical_devices('GPU')
try:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except:
    pass

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

# إعداد FastAPI
app = FastAPI(
    title="AI-MarketingX API",
    description="API للتحليل الذكي لحملات التسويق",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# إعداد القوالب
templates = Jinja2Templates(directory=os.path.join(project_root, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(project_root, "static")), name="static")

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
    campaign_name: str
    campaign_description: str
    campaign_objectives: List[str]
    target_audience_description: str
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
                "campaign_name": "Test Campaign",
                "campaign_description": "This is a test campaign",
                "campaign_objectives": ["increase brand awareness", "drive sales"],
                "target_audience_description": "People aged 25-34 interested in technology and marketing",
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

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """الصفحة الرئيسية"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api")
async def api_info():
    """معلومات عن API"""
    return {
        "name": "AI Marketing X API",
        "version": "1.0.0",
        "description": "واجهة برمجة التطبيقات للتسويق الذكي",
        "endpoints": {
            "/": "الصفحة الرئيسية",
            "/api": "معلومات API",
            "/docs": "توثيق Swagger",
            "/redoc": "توثيق ReDoc",
            "/analyze": "تحليل النص التسويقي",
            "/optimize": "تحسين النص التسويقي",
            "/generate": "توليد نص تسويقي"
        }
    }

@app.post("/api/v1/optimize", response_model=OptimizationResponse)
async def optimize_campaign(
    campaign: CampaignData,
    api_key: APIKey = Depends(get_api_key)
):
    try:
        # إنشاء بيانات إضافية مشتقة
        derived_metrics = {
            'ctr': campaign.clicks / max(campaign.impressions, 1),
            'cost_per_click': campaign.daily_spend / max(campaign.clicks, 1),
            'cost_per_conversion': campaign.daily_spend / max(campaign.clicks * campaign.conversion_rate, 1)
        }
        
        # تحليل الأهداف
        objective_weights = {
            'زيادة الوعي بالعلامة التجارية': 0.2,
            'جمع العملاء المحتملين': 0.3,
            'زيادة المبيعات': 0.4,
            'زيادة التفاعل': 0.2,
            'زيادة زيارات الموقع': 0.1
        }
        
        campaign_priority = sum(objective_weights[obj] for obj in campaign.campaign_objectives)
        
        # تحليل وصف الجمهور
        audience_keywords = set(campaign.target_audience_description.lower().split())
        important_keywords = {
            'شباب', 'طلاب', 'مهنيون', 'رجال أعمال', 'تقنية', 'تسوق',
            'رياضة', 'تعليم', 'صحة', 'سفر', 'ترفيه'
        }
        audience_matches = audience_keywords.intersection(important_keywords)
        
        # تجميع البيانات للتحليل
        analysis_data = {
            **campaign.dict(),
            **derived_metrics,
            'campaign_priority': campaign_priority,
            'audience_relevance': len(audience_matches) / len(important_keywords),
            'market_segment': _get_market_segment(campaign.daily_spend, campaign.conversion_rate)
        }
        
        # الحصول على التوصيات من المحسن
        recommendations = optimizer.get_recommendations(analysis_data)
        
        # تحديث النموذج مع النتائج الفعلية (التعلم المستمر)
        if hasattr(campaign, 'actual_results'):
            optimizer.update_models(analysis_data, campaign.actual_results)
        
        return OptimizationResponse(
            campaign_id=campaign.campaign_id,
            predicted_roi=recommendations['predicted_metrics']['roi'],
            recommendations=recommendations['recommendations'],
            timestamp=datetime.now(),
            best_times=recommendations.get('best_times'),
            confidence_score=recommendations['model_info']['confidence_score']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _get_market_segment(daily_spend: float, conversion_rate: float) -> str:
    """تحديد شريحة السوق بناءً على الإنفاق ومعدل التحويل"""
    if daily_spend > 1000:
        return 'enterprise'
    elif daily_spend > 100:
        return 'mid_market'
    else:
        return 'small_business'

@app.get("/api/v1/health")
async def health_check(api_key: APIKey = Depends(get_api_key)):
    """فحص حالة الخادم"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    error_msg = f"حدث خطأ غير متوقع: {str(exc)}"
    logging.error(error_msg)
    return JSONResponse(
        status_code=500,
        content={"detail": error_msg}
    )

from src.api.marketing_formulas_api import router as formulas_router

# إضافة مسارات الصيغ التسويقية
app.include_router(
    formulas_router,
    prefix="/api/v1/formulas",
    tags=["Marketing Formulas"],
    dependencies=[Depends(get_api_key)]
)

@app.on_event("startup")
async def startup_event():
    try:
        # إعداد logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # تهيئة النماذج
        optimizer._initialize_models()
        logging.info("تم تهيئة النماذج بنجاح")
    except Exception as e:
        logging.error(f"خطأ في تهيئة النماذج: {str(e)}")
        # استمرار التشغيل مع النماذج البسيطة
        pass

if __name__ == "__main__":
    # تشغيل التطبيق
    try:
        port = int(os.getenv("PORT", 8000))
        host = os.getenv("HOST", "0.0.0.0")
        
        # إعداد uvicorn بشكل صحيح
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
            workers=1
        )
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        logging.error(f"خطأ في تشغيل التطبيق: {str(e)}")
