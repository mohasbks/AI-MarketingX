# دليل استخدام AI-MarketingX API 🚀

## مقدمة
يوفر AI-MarketingX واجهة برمجة تطبيقات (API) قوية تتيح لك دمج قدرات الذكاء الاصطناعي في موقعك بسهولة.

الرابط الأساسي: `https://web-production-412a.up.railway.app/`

## المصادقة
لاستخدام API، تحتاج إلى مفتاح API. أضف المفتاح في رأس الطلب:
```
X-API-Key: your-api-key-here
```

## نقاط النهاية المتاحة

### 1. تحليل وتحسين النص التسويقي
```javascript
POST /api/v1/analyze
Content-Type: application/json
X-API-Key: your-api-key

{
    "text": "النص التسويقي هنا",
    "target_audience": "الجمهور المستهدف",
    "platform": "facebook"  // facebook, twitter, instagram, linkedin
}
```

مثال استخدام JavaScript:
```javascript
async function analyzeMarketingText() {
    const response = await fetch('https://web-production-412a.up.railway.app/api/v1/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'your-api-key'
        },
        body: JSON.stringify({
            text: "نص الإعلان هنا",
            target_audience: "شباب مهتمين بالتكنولوجيا",
            platform: "facebook"
        })
    });
    const data = await response.json();
    console.log(data);
}
```

### 2. تحسين الحملة التسويقية
```javascript
POST /api/v1/optimize
Content-Type: application/json
X-API-Key: your-api-key

{
    "campaign_id": "123",
    "campaign_name": "حملة تسويقية",
    "daily_spend": 100,
    "clicks": 500,
    "impressions": 10000,
    "conversion_rate": 0.02,
    "target_audience_description": "وصف الجمهور"
}
```

مثال استخدام Python:
```python
import requests

def optimize_campaign():
    url = "https://web-production-412a.up.railway.app/api/v1/optimize"
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "your-api-key"
    }
    data = {
        "campaign_id": "123",
        "campaign_name": "حملة تسويقية",
        "daily_spend": 100,
        "clicks": 500,
        "impressions": 10000,
        "conversion_rate": 0.02,
        "target_audience_description": "وصف الجمهور"
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()
```

### 3. الصيغ التسويقية الذكية

#### أ. التنبؤ باحتمالية التحويل
```javascript
POST /api/v1/formulas/predict_conversion
Content-Type: application/json
X-API-Key: your-api-key

{
    "features": [[100, 1000, 50, 0.05]],
    "feature_names": ["daily_spend", "impressions", "clicks", "ctr"]
}
```

#### ب. تحسين العروض
```javascript
POST /api/v1/formulas/optimize_offer
Content-Type: application/json
X-API-Key: your-api-key

{
    "offers": [
        {
            "price": 100,
            "discount": 0.2,
            "duration": 30
        }
    ],
    "user_profile": {
        "age": 25,
        "interests": ["technology"],
        "purchase_history": 5
    },
    "weights": {
        "price": 0.4,
        "discount": 0.3,
        "duration": 0.3
    }
}
```

## دمج API في موقعك

### 1. إضافة زر تحليل في نموذج HTML
```html
<form id="marketingForm">
    <textarea id="marketingText" placeholder="أدخل النص التسويقي هنا"></textarea>
    <input type="text" id="targetAudience" placeholder="الجمهور المستهدف">
    <select id="platform">
        <option value="facebook">Facebook</option>
        <option value="twitter">Twitter</option>
        <option value="instagram">Instagram</option>
    </select>
    <button type="submit">تحليل وتحسين</button>
</form>

<div id="results"></div>
```

### 2. إضافة كود JavaScript للتفاعل
```javascript
document.getElementById('marketingForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const text = document.getElementById('marketingText').value;
    const targetAudience = document.getElementById('targetAudience').value;
    const platform = document.getElementById('platform').value;
    
    try {
        const response = await fetch('https://web-production-412a.up.railway.app/api/v1/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': 'your-api-key'
            },
            body: JSON.stringify({
                text,
                target_audience: targetAudience,
                platform
            })
        });
        
        const data = await response.json();
        
        // عرض النتائج
        document.getElementById('results').innerHTML = `
            <h3>نتائج التحليل:</h3>
            <p>درجة الفعالية: ${data.effectiveness_score}</p>
            <p>التوصيات: ${data.recommendations.join('<br>')}</p>
            <p>النص المحسن: ${data.optimized_text}</p>
        `;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('results').innerHTML = 'حدث خطأ أثناء التحليل';
    }
});
```

## أمثلة متقدمة

### 1. تحليل مستمر للحملات
```javascript
// دالة لتحليل أداء الحملة كل ساعة
async function analyzeCampaignPerformance(campaignId) {
    setInterval(async () => {
        const response = await fetch('https://web-production-412a.up.railway.app/api/v1/analyze_campaign', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': 'your-api-key'
            },
            body: JSON.stringify({ campaign_id: campaignId })
        });
        
        const data = await response.json();
        updateDashboard(data);
    }, 3600000); // كل ساعة
}

function updateDashboard(data) {
    // تحديث لوحة المعلومات بالبيانات الجديدة
}
```

### 2. تحسين تلقائي للميزانية
```javascript
async function autoOptimizeBudget(campaignId) {
    const response = await fetch('https://web-production-412a.up.railway.app/api/v1/optimize_budget', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'your-api-key'
        },
        body: JSON.stringify({
            campaign_id: campaignId,
            auto_adjust: true
        })
    });
    
    return await response.json();
}
```

## نصائح وأفضل الممارسات

1. **التخزين المؤقت**: قم بتخزين النتائج مؤقتاً لتحسين الأداء
```javascript
const cache = new Map();

async function analyzeWithCache(text) {
    if (cache.has(text)) {
        return cache.get(text);
    }
    
    const result = await analyzeText(text);
    cache.set(text, result);
    return result;
}
```

2. **معالجة الأخطاء**: تأكد من معالجة الأخطاء بشكل صحيح
```javascript
async function safeApiCall(endpoint, data) {
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': 'your-api-key'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        // معالجة الخطأ بشكل مناسب
        return null;
    }
}
```

3. **التحقق من صحة البيانات**: تحقق من البيانات قبل إرسالها
```javascript
function validateMarketingData(data) {
    if (!data.text || data.text.length < 10) {
        throw new Error('النص قصير جداً');
    }
    if (!data.target_audience) {
        throw new Error('الجمهور المستهدف مطلوب');
    }
    // المزيد من التحققات
}
```

## الحصول على مفتاح API

للحصول على مفتاح API، يرجى التواصل معنا عبر:
- البريد الإلكتروني: support@ai-marketingx.com
- نموذج الاتصال: https://web-production-412a.up.railway.app/contact

## الدعم والمساعدة

إذا واجهت أي مشاكل أو لديك أسئلة، يمكنك:
1. مراجعة التوثيق الكامل على: https://web-production-412a.up.railway.app/docs
2. فتح issue على GitHub: https://github.com/mohasbks/AI-MarketingX/issues
3. التواصل مع فريق الدعم الفني
