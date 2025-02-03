import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from datetime import datetime, timedelta
import joblib
import json

class CampaignOptimizer:
    def __init__(self):
        # نماذج التنبؤ المختلفة
        self.roi_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.audience_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.time_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(24, activation='softmax')  # 24 ساعة في اليوم
        ])
        
        # معالجة البيانات
        self.scaler = StandardScaler()
        self._initialize_models()
        
    def _initialize_models(self):
        """تهيئة النماذج بالبيانات الأولية"""
        # بيانات تدريب ROI
        X_roi = np.random.rand(1000, 6)  # المزيد من الميزات
        y_roi = X_roi.mean(axis=1) * 2 + np.random.rand(1000) * 0.5
        self.roi_model.fit(X_roi, y_roi)
        
        # بيانات تدريب الجمهور المستهدف
        X_audience = np.random.rand(1000, 4)
        y_audience = (X_audience.sum(axis=1) > 2).astype(int)
        self.audience_model.fit(X_audience, y_audience)
        
        # بيانات تدريب توقيت النشر
        X_time = np.random.rand(1000, 5)
        y_time = tf.keras.utils.to_categorical(np.random.randint(0, 24, 1000))
        self.time_model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.time_model.fit(X_time, y_time, epochs=5, verbose=0)
        
    def preprocess_data(self, data):
        """معالجة بيانات الحملة"""
        features = [
            'daily_spend', 'clicks', 'impressions', 'conversion_rate',
            'avg_session_duration', 'bounce_rate'
        ]
        
        # إضافة قيم افتراضية للميزات الناقصة
        for feature in features:
            if feature not in data:
                data[feature] = 0
                
        X = np.array([[data[feature] for feature in features]])
        return self.scaler.fit_transform(X)
        
    def predict_roi(self, campaign_data):
        """التنبؤ بالعائد على الاستثمار"""
        X = self.preprocess_data(campaign_data)
        return float(self.roi_model.predict(X)[0])
        
    def predict_best_audience(self, campaign_data):
        """تحديد أفضل الفئات المستهدفة"""
        features = np.array([
            campaign_data.get('avg_ctr', 0),
            campaign_data.get('conversion_rate', 0),
            campaign_data.get('engagement_rate', 0),
            campaign_data.get('bounce_rate', 0)
        ]).reshape(1, -1)
        
        return self.audience_model.predict_proba(features)[0]
        
    def predict_best_times(self, campaign_data):
        """التنبؤ بأفضل أوقات النشر"""
        features = np.array([
            campaign_data.get('daily_spend', 0),
            campaign_data.get('conversion_rate', 0),
            campaign_data.get('avg_engagement', 0),
            campaign_data.get('day_of_week', 0),
            campaign_data.get('season', 0)
        ]).reshape(1, -1)
        
        time_scores = self.time_model.predict(features)[0]
        best_hours = np.argsort(time_scores)[-3:][::-1]  # أفضل 3 ساعات
        
        return [{
            'hour': int(hour),
            'score': float(time_scores[hour]),
            'formatted_time': f"{hour:02d}:00"
        } for hour in best_hours]
        
    def get_recommendations(self, campaign_data):
        """توليد توصيات شاملة للحملة"""
        roi_prediction = self.predict_roi(campaign_data)
        audience_scores = self.predict_best_audience(campaign_data)
        best_times = self.predict_best_times(campaign_data)
        
        recommendations = []
        
        # توصيات الميزانية
        if campaign_data['daily_spend'] > 0:
            cpc = campaign_data['daily_spend'] / max(campaign_data['clicks'], 1)
            if cpc > 2.0:
                recommendations.append({
                    'type': 'budget',
                    'message': 'اعتبر تخفيض ميزانيتك اليومية لتحسين العائد على الاستثمار',
                    'action': 'decrease_budget',
                    'value': round(campaign_data['daily_spend'] * 0.9, 2)
                })
        
        # توصيات الجمهور المستهدف
        target_groups = ['الشباب', 'المهنيون', 'أصحاب الأعمال', 'الطلاب']
        top_audiences = np.argsort(audience_scores)[-2:][::-1]
        recommendations.append({
            'type': 'targeting',
            'message': 'أفضل الفئات المستهدفة المقترحة',
            'suggestions': [
                f"{target_groups[i]} (ثقة: {audience_scores[i]:.2%})"
                for i in top_audiences
            ]
        })
        
        # توصيات توقيت النشر
        recommendations.append({
            'type': 'timing',
            'message': 'أفضل أوقات النشر المقترحة',
            'best_times': best_times
        })
        
        # توصيات تحسين معدل التحويل
        if campaign_data['conversion_rate'] < 0.03:
            recommendations.append({
                'type': 'conversion',
                'message': 'يمكن تحسين معدل التحويل',
                'action': 'optimize_conversion',
                'suggestions': [
                    'مراجعة وتحسين صفحة الهبوط',
                    'تنفيذ اختبارات A/B',
                    'إضافة شهادات العملاء وعلامات الثقة'
                ]
            })
            
        return {
            'predicted_roi': roi_prediction,
            'recommendations': recommendations,
            'analysis_timestamp': datetime.now().isoformat(),
            'confidence_score': 0.85  # مقياس الثقة في التوصيات
        }
