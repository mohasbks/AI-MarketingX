import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import joblib
import json
import logging

from .deep_learning.transformer_model import AdvancedMarketingTransformer
from .deep_learning.performance_predictor import PerformancePredictor
from ..data.data_processor import DataProcessor

class CampaignOptimizer:
    def __init__(self):
        # تهيئة نماذج التعلم العميق
        self.transformer_model = AdvancedMarketingTransformer(input_dim=10)
        self.performance_predictor = PerformancePredictor(input_dim=10)
        self.data_processor = DataProcessor()
        
        # حفظ تاريخ التدريب
        self.training_history = []
        self.model_version = 0
        self.last_training_time = None
        
    def _initialize_models(self):
        """تهيئة النماذج بالبيانات الأولية"""
        try:
            # محاولة تحميل النماذج المحفوظة
            self.transformer_model.model.load_weights('models/transformer_weights.h5')
            self.performance_predictor.model.load_weights('models/predictor_weights.h5')
            logging.info("تم تحميل النماذج المحفوظة بنجاح")
        except:
            logging.info("لم يتم العثور على نماذج محفوظة. سيتم تدريب نماذج جديدة")
            
    def update_models(self, campaign_data, actual_results):
        """تحديث النماذج مع نتائج الحملة الفعلية"""
        try:
            # معالجة البيانات
            processed_data = self.data_processor.process_campaign_data(campaign_data)
            
            # تحديث النماذج
            self.transformer_model.train(
                processed_data,
                actual_results,
                epochs=5,
                batch_size=32
            )
            
            self.performance_predictor.train(
                processed_data,
                actual_results,
                epochs=5,
                batch_size=32
            )
            
            # حفظ النماذج
            self.transformer_model.model.save_weights('models/transformer_weights.h5')
            self.performance_predictor.model.save_weights('models/predictor_weights.h5')
            
            # تحديث معلومات التدريب
            self.model_version += 1
            self.last_training_time = datetime.now()
            self.training_history.append({
                'version': self.model_version,
                'timestamp': self.last_training_time,
                'performance_metrics': self._calculate_performance_metrics(actual_results)
            })
            
            logging.info(f"تم تحديث النماذج بنجاح. إصدار النموذج: {self.model_version}")
            
        except Exception as e:
            logging.error(f"خطأ في تحديث النماذج: {str(e)}")
            raise
            
    def _calculate_performance_metrics(self, actual_results):
        """حساب مقاييس أداء النموذج"""
        return {
            'mae': np.mean(np.abs(actual_results['predicted'] - actual_results['actual'])),
            'rmse': np.sqrt(np.mean((actual_results['predicted'] - actual_results['actual'])**2))
        }
        
    def get_recommendations(self, campaign_data):
        """توليد توصيات شاملة للحملة مع التعلم المستمر"""
        try:
            # معالجة البيانات
            processed_data = self.data_processor.process_campaign_data(campaign_data)
            
            # الحصول على التنبؤات من النماذج
            transformer_predictions = self.transformer_model.predict(processed_data)
            performance_predictions = self.performance_predictor.predict(processed_data)
            
            recommendations = []
            
            # توصيات الميزانية
            if campaign_data['daily_spend'] > 0:
                predicted_roi = performance_predictions['roi'][0]
                if predicted_roi < campaign_data.get('target_roi', 2.0):
                    recommendations.append({
                        'type': 'budget',
                        'message': 'اقتراح تعديل الميزانية اليومية لتحسين العائد على الاستثمار',
                        'action': 'optimize_budget',
                        'suggested_value': self._optimize_budget(campaign_data, predicted_roi)
                    })
            
            # توصيات الجمهور المستهدف
            audience_predictions = transformer_predictions['audience_scores']
            top_audiences = np.argsort(audience_predictions)[-2:][::-1]
            recommendations.append({
                'type': 'targeting',
                'message': 'أفضل الفئات المستهدفة بناءً على التحليل المتقدم',
                'suggestions': [
                    {
                        'audience': self._get_audience_name(i),
                        'confidence': float(audience_predictions[i])
                    } for i in top_audiences
                ]
            })
            
            # إضافة معلومات النموذج
            model_info = {
                'model_version': self.model_version,
                'last_training': self.last_training_time.isoformat() if self.last_training_time else None,
                'confidence_score': float(np.mean([pred['confidence'] for pred in recommendations if 'confidence' in pred]))
            }
            
            return {
                'recommendations': recommendations,
                'model_info': model_info,
                'predicted_metrics': {
                    'roi': float(performance_predictions['roi'][0]),
                    'ctr': float(performance_predictions['ctr'][0]),
                    'conversion_rate': float(performance_predictions['conversion_rate'][0])
                }
            }
            
        except Exception as e:
            logging.error(f"خطأ في توليد التوصيات: {str(e)}")
            raise
            
    def _optimize_budget(self, campaign_data, predicted_roi):
        """تحسين الميزانية بناءً على ROI المتوقع"""
        current_budget = campaign_data['daily_spend']
        target_roi = campaign_data.get('target_roi', 2.0)
        
        if predicted_roi < target_roi:
            return current_budget * 0.9  # تخفيض الميزانية بنسبة 10%
        else:
            return current_budget * 1.1  # زيادة الميزانية بنسبة 10%
            
    def _get_audience_name(self, index):
        """تحويل مؤشر الجمهور إلى اسم"""
        audiences = ['الشباب', 'المهنيون', 'أصحاب الأعمال', 'الطلاب']
        return audiences[index % len(audiences)]
