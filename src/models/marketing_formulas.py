import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Union
import logging

class MarketingFormulas:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.sentiment_model = None
        self.conversion_model = LogisticRegression()
        
    def predict_conversion_probability(self, features: np.ndarray) -> float:
        """
        حساب احتمالية التحويل باستخدام الانحدار اللوجستي
        P(Conversion) = 1 / (1 + exp(-(B0 + B1*X1 + ... + Bn*Xn)))
        """
        try:
            return self.conversion_model.predict_proba(features)[:, 1]
        except Exception as e:
            logging.error(f"خطأ في حساب احتمالية التحويل: {str(e)}")
            return 0.0
            
    def optimize_offer(self, offers: List[Dict], user_profile: Dict,
                      weights: Dict[str, float]) -> Dict:
        """
        تحسين العرض بناءً على ملف المستخدم
        Offer_opt = argmax_O ( sum( w_i * f_i(O, UserProfile) ) )
        """
        try:
            offer_scores = []
            for offer in offers:
                score = sum(
                    weights[attr] * self._calculate_relevance(offer, user_profile, attr)
                    for attr in weights
                )
                offer_scores.append(score)
            
            best_offer_idx = np.argmax(offer_scores)
            return {
                'offer': offers[best_offer_idx],
                'score': offer_scores[best_offer_idx]
            }
        except Exception as e:
            logging.error(f"خطأ في تحسين العرض: {str(e)}")
            return offers[0] if offers else {}
            
    def optimize_commission(self, revenue: np.ndarray, cost: np.ndarray,
                          commission_rates: np.ndarray) -> Dict:
        """
        تحسين العمولة لتعظيم الربح
        MaxProfit = sum( (Revenue_i - Cost_i) * alpha_i )
        """
        try:
            profits = (revenue - cost) * commission_rates
            total_profit = np.sum(profits)
            roi = np.sum(profits) / np.sum(cost) if np.sum(cost) > 0 else 0
            
            return {
                'total_profit': total_profit,
                'roi': roi,
                'profit_per_item': profits.tolist(),
                'optimal_commission': commission_rates[np.argmax(profits)]
            }
        except Exception as e:
            logging.error(f"خطأ في تحسين العمولة: {str(e)}")
            return {}
            
    def analyze_sentiment(self, texts: List[str]) -> Dict:
        """
        تحليل المشاعر باستخدام TF-IDF
        SentimentScore = sum( theta_j * TF-IDF(word_j) )
        """
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            sentiment_scores = np.mean(tfidf_matrix.toarray(), axis=1)
            
            return {
                'scores': sentiment_scores.tolist(),
                'average_sentiment': float(np.mean(sentiment_scores)),
                'positive_ratio': float(np.mean(sentiment_scores > 0.5))
            }
        except Exception as e:
            logging.error(f"خطأ في تحليل المشاعر: {str(e)}")
            return {}
            
    def calculate_dynamic_price(self, base_price: float, demand: float,
                              supply: float, sensitivity: float = 0.1) -> Dict:
        """
        حساب السعر الديناميكي
        Price_t = P_0 * (1 + eta * (Demand_t - Supply_t) / Supply_t)
        """
        try:
            if supply <= 0:
                return {'price': base_price, 'adjustment_factor': 1.0}
                
            adjustment = 1 + sensitivity * (demand - supply) / supply
            new_price = base_price * adjustment
            
            return {
                'price': float(new_price),
                'adjustment_factor': float(adjustment),
                'price_change_percentage': float((new_price - base_price) / base_price * 100)
            }
        except Exception as e:
            logging.error(f"خطأ في حساب السعر الديناميكي: {str(e)}")
            return {'price': base_price, 'adjustment_factor': 1.0}
            
    def calculate_campaign_efficiency(self, conversion_values: np.ndarray,
                                   ad_costs: np.ndarray) -> Dict:
        """
        حساب كفاءة الحملة
        ROI = sum(ConversionValue - AdCost) / sum(AdCost)
        """
        try:
            total_cost = np.sum(ad_costs)
            total_value = np.sum(conversion_values)
            
            if total_cost == 0:
                return {'roi': 0, 'efficiency_ratio': 0}
                
            roi = (total_value - total_cost) / total_cost
            efficiency_ratio = total_value / total_cost
            
            return {
                'roi': float(roi),
                'efficiency_ratio': float(efficiency_ratio),
                'total_cost': float(total_cost),
                'total_value': float(total_value),
                'profit': float(total_value - total_cost)
            }
        except Exception as e:
            logging.error(f"خطأ في حساب كفاءة الحملة: {str(e)}")
            return {}
            
    def calculate_lead_score(self, user_data: Dict[str, float],
                           feature_weights: Dict[str, float]) -> Dict:
        """
        حساب درجة العميل المحتمل
        LeadScore = sum( gamma_k * Feature_k )
        """
        try:
            score = sum(
                feature_weights.get(feature, 0) * value
                for feature, value in user_data.items()
            )
            
            # تصنيف العملاء المحتملين
            category = self._categorize_lead(score)
            
            return {
                'score': float(score),
                'category': category,
                'normalized_score': float(min(max(score / 100, 0), 1)),
                'contributing_factors': self._get_contributing_factors(
                    user_data, feature_weights
                )
            }
        except Exception as e:
            logging.error(f"خطأ في حساب درجة العميل المحتمل: {str(e)}")
            return {}
    
    def _calculate_relevance(self, offer: Dict, user_profile: Dict,
                           attribute: str) -> float:
        """حساب مدى ملاءمة العرض لسمة معينة في ملف المستخدم"""
        try:
            if attribute not in offer or attribute not in user_profile:
                return 0.0
            
            offer_value = offer[attribute]
            user_value = user_profile[attribute]
            
            if isinstance(offer_value, (int, float)) and isinstance(user_value, (int, float)):
                return 1 - abs(offer_value - user_value) / max(offer_value, user_value)
            
            return 1.0 if offer_value == user_value else 0.0
            
        except Exception:
            return 0.0
    
    def _categorize_lead(self, score: float) -> str:
        """تصنيف العملاء المحتملين بناءً على درجاتهم"""
        if score >= 80:
            return "ممتاز"
        elif score >= 60:
            return "جيد جداً"
        elif score >= 40:
            return "جيد"
        elif score >= 20:
            return "متوسط"
        else:
            return "ضعيف"
    
    def _get_contributing_factors(self, user_data: Dict[str, float],
                                weights: Dict[str, float]) -> List[Dict]:
        """تحديد العوامل المؤثرة في درجة العميل المحتمل"""
        factors = []
        for feature, value in user_data.items():
            weight = weights.get(feature, 0)
            contribution = value * weight
            
            if contribution > 0:
                factors.append({
                    'feature': feature,
                    'contribution': float(contribution),
                    'importance': float(weight)
                })
        
        return sorted(factors, key=lambda x: x['contribution'], reverse=True)
