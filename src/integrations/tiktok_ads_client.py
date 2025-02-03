import requests
import json
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging

class TikTokAdsIntegration:
    def __init__(self, access_token: str, app_id: str, secret: str):
        self.access_token = access_token
        self.app_id = app_id
        self.secret = secret
        self.base_url = "https://business-api.tiktok.com/open_api/v1.3"
        self.headers = {
            "Access-Token": self.access_token,
            "Content-Type": "application/json"
        }
        
    def _make_request(
        self,
        endpoint: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Dict:
        """Make HTTP request to TikTok Ads API"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers, params=params)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logging.error(f"TikTok API request failed: {str(e)}")
            raise
            
    def get_campaign_performance(
        self,
        advertiser_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch campaign performance metrics"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
            
        params = {
            "advertiser_id": advertiser_id,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "page_size": 1000,
            "metrics": [
                "campaign_name",
                "spend",
                "impressions",
                "clicks",
                "conversions",
                "conversion_rate",
                "ctr",
                "cpc"
            ]
        }
        
        try:
            response = self._make_request("reports/integrated/get/", params=params)
            
            if response["code"] == 0:
                data = response["data"]["list"]
                return pd.DataFrame(data)
            else:
                raise Exception(f"TikTok API error: {response['message']}")
                
        except Exception as e:
            logging.error(f"Error fetching campaign performance: {str(e)}")
            raise
            
    def get_audience_insights(self, advertiser_id: str, campaign_id: str) -> Dict:
        """Get detailed audience insights for a campaign"""
        params = {
            "advertiser_id": advertiser_id,
            "campaign_ids": [campaign_id],
            "dimensions": ["age", "gender", "country", "platform"]
        }
        
        try:
            response = self._make_request("audience/analysis/get/", params=params)
            
            if response["code"] == 0:
                return self._process_audience_data(response["data"])
            else:
                raise Exception(f"TikTok API error: {response['message']}")
                
        except Exception as e:
            logging.error(f"Error fetching audience insights: {str(e)}")
            raise
            
    def _process_audience_data(self, raw_data: Dict) -> Dict:
        """Process raw audience data into structured insights"""
        processed_data = {
            "demographics": {
                "age_groups": {},
                "gender": {},
                "countries": {}
            },
            "platforms": {},
            "engagement_metrics": {
                "avg_watch_time": raw_data.get("avg_watch_time", 0),
                "completion_rate": raw_data.get("completion_rate", 0),
                "engagement_rate": raw_data.get("engagement_rate", 0)
            }
        }
        
        # Process age groups
        for age_data in raw_data.get("age_distribution", []):
            processed_data["demographics"]["age_groups"][age_data["age_group"]] = {
                "percentage": age_data["percentage"],
                "engagement_rate": age_data["engagement_rate"]
            }
            
        # Process gender distribution
        for gender_data in raw_data.get("gender_distribution", []):
            processed_data["demographics"]["gender"][gender_data["gender"]] = {
                "percentage": gender_data["percentage"]
            }
            
        # Process geographical distribution
        for country_data in raw_data.get("country_distribution", []):
            processed_data["demographics"]["countries"][country_data["country"]] = {
                "percentage": country_data["percentage"],
                "engagement_rate": country_data["engagement_rate"]
            }
            
        return processed_data
        
    def optimize_campaign(
        self,
        advertiser_id: str,
        campaign_id: str,
        optimizations: Dict
    ) -> Dict:
        """Apply optimization recommendations to campaign"""
        data = {
            "advertiser_id": advertiser_id,
            "campaign_id": campaign_id,
            "modifications": optimizations
        }
        
        try:
            response = self._make_request(
                "campaign/update/",
                method="POST",
                data=data
            )
            
            if response["code"] == 0:
                return {"status": "success", "message": "Campaign updated successfully"}
            else:
                raise Exception(f"TikTok API error: {response['message']}")
                
        except Exception as e:
            logging.error(f"Error optimizing campaign: {str(e)}")
            raise
            
    def get_creative_analysis(
        self,
        advertiser_id: str,
        campaign_id: str
    ) -> Dict:
        """Analyze creative performance for a campaign"""
        params = {
            "advertiser_id": advertiser_id,
            "campaign_id": campaign_id,
            "metrics": [
                "video_watch_actions",
                "video_views_p25",
                "video_views_p50",
                "video_views_p75",
                "video_views_p100",
                "engagement_rate",
                "shares",
                "comments",
                "likes"
            ]
        }
        
        try:
            response = self._make_request("creative/analysis/get/", params=params)
            
            if response["code"] == 0:
                return self._process_creative_data(response["data"])
            else:
                raise Exception(f"TikTok API error: {response['message']}")
                
        except Exception as e:
            logging.error(f"Error analyzing creative performance: {str(e)}")
            raise
            
    def _process_creative_data(self, raw_data: Dict) -> Dict:
        """Process raw creative performance data"""
        return {
            "video_metrics": {
                "watch_time": raw_data.get("avg_watch_time", 0),
                "completion_rates": {
                    "25%": raw_data.get("video_views_p25", 0),
                    "50%": raw_data.get("video_views_p50", 0),
                    "75%": raw_data.get("video_views_p75", 0),
                    "100%": raw_data.get("video_views_p100", 0)
                }
            },
            "engagement_metrics": {
                "engagement_rate": raw_data.get("engagement_rate", 0),
                "shares": raw_data.get("shares", 0),
                "comments": raw_data.get("comments", 0),
                "likes": raw_data.get("likes", 0)
            },
            "recommendations": self._generate_creative_recommendations(raw_data)
        }
        
    def _generate_creative_recommendations(self, data: Dict) -> List[Dict]:
        """Generate recommendations based on creative performance"""
        recommendations = []
        
        # Analyze video completion rates
        if data.get("video_views_p25", 0) > 0:
            drop_off_rate = 1 - (data.get("video_views_p100", 0) / data.get("video_views_p25", 0))
            if drop_off_rate > 0.7:
                recommendations.append({
                    "type": "video_length",
                    "message": "Consider shortening video length due to high drop-off rate",
                    "priority": "high"
                })
                
        # Analyze engagement metrics
        if data.get("engagement_rate", 0) < 0.02:
            recommendations.append({
                "type": "engagement",
                "message": "Improve video engagement with more interactive elements",
                "priority": "medium"
            })
            
        return recommendations
