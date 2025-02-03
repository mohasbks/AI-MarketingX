from facebook_business.api import FacebookAdsApi
from facebook_business.adobjects.adaccount import AdAccount
from facebook_business.adobjects.campaign import Campaign
import pandas as pd
from typing import Dict, List
import logging

class FacebookAdsIntegration:
    def __init__(self, app_id: str, app_secret: str, access_token: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.access_token = access_token
        self.api = None
        
    def initialize_client(self):
        """Initialize Facebook Ads API client"""
        try:
            FacebookAdsApi.init(self.app_id, self.app_secret, self.access_token)
            self.api = FacebookAdsApi.get_default()
        except Exception as e:
            logging.error(f"Failed to initialize Facebook Ads API: {str(e)}")
            raise
            
    def get_campaign_performance(self, ad_account_id: str, date_range: Dict = None) -> pd.DataFrame:
        """Fetch campaign performance data"""
        if not self.api:
            self.initialize_client()
            
        try:
            account = AdAccount(f'act_{ad_account_id}')
            fields = [
                'campaign_name',
                'spend',
                'impressions',
                'clicks',
                'conversions',
                'cpc',
                'ctr'
            ]
            
            params = {
                'time_range': date_range or {'since': '7d'},
                'level': 'campaign'
            }
            
            insights = account.get_insights(fields=fields, params=params)
            
            campaign_data = []
            for insight in insights:
                campaign_data.append({
                    'campaign_name': insight['campaign_name'],
                    'spend': float(insight['spend']),
                    'impressions': int(insight['impressions']),
                    'clicks': int(insight['clicks']),
                    'conversions': int(insight.get('conversions', 0)),
                    'cpc': float(insight['cpc']),
                    'ctr': float(insight['ctr'])
                })
                
            return pd.DataFrame(campaign_data)
            
        except Exception as e:
            logging.error(f"Error fetching Facebook campaign data: {str(e)}")
            raise
            
    def optimize_campaign(self, ad_account_id: str, campaign_id: str, optimizations: Dict):
        """Apply optimization recommendations to campaign"""
        if not self.api:
            self.initialize_client()
            
        try:
            campaign = Campaign(campaign_id)
            
            if 'budget' in optimizations:
                campaign.api_update(
                    fields=[],
                    params={'daily_budget': optimizations['budget']}
                )
                
            if 'targeting' in optimizations:
                campaign.api_update(
                    fields=[],
                    params={'targeting': optimizations['targeting']}
                )
                
            return {"status": "success", "message": "Campaign updated successfully"}
            
        except Exception as e:
            logging.error(f"Error optimizing Facebook campaign: {str(e)}")
            raise
            
    def get_audience_insights(self, ad_account_id: str, campaign_id: str) -> Dict:
        """Get detailed audience insights for a campaign"""
        if not self.api:
            self.initialize_client()
            
        try:
            campaign = Campaign(campaign_id)
            insights = campaign.get_insights(
                fields=[
                    'age',
                    'gender',
                    'country',
                    'placement',
                    'device_platform'
                ],
                params={'breakdowns': ['age', 'gender', 'country', 'placement']}
            )
            
            return {
                'demographics': self._process_demographics(insights),
                'placements': self._process_placements(insights),
                'devices': self._process_devices(insights)
            }
            
        except Exception as e:
            logging.error(f"Error fetching audience insights: {str(e)}")
            raise
            
    def _process_demographics(self, insights: List) -> Dict:
        """Process demographic data from insights"""
        demographics = {
            'age_groups': {},
            'gender': {},
            'countries': {}
        }
        
        for insight in insights:
            # Process age groups
            age = insight.get('age')
            if age:
                demographics['age_groups'][age] = demographics['age_groups'].get(age, 0) + 1
                
            # Process gender
            gender = insight.get('gender')
            if gender:
                demographics['gender'][gender] = demographics['gender'].get(gender, 0) + 1
                
            # Process countries
            country = insight.get('country')
            if country:
                demographics['countries'][country] = demographics['countries'].get(country, 0) + 1
                
        return demographics
