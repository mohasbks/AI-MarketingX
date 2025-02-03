from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
import pandas as pd

class GoogleAdsIntegration:
    def __init__(self, credentials_path):
        self.client = None
        self.credentials_path = credentials_path
        
    def initialize_client(self):
        """Initialize Google Ads API client"""
        try:
            self.client = GoogleAdsClient.load_from_storage(self.credentials_path)
        except GoogleAdsException as ex:
            for error in ex.failure.errors:
                print(f'Error initializing Google Ads client: {error.message}')
            raise
            
    def get_campaign_performance(self, customer_id, date_range=None):
        """Fetch campaign performance data"""
        if not self.client:
            self.initialize_client()
            
        ga_service = self.client.get_service("GoogleAdsService")
        
        query = """
            SELECT
                campaign.id,
                campaign.name,
                metrics.impressions,
                metrics.clicks,
                metrics.cost_micros,
                metrics.conversions
            FROM campaign
            WHERE campaign.status = 'ENABLED'
        """
        
        try:
            response = ga_service.search_stream(customer_id=customer_id, query=query)
            
            campaign_data = []
            for batch in response:
                for row in batch.results:
                    campaign_data.append({
                        'campaign_id': row.campaign.id,
                        'campaign_name': row.campaign.name,
                        'impressions': row.metrics.impressions,
                        'clicks': row.metrics.clicks,
                        'cost': row.metrics.cost_micros / 1000000,
                        'conversions': row.metrics.conversions
                    })
                    
            return pd.DataFrame(campaign_data)
            
        except GoogleAdsException as ex:
            for error in ex.failure.errors:
                print(f'Request failed with error: {error.message}')
            raise
            
    def update_campaign_budget(self, customer_id, campaign_id, new_budget):
        """Update campaign budget"""
        if not self.client:
            self.initialize_client()
            
        campaign_budget_service = self.client.get_service("CampaignBudgetService")
        campaign_service = self.client.get_service("CampaignService")
        
        try:
            # Implementation for updating campaign budget
            pass
        except GoogleAdsException as ex:
            for error in ex.failure.errors:
                print(f'Failed to update campaign budget: {error.message}')
            raise
