from pymongo import MongoClient
from typing import Dict, List
import logging
from datetime import datetime

class MongoDBClient:
    def __init__(self, connection_string: str, database_name: str):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.db = None
        
    def connect(self):
        """Establish connection to MongoDB"""
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.database_name]
            logging.info("Successfully connected to MongoDB")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
            
    def store_campaign_data(self, campaign_data: Dict):
        """Store campaign performance data"""
        try:
            collection = self.db.campaign_performance
            campaign_data['timestamp'] = datetime.now()
            result = collection.insert_one(campaign_data)
            return result.inserted_id
            
        except Exception as e:
            logging.error(f"Error storing campaign data: {str(e)}")
            raise
            
    def store_optimization_results(self, optimization_data: Dict):
        """Store optimization results and recommendations"""
        try:
            collection = self.db.optimization_results
            optimization_data['timestamp'] = datetime.now()
            result = collection.insert_one(optimization_data)
            return result.inserted_id
            
        except Exception as e:
            logging.error(f"Error storing optimization results: {str(e)}")
            raise
            
    def get_campaign_history(self, campaign_id: str) -> List[Dict]:
        """Retrieve historical campaign performance"""
        try:
            collection = self.db.campaign_performance
            return list(collection.find(
                {'campaign_id': campaign_id},
                {'_id': 0}
            ).sort('timestamp', -1))
            
        except Exception as e:
            logging.error(f"Error retrieving campaign history: {str(e)}")
            raise
            
    def get_optimization_history(self, campaign_id: str) -> List[Dict]:
        """Retrieve optimization history for a campaign"""
        try:
            collection = self.db.optimization_results
            return list(collection.find(
                {'campaign_id': campaign_id},
                {'_id': 0}
            ).sort('timestamp', -1))
            
        except Exception as e:
            logging.error(f"Error retrieving optimization history: {str(e)}")
            raise
            
    def store_model_metrics(self, model_metrics: Dict):
        """Store model performance metrics"""
        try:
            collection = self.db.model_metrics
            model_metrics['timestamp'] = datetime.now()
            result = collection.insert_one(model_metrics)
            return result.inserted_id
            
        except Exception as e:
            logging.error(f"Error storing model metrics: {str(e)}")
            raise
            
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logging.info("MongoDB connection closed")
