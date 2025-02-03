import pandas as pd
import numpy as np
from typing import Dict, List, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

class DataProcessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {
            'numerical': [
                'daily_spend',
                'impressions',
                'clicks',
                'conversions',
                'ctr',
                'cpc'
            ],
            'categorical': [
                'campaign_type',
                'platform',
                'targeting_type',
                'device_type'
            ]
        }
        
    def process_campaign_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process raw campaign data for model input"""
        try:
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Process numerical features
            data = self._process_numerical_features(data)
            
            # Process categorical features
            data = self._process_categorical_features(data)
            
            # Create derived features
            data = self._create_derived_features(data)
            
            return data
            
        except Exception as e:
            logging.error(f"Error processing campaign data: {str(e)}")
            raise
            
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill numerical missing values with median
        for col in self.feature_columns['numerical']:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].median())
                
        # Fill categorical missing values with mode
        for col in self.feature_columns['categorical']:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].mode()[0])
                
        return data
        
    def _process_numerical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        for col in self.feature_columns['numerical']:
            if col in data.columns:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    data[f'{col}_scaled'] = self.scalers[col].fit_transform(data[[col]])
                else:
                    data[f'{col}_scaled'] = self.scalers[col].transform(data[[col]])
                    
        return data
        
    def _process_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        for col in self.feature_columns['categorical']:
            if col in data.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    data[f'{col}_encoded'] = self.encoders[col].fit_transform(data[col])
                else:
                    data[f'{col}_encoded'] = self.encoders[col].transform(data[col])
                    
        return data
        
    def _create_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing ones"""
        # Calculate ROI
        if all(col in data.columns for col in ['conversions', 'daily_spend']):
            data['roi'] = (data['conversions'] * 100) / data['daily_spend']
            
        # Calculate engagement rate
        if all(col in data.columns for col in ['clicks', 'impressions']):
            data['engagement_rate'] = data['clicks'] / data['impressions']
            
        # Calculate cost per conversion
        if all(col in data.columns for col in ['daily_spend', 'conversions']):
            data['cost_per_conversion'] = data['daily_spend'] / data['conversions'].replace(0, 1)
            
        return data
        
    def prepare_features_for_model(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare final feature matrix for model input"""
        feature_cols = (
            [f'{col}_scaled' for col in self.feature_columns['numerical']] +
            [f'{col}_encoded' for col in self.feature_columns['categorical']]
        )
        
        return data[feature_cols].values
        
    def get_feature_names(self) -> List[str]:
        """Get list of processed feature names"""
        return (
            [f'{col}_scaled' for col in self.feature_columns['numerical']] +
            [f'{col}_encoded' for col in self.feature_columns['categorical']]
        )
