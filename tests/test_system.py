import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predictive.regression_models import AdvancedRegressionModel
from src.models.reinforcement.ad_optimizer import AdCampaignOptimizer
from src.monitoring.system_monitor import SystemMonitor
from src.reporting.report_generator import ReportGenerator
from src.api.endpoints import app
from fastapi.testclient import TestClient

class TestAIMarketingSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.client = TestClient(app)
        cls.test_data = cls._generate_test_data()
        
    def setUp(self):
        """Set up before each test"""
        self.regression_model = AdvancedRegressionModel()
        self.optimizer = AdCampaignOptimizer(
            initial_budget=1000,
            initial_metrics={
                'ctr': 0.02,
                'cpc': 0.5,
                'conversion_rate': 0.03,
                'roi': 2.0
            }
        )
        self.monitor = SystemMonitor()
        self.report_generator = ReportGenerator()
        
    @staticmethod
    def _generate_test_data():
        """Generate synthetic test data"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='D'
        )
        
        return {
            'dates': dates,
            'daily_spend': np.random.uniform(100, 1000, len(dates)),
            'impressions': np.random.randint(1000, 10000, len(dates)),
            'clicks': np.random.randint(50, 500, len(dates)),
            'conversions': np.random.randint(1, 50, len(dates)),
            'revenue': np.random.uniform(200, 2000, len(dates)),
            'ctr': np.random.uniform(0.01, 0.05, len(dates)),
            'conversion_rate': np.random.uniform(0.01, 0.04, len(dates)),
            'roi': np.random.uniform(1.0, 3.0, len(dates))
        }
        
    def test_regression_model(self):
        """Test regression model performance"""
        # Prepare test data
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        X_test = np.random.rand(20, 5)
        y_test = np.random.rand(20)
        
        # Train model
        train_results = self.regression_model.train(X, y, optimize=True)
        
        # Make predictions
        predictions, confidence = self.regression_model.predict(
            X_test,
            return_confidence=True
        )
        
        # Evaluate model
        metrics = self.regression_model.evaluate(X_test, y_test)
        
        # Assertions
        self.assertIsNotNone(train_results)
        self.assertEqual(len(predictions), len(y_test))
        self.assertTrue(metrics['r2'] >= 0)
        self.assertTrue(metrics['mse'] >= 0)
        
    def test_reinforcement_learning(self):
        """Test reinforcement learning optimization"""
        # Train for a few episodes
        scores = self.optimizer.train(batch_size=32)
        
        # Generate recommendations
        state = np.random.rand(6)  # 6 features in state space
        actions, reward = self.optimizer.optimize_campaign(state)
        
        # Assertions
        self.assertTrue(len(scores) > 0)
        self.assertEqual(len(actions), 3)  # 3 actions: budget, bid, targeting
        self.assertIsInstance(reward, float)
        
    def test_system_monitoring(self):
        """Test system monitoring functionality"""
        # Monitor system resources
        metrics = self.monitor.monitor_system_resources()
        
        # Monitor model performance
        self.monitor.monitor_model_performance({
            'mse': 0.1,
            'r2': 0.85,
            'accuracy': 0.9
        })
        
        # Generate report
        report = self.monitor.generate_report()
        
        # Assertions
        self.assertIn('cpu_usage_percent', metrics)
        self.assertIn('memory_usage_percent', metrics)
        self.assertIn('disk_usage_percent', metrics)
        self.assertIsNotNone(report)
        
    def test_report_generation(self):
        """Test report generation functionality"""
        # Generate performance report
        report = self.report_generator.generate_performance_report(
            self.test_data,
            predictions={'ctr': (0.03, 0.01, 'improving')},
            recommendations={'immediate': [], 'short_term': [], 'long_term': []}
        )
        
        # Assertions
        self.assertIn('summary', report)
        self.assertIn('detailed_analysis', report)
        self.assertIn('predictions', report)
        self.assertIn('recommendations', report)
        self.assertIn('visualizations', report)
        
    def test_api_endpoints(self):
        """Test API endpoints"""
        # Test campaign analysis endpoint
        response = self.client.post(
            "/api/v1/analyze_campaign",
            headers={"X-API-Key": "your-api-key"},
            json=self.test_data
        )
        self.assertEqual(response.status_code, 200)
        
        # Test campaign optimization endpoint
        response = self.client.post(
            "/api/v1/optimize_campaign",
            headers={"X-API-Key": "your-api-key"},
            json={
                'budget': 1000,
                'metrics': {
                    'ctr': 0.02,
                    'cpc': 0.5,
                    'conversion_rate': 0.03,
                    'roi': 2.0
                }
            }
        )
        self.assertEqual(response.status_code, 200)
        
        # Test reports endpoint
        response = self.client.get(
            "/api/v1/campaign_reports",
            headers={"X-API-Key": "your-api-key"},
            params={
                'campaign_id': '123',
                'start_date': '2025-01-01',
                'end_date': '2025-01-29'
            }
        )
        self.assertEqual(response.status_code, 200)
        
    def test_end_to_end(self):
        """Test complete system workflow"""
        # 1. Train models
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        self.regression_model.train(X, y, optimize=True)
        self.optimizer.train(batch_size=32)
        
        # 2. Analyze campaign
        response = self.client.post(
            "/api/v1/analyze_campaign",
            headers={"X-API-Key": "your-api-key"},
            json=self.test_data
        )
        analysis_results = response.json()
        
        # 3. Generate recommendations
        response = self.client.post(
            "/api/v1/optimize_campaign",
            headers={"X-API-Key": "your-api-key"},
            json={
                'budget': 1000,
                'metrics': {
                    'ctr': 0.02,
                    'cpc': 0.5,
                    'conversion_rate': 0.03,
                    'roi': 2.0
                }
            }
        )
        optimization_results = response.json()
        
        # 4. Generate report
        report = self.report_generator.generate_performance_report(
            self.test_data,
            predictions=analysis_results['predictions'],
            recommendations=optimization_results['recommendations']
        )
        
        # Assertions
        self.assertTrue(all(key in analysis_results for key in ['status', 'historical_analysis', 'predictions']))
        self.assertTrue(all(key in optimization_results for key in ['status', 'recommendations']))
        self.assertTrue(all(key in report for key in ['summary', 'detailed_analysis', 'predictions', 'recommendations']))
        
if __name__ == '__main__':
    unittest.main()
