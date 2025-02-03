import logging
import time
from datetime import datetime
import psutil
import numpy as np
from typing import Dict, List
import json
from pathlib import Path

class SystemMonitor:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        self._setup_logging()
        
        # Initialize metrics storage
        self.metrics = {
            'system': {},
            'model': {},
            'api': {}
        }
        
    def _setup_logging(self):
        """Configure logging system"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Main logger
        main_logger = logging.getLogger('AI-MarketingX')
        main_logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(
            self.log_dir / 'system.log'
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        main_logger.addHandler(file_handler)
        
        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(log_format))
        main_logger.addHandler(stream_handler)
        
    def monitor_system_resources(self) -> Dict:
        """Monitor system resource usage"""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage_percent': cpu_usage,
                'memory_usage_percent': memory.percent,
                'memory_available_gb': memory.available / (1024 ** 3),
                'disk_usage_percent': disk.percent,
                'disk_free_gb': disk.free / (1024 ** 3)
            }
            
            self.metrics['system'] = metrics
            logging.info(f"System metrics: {json.dumps(metrics, indent=2)}")
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error monitoring system resources: {str(e)}")
            raise
            
    def monitor_model_performance(self, model_metrics: Dict):
        """Monitor ML model performance metrics"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'metrics': model_metrics
            }
            
            # Calculate moving averages
            if 'history' not in self.metrics['model']:
                self.metrics['model']['history'] = []
                
            self.metrics['model']['history'].append(metrics)
            self.metrics['model']['current'] = metrics
            
            # Keep only last 100 records
            if len(self.metrics['model']['history']) > 100:
                self.metrics['model']['history'].pop(0)
                
            # Calculate trends
            self._analyze_model_trends()
            
            logging.info(f"Model metrics: {json.dumps(metrics, indent=2)}")
            
        except Exception as e:
            logging.error(f"Error monitoring model performance: {str(e)}")
            raise
            
    def monitor_api_health(self, api_metrics: Dict):
        """Monitor API health and performance"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'response_times': api_metrics.get('response_times', {}),
                'error_rates': api_metrics.get('error_rates', {}),
                'request_count': api_metrics.get('request_count', 0)
            }
            
            self.metrics['api'] = metrics
            
            # Check for anomalies
            self._detect_api_anomalies(metrics)
            
            logging.info(f"API metrics: {json.dumps(metrics, indent=2)}")
            
        except Exception as e:
            logging.error(f"Error monitoring API health: {str(e)}")
            raise
            
    def _analyze_model_trends(self):
        """Analyze trends in model performance"""
        try:
            history = self.metrics['model']['history']
            if len(history) < 2:
                return
                
            # Calculate metrics trends
            metrics_trends = {}
            for metric in history[-1]['metrics'].keys():
                values = [h['metrics'][metric] for h in history]
                trend = np.polyfit(range(len(values)), values, 1)[0]
                metrics_trends[metric] = {
                    'trend': float(trend),
                    'status': 'improving' if trend > 0 else 'declining'
                }
                
            self.metrics['model']['trends'] = metrics_trends
            
        except Exception as e:
            logging.error(f"Error analyzing model trends: {str(e)}")
            raise
            
    def _detect_api_anomalies(self, current_metrics: Dict):
        """Detect anomalies in API performance"""
        try:
            # Define thresholds
            thresholds = {
                'response_time_ms': 1000,  # 1 second
                'error_rate': 0.05,  # 5%
            }
            
            # Check response times
            for endpoint, response_time in current_metrics['response_times'].items():
                if response_time > thresholds['response_time_ms']:
                    logging.warning(
                        f"High response time detected for {endpoint}: "
                        f"{response_time}ms"
                    )
                    
            # Check error rates
            for endpoint, error_rate in current_metrics['error_rates'].items():
                if error_rate > thresholds['error_rate']:
                    logging.warning(
                        f"High error rate detected for {endpoint}: "
                        f"{error_rate*100}%"
                    )
                    
        except Exception as e:
            logging.error(f"Error detecting API anomalies: {str(e)}")
            raise
            
    def generate_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'system_health': self.metrics['system'],
                'model_performance': {
                    'current': self.metrics['model'].get('current', {}),
                    'trends': self.metrics['model'].get('trends', {})
                },
                'api_health': self.metrics['api']
            }
            
            # Save report
            report_path = self.log_dir / f"report_{int(time.time())}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            return report
            
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")
            raise
