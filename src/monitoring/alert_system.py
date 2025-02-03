import logging
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from datetime import datetime
import requests

class AlertSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.email_config = config.get('email', {})
        self.slack_config = config.get('slack', {})
        self.thresholds = config.get('thresholds', {})
        
    def check_alerts(self, metrics: Dict):
        """Check metrics against thresholds and trigger alerts if needed"""
        try:
            alerts = []
            
            # Check system metrics
            if 'system' in metrics:
                system_alerts = self._check_system_metrics(metrics['system'])
                alerts.extend(system_alerts)
                
            # Check model metrics
            if 'model' in metrics:
                model_alerts = self._check_model_metrics(metrics['model'])
                alerts.extend(model_alerts)
                
            # Check API metrics
            if 'api' in metrics:
                api_alerts = self._check_api_metrics(metrics['api'])
                alerts.extend(api_alerts)
                
            # Send alerts if any
            if alerts:
                self._send_alerts(alerts)
                
        except Exception as e:
            logging.error(f"Error checking alerts: {str(e)}")
            raise
            
    def _check_system_metrics(self, metrics: Dict) -> List[Dict]:
        """Check system metrics against thresholds"""
        alerts = []
        
        # CPU Usage
        if metrics.get('cpu_usage_percent', 0) > self.thresholds.get('cpu_usage', 90):
            alerts.append({
                'level': 'critical',
                'type': 'system',
                'message': f"High CPU usage: {metrics['cpu_usage_percent']}%"
            })
            
        # Memory Usage
        if metrics.get('memory_usage_percent', 0) > self.thresholds.get('memory_usage', 90):
            alerts.append({
                'level': 'critical',
                'type': 'system',
                'message': f"High memory usage: {metrics['memory_usage_percent']}%"
            })
            
        # Disk Usage
        if metrics.get('disk_usage_percent', 0) > self.thresholds.get('disk_usage', 90):
            alerts.append({
                'level': 'warning',
                'type': 'system',
                'message': f"High disk usage: {metrics['disk_usage_percent']}%"
            })
            
        return alerts
        
    def _check_model_metrics(self, metrics: Dict) -> List[Dict]:
        """Check model performance metrics"""
        alerts = []
        
        if 'trends' in metrics:
            for metric, trend_data in metrics['trends'].items():
                if trend_data['status'] == 'declining':
                    alerts.append({
                        'level': 'warning',
                        'type': 'model',
                        'message': f"Model performance declining for {metric}"
                    })
                    
        return alerts
        
    def _check_api_metrics(self, metrics: Dict) -> List[Dict]:
        """Check API health metrics"""
        alerts = []
        
        # Check response times
        for endpoint, response_time in metrics.get('response_times', {}).items():
            if response_time > self.thresholds.get('api_response_time', 1000):
                alerts.append({
                    'level': 'warning',
                    'type': 'api',
                    'message': f"High response time for {endpoint}: {response_time}ms"
                })
                
        # Check error rates
        for endpoint, error_rate in metrics.get('error_rates', {}).items():
            if error_rate > self.thresholds.get('api_error_rate', 0.05):
                alerts.append({
                    'level': 'critical',
                    'type': 'api',
                    'message': f"High error rate for {endpoint}: {error_rate*100}%"
                })
                
        return alerts
        
    def _send_alerts(self, alerts: List[Dict]):
        """Send alerts through configured channels"""
        try:
            # Send email alerts
            if self.email_config:
                self._send_email_alerts(alerts)
                
            # Send Slack alerts
            if self.slack_config:
                self._send_slack_alerts(alerts)
                
            # Log alerts
            for alert in alerts:
                log_level = logging.CRITICAL if alert['level'] == 'critical' else logging.WARNING
                logging.log(log_level, f"Alert: {alert['message']}")
                
        except Exception as e:
            logging.error(f"Error sending alerts: {str(e)}")
            raise
            
    def _send_email_alerts(self, alerts: List[Dict]):
        """Send alerts via email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from']
            msg['To'] = self.email_config['to']
            msg['Subject'] = f"AI-MarketingX Alerts - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Create email body
            body = "The following alerts have been detected:\n\n"
            for alert in alerts:
                body += f"[{alert['level'].upper()}] {alert['type']}: {alert['message']}\n"
                
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
                
        except Exception as e:
            logging.error(f"Error sending email alerts: {str(e)}")
            raise
            
    def _send_slack_alerts(self, alerts: List[Dict]):
        """Send alerts to Slack"""
        try:
            webhook_url = self.slack_config['webhook_url']
            
            for alert in alerts:
                # Create Slack message
                message = {
                    "text": f"*[{alert['level'].upper()}] {alert['type']}*\n{alert['message']}",
                    "attachments": [{
                        "color": "danger" if alert['level'] == 'critical' else "warning",
                        "fields": [
                            {
                                "title": "Type",
                                "value": alert['type'],
                                "short": True
                            },
                            {
                                "title": "Level",
                                "value": alert['level'],
                                "short": True
                            }
                        ]
                    }]
                }
                
                # Send to Slack
                response = requests.post(webhook_url, json=message)
                response.raise_for_status()
                
        except Exception as e:
            logging.error(f"Error sending Slack alerts: {str(e)}")
            raise
