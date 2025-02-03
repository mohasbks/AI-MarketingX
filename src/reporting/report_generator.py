import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging

class ReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_performance_report(
        self,
        campaign_data: Dict,
        predictions: Dict,
        recommendations: Dict
    ) -> Dict:
        """Generate comprehensive performance report"""
        try:
            report = {
                "summary": self._generate_summary(campaign_data),
                "detailed_analysis": self._generate_detailed_analysis(campaign_data),
                "predictions": self._format_predictions(predictions),
                "recommendations": self._format_recommendations(recommendations),
                "visualizations": self._generate_visualizations(campaign_data),
                "timestamp": datetime.now().isoformat()
            }
            
            # Save report
            report_path = self.output_dir / f"report_{int(datetime.now().timestamp())}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            return report
            
        except Exception as e:
            logging.error(f"Error generating performance report: {str(e)}")
            raise
            
    def _generate_summary(self, data: Dict) -> Dict:
        """Generate high-level performance summary"""
        return {
            "total_spend": sum(data['daily_spend']),
            "total_impressions": sum(data['impressions']),
            "total_clicks": sum(data['clicks']),
            "total_conversions": sum(data['conversions']),
            "average_ctr": np.mean(data['ctr']),
            "average_conversion_rate": np.mean(data['conversion_rate']),
            "roi": self._calculate_roi(data),
            "performance_trend": self._analyze_trend(data)
        }
        
    def _generate_detailed_analysis(self, data: Dict) -> Dict:
        """Generate detailed performance analysis"""
        return {
            "hourly_analysis": self._analyze_hourly_performance(data),
            "audience_analysis": self._analyze_audience_performance(data),
            "platform_analysis": self._analyze_platform_performance(data),
            "creative_analysis": self._analyze_creative_performance(data),
            "budget_analysis": self._analyze_budget_efficiency(data)
        }
        
    def _format_predictions(self, predictions: Dict) -> Dict:
        """Format and explain predictions"""
        return {
            "expected_performance": {
                metric: {
                    "value": value,
                    "confidence_interval": confidence,
                    "trend": trend
                }
                for metric, (value, confidence, trend) in predictions.items()
            },
            "risk_assessment": self._assess_prediction_risks(predictions),
            "opportunities": self._identify_opportunities(predictions)
        }
        
    def _format_recommendations(self, recommendations: Dict) -> Dict:
        """Format and prioritize recommendations"""
        return {
            "immediate_actions": self._prioritize_actions(
                recommendations['immediate'],
                "high"
            ),
            "short_term_actions": self._prioritize_actions(
                recommendations['short_term'],
                "medium"
            ),
            "long_term_actions": self._prioritize_actions(
                recommendations['long_term'],
                "low"
            ),
            "expected_impact": self._estimate_recommendation_impact(recommendations)
        }
        
    def _generate_visualizations(self, data: Dict) -> Dict:
        """Generate interactive visualizations"""
        return {
            "performance_trends": self._create_trend_charts(data),
            "audience_insights": self._create_audience_charts(data),
            "budget_allocation": self._create_budget_charts(data),
            "platform_comparison": self._create_platform_charts(data)
        }
        
    def _calculate_roi(self, data: Dict) -> float:
        """Calculate campaign ROI"""
        total_revenue = sum(data['revenue'])
        total_cost = sum(data['daily_spend'])
        return (total_revenue - total_cost) / total_cost if total_cost > 0 else 0
        
    def _analyze_trend(self, data: Dict) -> str:
        """Analyze overall performance trend"""
        recent_performance = data['roi'][-7:]  # Last 7 days
        trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        
        if trend > 0.05:
            return "strongly_improving"
        elif trend > 0:
            return "improving"
        elif trend > -0.05:
            return "stable"
        else:
            return "declining"
            
    def _analyze_hourly_performance(self, data: Dict) -> Dict:
        """Analyze performance by hour of day"""
        hourly_data = pd.DataFrame(data['hourly_metrics'])
        
        return {
            "best_performing_hours": self._identify_best_hours(hourly_data),
            "worst_performing_hours": self._identify_worst_hours(hourly_data),
            "hourly_patterns": self._identify_patterns(hourly_data)
        }
        
    def _analyze_audience_performance(self, data: Dict) -> Dict:
        """Analyze performance by audience segment"""
        audience_data = pd.DataFrame(data['audience_metrics'])
        
        return {
            "top_segments": self._identify_top_segments(audience_data),
            "underperforming_segments": self._identify_underperforming_segments(audience_data),
            "segment_opportunities": self._identify_segment_opportunities(audience_data)
        }
        
    def _analyze_platform_performance(self, data: Dict) -> Dict:
        """Analyze performance by platform"""
        platform_data = pd.DataFrame(data['platform_metrics'])
        
        return {
            "platform_comparison": self._compare_platforms(platform_data),
            "platform_trends": self._analyze_platform_trends(platform_data),
            "platform_recommendations": self._generate_platform_recommendations(platform_data)
        }
        
    def _analyze_creative_performance(self, data: Dict) -> Dict:
        """Analyze performance by creative"""
        creative_data = pd.DataFrame(data['creative_metrics'])
        
        return {
            "top_performing_creatives": self._identify_top_creatives(creative_data),
            "creative_insights": self._analyze_creative_elements(creative_data),
            "creative_recommendations": self._generate_creative_recommendations(creative_data)
        }
        
    def _analyze_budget_efficiency(self, data: Dict) -> Dict:
        """Analyze budget allocation efficiency"""
        budget_data = pd.DataFrame(data['budget_metrics'])
        
        return {
            "budget_utilization": self._analyze_budget_utilization(budget_data),
            "spend_efficiency": self._analyze_spend_efficiency(budget_data),
            "budget_recommendations": self._generate_budget_recommendations(budget_data)
        }
        
    def _create_trend_charts(self, data: Dict) -> List[Dict]:
        """Create interactive trend visualizations"""
        charts = []
        
        # Performance metrics over time
        fig = go.Figure()
        for metric in ['ctr', 'conversion_rate', 'roi']:
            fig.add_trace(go.Scatter(
                x=data['dates'],
                y=data[metric],
                name=metric.upper(),
                mode='lines+markers'
            ))
        charts.append(self._fig_to_dict(fig, "Performance Trends"))
        
        return charts
        
    def _create_audience_charts(self, data: Dict) -> List[Dict]:
        """Create audience insight visualizations"""
        charts = []
        
        # Audience performance comparison
        fig = px.treemap(
            data['audience_metrics'],
            path=['segment'],
            values='conversions',
            color='roi'
        )
        charts.append(self._fig_to_dict(fig, "Audience Performance"))
        
        return charts
        
    def _create_budget_charts(self, data: Dict) -> List[Dict]:
        """Create budget allocation visualizations"""
        charts = []
        
        # Budget allocation sunburst
        fig = px.sunburst(
            data['budget_metrics'],
            path=['platform', 'campaign_type', 'audience'],
            values='spend'
        )
        charts.append(self._fig_to_dict(fig, "Budget Allocation"))
        
        return charts
        
    def _create_platform_charts(self, data: Dict) -> List[Dict]:
        """Create platform comparison visualizations"""
        charts = []
        
        # Platform performance comparison
        fig = px.bar(
            data['platform_metrics'],
            x='platform',
            y=['impressions', 'clicks', 'conversions'],
            barmode='group'
        )
        charts.append(self._fig_to_dict(fig, "Platform Performance"))
        
        return charts
        
    def _fig_to_dict(self, fig: go.Figure, title: str) -> Dict:
        """Convert Plotly figure to dictionary"""
        return {
            "title": title,
            "data": fig.to_dict()
        }
        
    def _assess_prediction_risks(self, predictions: Dict) -> Dict:
        """Assess risks in predictions"""
        # Implementation here
        return {}
        
    def _identify_opportunities(self, predictions: Dict) -> List[Dict]:
        """Identify potential opportunities"""
        # Implementation here
        return []
        
    def _prioritize_actions(
        self,
        actions: List[Dict],
        priority: str
    ) -> List[Dict]:
        """Prioritize recommended actions"""
        # Implementation here
        return []
        
    def _estimate_recommendation_impact(
        self,
        recommendations: Dict
    ) -> Dict:
        """Estimate impact of recommendations"""
        # Implementation here
        return {}
