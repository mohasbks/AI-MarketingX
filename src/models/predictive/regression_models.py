import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import logging
from typing import Dict, Tuple, List, Optional

class AdvancedRegressionModel:
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        
    def _initialize_model(self):
        """Initialize the regression model based on type"""
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif self.model_type == 'lasso':
            self.model = Lasso(alpha=1.0)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict:
        """Optimize model hyperparameters using GridSearchCV"""
        param_grid = self._get_param_grid()
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        return {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_
        }
        
    def _get_param_grid(self) -> Dict:
        """Get parameter grid for hyperparameter optimization"""
        if self.model_type == 'xgboost':
            return {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'n_estimators': [100, 200, 300],
                'min_child_weight': [1, 3, 5]
            }
        elif self.model_type in ['gradient_boosting', 'random_forest']:
            return {
                'max_depth': [3, 5, 7],
                'n_estimators': [100, 200, 300],
                'min_samples_split': [2, 5, 10]
            }
        else:
            return {}
            
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimize: bool = True
    ) -> Dict:
        """Train the regression model"""
        try:
            if self.model is None:
                self._initialize_model()
                
            if optimize:
                optimization_results = self.optimize_hyperparameters(X_train, y_train)
                logging.info(f"Optimization results: {optimization_results}")
                
            self.model.fit(X_train, y_train)
            
            # Calculate feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = self.model.feature_importances_
                
            # Perform cross-validation
            cv_scores = cross_val_score(
                self.model,
                X_train,
                y_train,
                cv=5,
                scoring='neg_mean_squared_error'
            )
            
            return {
                'cv_scores': -cv_scores,
                'cv_mean': -cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise
            
    def predict(
        self,
        X: np.ndarray,
        return_confidence: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with confidence intervals for supported models"""
        try:
            predictions = self.model.predict(X)
            
            if return_confidence and self.model_type in ['random_forest', 'gradient_boosting']:
                if hasattr(self.model, 'estimators_'):
                    # Get predictions from all trees
                    tree_predictions = np.array([
                        tree.predict(X) for tree in self.model.estimators_
                    ])
                    confidence = np.std(tree_predictions, axis=0)
                    return predictions, confidence
                    
            return predictions, None
            
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            raise
            
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """Evaluate model performance"""
        try:
            predictions = self.predict(X_test)[0]
            
            metrics = {
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'r2': r2_score(y_test, predictions)
            }
            
            # Calculate additional metrics
            metrics['mae'] = np.mean(np.abs(y_test - predictions))
            metrics['mape'] = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating model: {str(e)}")
            raise
            
    def get_feature_importance(self, feature_names: List[str] = None) -> Dict:
        """Get feature importance scores"""
        if self.feature_importance is None:
            return {}
            
        importance_dict = {}
        for idx, importance in enumerate(self.feature_importance):
            feature_name = feature_names[idx] if feature_names else f"feature_{idx}"
            importance_dict[feature_name] = float(importance)
            
        return dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
