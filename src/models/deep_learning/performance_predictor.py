import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from typing import Dict, Tuple, List
import logging

class PerformancePredictor:
    def __init__(self, input_dim: int, learning_rate: float = 0.001):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self) -> models.Model:
        """Build neural network architecture for performance prediction"""
        try:
            model = models.Sequential([
                layers.Input(shape=(self.input_dim,)),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(3, activation='linear')  # Predicting ROI, CTR, and Conversion Rate
            ])
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
            return model
            
        except Exception as e:
            logging.error(f"Error building performance predictor model: {str(e)}")
            raise
            
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             validation_data: Tuple[np.ndarray, np.ndarray] = None,
             epochs: int = 100,
             batch_size: int = 32) -> Dict:
        """Train the performance prediction model"""
        try:
            history = self.model.fit(
                X_train,
                y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            return {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'] if validation_data else None,
                'mae': history.history['mae'],
                'val_mae': history.history['val_mae'] if validation_data else None
            }
            
        except Exception as e:
            logging.error(f"Error training performance predictor: {str(e)}")
            raise
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for new campaign data"""
        try:
            predictions = self.model.predict(X)
            return {
                'roi': predictions[:, 0],
                'ctr': predictions[:, 1],
                'conversion_rate': predictions[:, 2]
            }
            
        except Exception as e:
            logging.error(f"Error making predictions: {str(e)}")
            raise
            
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance on test data"""
        try:
            metrics = self.model.evaluate(X_test, y_test, verbose=0)
            return {
                'loss': metrics[0],
                'mae': metrics[1]
            }
            
        except Exception as e:
            logging.error(f"Error evaluating model: {str(e)}")
            raise
            
    def save_model(self, path: str):
        """Save model weights and architecture"""
        try:
            self.model.save(path)
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
            
    def load_model(self, path: str):
        """Load saved model"""
        try:
            self.model = models.load_model(path)
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
