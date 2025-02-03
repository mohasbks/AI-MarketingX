import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# إعدادات للعمل على CPU
tf.config.set_visible_devices([], 'GPU')

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # تقليل حجم النموذج للعمل بكفاءة على CPU
        self.query = layers.Dense(embed_dim // 2)
        self.key = layers.Dense(embed_dim // 2)
        self.value = layers.Dense(embed_dim // 2)
        self.combine = layers.Dense(embed_dim // 2)
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        
        # Linear transformations
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        
        # Split heads
        query = tf.reshape(query, (batch_size, -1, self.num_heads, self.head_dim))
        key = tf.reshape(key, (batch_size, -1, self.num_heads, self.head_dim))
        value = tf.reshape(value, (batch_size, -1, self.num_heads, self.head_dim))
        
        # Transpose for attention calculation
        query = tf.transpose(query, perm=[0, 2, 1, 3])
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        value = tf.transpose(value, perm=[0, 2, 1, 3])
        
        # Calculate attention scores
        scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        attention = tf.matmul(query, key, transpose_b=True) / scale
        attention_weights = tf.nn.softmax(attention, axis=-1)
        
        # Apply attention to values
        output = tf.matmul(attention_weights, value)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.embed_dim // 2))
        
        return self.combine(output)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim // 2),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        
    def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
        # Multi-head self attention
        attention_output = self.attention(inputs)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm1(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class AdvancedMarketingTransformer:
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 4,
        embed_dim: int = 256,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        learning_rate: float = 0.001
    ):
        self.input_dim = input_dim
        self.model = self._build_model(
            num_layers, embed_dim, num_heads, ff_dim, dropout
        )
        self.learning_rate = learning_rate
        self._compile_model()
        
    def _build_model(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float
    ) -> models.Model:
        """Build the transformer-based model architecture"""
        inputs = layers.Input(shape=(None, self.input_dim))
        
        # Input embedding
        x = layers.Dense(embed_dim // 2)(inputs)
        
        # Transformer blocks
        for _ in range(num_layers):
            x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)(x)
            
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Final prediction layers
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        
        # Multiple output heads
        roi_output = layers.Dense(1, name="roi")(x)
        ctr_output = layers.Dense(1, activation="sigmoid", name="ctr")(x)
        conv_output = layers.Dense(1, activation="sigmoid", name="conversion_rate")(x)
        budget_output = layers.Dense(1, name="budget_optimization")(x)
        
        return models.Model(
            inputs=inputs,
            outputs=[roi_output, ctr_output, conv_output, budget_output]
        )
        
    def _compile_model(self):
        """Compile the model with appropriate loss functions and metrics"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss={
                "roi": "mse",
                "ctr": "binary_crossentropy",
                "conversion_rate": "binary_crossentropy",
                "budget_optimization": "mse"
            },
            loss_weights={
                "roi": 1.0,
                "ctr": 0.5,
                "conversion_rate": 0.5,
                "budget_optimization": 0.8
            },
            metrics={
                "roi": ["mae", "mse"],
                "ctr": ["accuracy", "AUC"],
                "conversion_rate": ["accuracy", "AUC"],
                "budget_optimization": ["mae"]
            }
        )
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        validation_data: Optional[Tuple] = None,
        epochs: int = 100,
        batch_size: int = 32,
        callbacks: List = None
    ) -> Dict:
        """Train the model with the provided data"""
        try:
            history = self.model.fit(
                X_train,
                {
                    "roi": y_train["roi"],
                    "ctr": y_train["ctr"],
                    "conversion_rate": y_train["conversion_rate"],
                    "budget_optimization": y_train["budget"]
                },
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            return history.history
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            raise
            
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate predictions for new data"""
        try:
            predictions = self.model.predict(X)
            return {
                "roi": predictions[0],
                "ctr": predictions[1],
                "conversion_rate": predictions[2],
                "budget_optimization": predictions[3]
            }
        except Exception as e:
            logging.error(f"Error generating predictions: {str(e)}")
            raise
            
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        try:
            results = self.model.evaluate(
                X_test,
                {
                    "roi": y_test["roi"],
                    "ctr": y_test["ctr"],
                    "conversion_rate": y_test["conversion_rate"],
                    "budget_optimization": y_test["budget"]
                },
                verbose=0
            )
            
            metrics = {}
            for metric_name, value in zip(self.model.metrics_names, results):
                metrics[metric_name] = value
                
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating model: {str(e)}")
            raise
