from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.functions import col, when, expr
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from typing import Dict, List, Optional
import logging

class SparkProcessor:
    def __init__(self, app_name: str = "AI-MarketingX"):
        self.spark = self._create_spark_session(app_name)
        self.feature_columns = [
            "daily_spend",
            "impressions",
            "clicks",
            "conversions",
            "ctr",
            "cpc",
            "engagement_rate"
        ]
        
    def _create_spark_session(self, app_name: str) -> SparkSession:
        """Create and configure Spark session"""
        try:
            return (SparkSession.builder
                    .appName(app_name)
                    .config("spark.driver.memory", "4g")
                    .config("spark.executor.memory", "4g")
                    .config("spark.sql.adaptive.enabled", "true")
                    .config("spark.sql.shuffle.partitions", "200")
                    .getOrCreate())
        except Exception as e:
            logging.error(f"Error creating Spark session: {str(e)}")
            raise
            
    def process_campaign_data(
        self,
        data_path: str,
        output_path: Optional[str] = None
    ) -> None:
        """Process campaign data using Spark"""
        try:
            # Define schema for better performance
            schema = StructType([
                StructField("campaign_id", StringType(), True),
                StructField("daily_spend", DoubleType(), True),
                StructField("impressions", IntegerType(), True),
                StructField("clicks", IntegerType(), True),
                StructField("conversions", IntegerType(), True),
                StructField("ctr", DoubleType(), True),
                StructField("cpc", DoubleType(), True),
                StructField("engagement_rate", DoubleType(), True),
                StructField("platform", StringType(), True),
                StructField("campaign_type", StringType(), True)
            ])
            
            # Read data
            df = self.spark.read.schema(schema).parquet(data_path)
            
            # Process data
            processed_df = self._preprocess_data(df)
            
            # Train models
            models = self._train_models(processed_df)
            
            # Generate predictions
            predictions = self._generate_predictions(processed_df, models)
            
            # Save results
            if output_path:
                predictions.write.mode("overwrite").parquet(output_path)
                
            return predictions
            
        except Exception as e:
            logging.error(f"Error processing campaign data: {str(e)}")
            raise
            
    def _preprocess_data(self, df):
        """Preprocess data using Spark ML"""
        try:
            # Handle missing values
            for column in self.feature_columns:
                df = df.withColumn(
                    column,
                    when(col(column).isNull(), 0).otherwise(col(column))
                )
                
            # Create feature vector
            assembler = VectorAssembler(
                inputCols=self.feature_columns,
                outputCol="features"
            )
            
            # Scale features
            scaler = StandardScaler(
                inputCol="features",
                outputCol="scaled_features",
                withStd=True,
                withMean=True
            )
            
            # String indexing for categorical variables
            indexers = [
                StringIndexer(
                    inputCol=col_name,
                    outputCol=f"{col_name}_index"
                )
                for col_name in ["platform", "campaign_type"]
            ]
            
            # Create and fit pipeline
            pipeline = Pipeline(stages=indexers + [assembler, scaler])
            processed_df = pipeline.fit(df).transform(df)
            
            return processed_df
            
        except Exception as e:
            logging.error(f"Error preprocessing data: {str(e)}")
            raise
            
    def _train_models(self, df) -> Dict:
        """Train Spark ML models"""
        try:
            # Split data
            train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
            
            # Train ROI prediction model
            rf_model = RandomForestRegressor(
                featuresCol="scaled_features",
                labelCol="roi",
                numTrees=100,
                maxDepth=10
            )
            roi_model = rf_model.fit(train_data)
            
            # Train conversion prediction model
            gbt_model = GBTClassifier(
                featuresCol="scaled_features",
                labelCol="conversion_index",
                maxIter=10
            )
            conversion_model = gbt_model.fit(train_data)
            
            # Evaluate models
            roi_predictions = roi_model.transform(test_data)
            conversion_predictions = conversion_model.transform(test_data)
            
            roi_evaluator = RegressionEvaluator(
                labelCol="roi",
                predictionCol="prediction",
                metricName="rmse"
            )
            
            conversion_evaluator = MulticlassClassificationEvaluator(
                labelCol="conversion_index",
                predictionCol="prediction",
                metricName="f1"
            )
            
            metrics = {
                "roi_rmse": roi_evaluator.evaluate(roi_predictions),
                "conversion_f1": conversion_evaluator.evaluate(conversion_predictions)
            }
            
            logging.info(f"Model metrics: {metrics}")
            
            return {
                "roi_model": roi_model,
                "conversion_model": conversion_model,
                "metrics": metrics
            }
            
        except Exception as e:
            logging.error(f"Error training models: {str(e)}")
            raise
            
    def _generate_predictions(self, df, models: Dict):
        """Generate predictions using trained models"""
        try:
            # Generate ROI predictions
            roi_predictions = models["roi_model"].transform(df)
            
            # Generate conversion predictions
            conversion_predictions = models["conversion_model"].transform(df)
            
            # Combine predictions
            final_predictions = (roi_predictions
                               .select("campaign_id", "prediction")
                               .join(conversion_predictions.select(
                                   "campaign_id",
                                   col("prediction").alias("conversion_prediction")
                               ), "campaign_id"))
            
            return final_predictions
            
        except Exception as e:
            logging.error(f"Error generating predictions: {str(e)}")
            raise
            
    def analyze_performance_patterns(self, df) -> Dict:
        """Analyze campaign performance patterns"""
        try:
            # Aggregate metrics by platform
            platform_metrics = (df.groupBy("platform")
                              .agg({"daily_spend": "avg",
                                   "ctr": "avg",
                                   "conversions": "sum",
                                   "roi": "avg"})
                              .collect())
            
            # Analyze temporal patterns
            temporal_patterns = (df.groupBy("hour_of_day")
                               .agg({"engagement_rate": "avg",
                                    "conversions": "avg"})
                               .collect())
            
            # Identify top performing campaigns
            top_campaigns = (df.orderBy(col("roi").desc())
                           .limit(10)
                           .select("campaign_id", "roi", "platform")
                           .collect())
            
            return {
                "platform_performance": platform_metrics,
                "temporal_patterns": temporal_patterns,
                "top_campaigns": top_campaigns
            }
            
        except Exception as e:
            logging.error(f"Error analyzing performance patterns: {str(e)}")
            raise
            
    def stop_spark(self):
        """Stop Spark session"""
        if self.spark:
            self.spark.stop()
