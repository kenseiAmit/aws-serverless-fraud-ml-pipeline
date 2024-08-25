import sys
import os
import datetime

# AWS Glue imports
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame, DynamicFrameCollection

# PySpark ML imports
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when
from pyspark.sql.types import DoubleType

# Glue function signature for Custom Transform & ETL pipelines
def MyTransform(glueContext, dfc, s3_bucket=None, model_save_path=None):
    """
    AWS Glue Custom Transform:
    Converts DynamicFrameCollection to Spark DataFrame, performs feature engineering,
    trains a RandomForestClassifier with class imbalance weighting, saves the trained
    PipelineModel to Amazon S3, and returns the output DynamicFrameCollection.
    """
    try:
        # Convert DynamicFrameCollection to DataFrame for Glue
        if isinstance(dfc, DynamicFrameCollection):
            first_key = list(dfc.keys())[0]
            dynamic_frame = dfc.select(first_key)
            df = dynamic_frame.toDF()
        elif isinstance(dfc, DynamicFrame):
            df = dfc.toDF()
        else:
            df = dfc

        if s3_bucket is None:
            s3_bucket = os.environ.get("S3_BUCKET", "credit-card-fraud-pipeline-117913377161")

        if model_save_path is None:
            model_save_path = f"s3://{s3_bucket}/model/fraud_detection_model_latest"

        # Determine feature columns (all columns except 'Class')
        numeric_columns = [c for c in df.columns if c != 'Class']
        print(f"Numeric feature columns ({len(numeric_columns)}): {numeric_columns}")

        # Cast all numeric columns to DoubleType efficiently in a single projection
        cast_exprs = [col(c).cast(DoubleType()).alias(c) if c != 'Class' else col(c) for c in df.columns]
        df = df.select(*cast_exprs)

        # Normalize numeric features
        assembler = VectorAssembler(inputCols=numeric_columns, outputCol="numericFeatures", handleInvalid="skip")
        scaler = StandardScaler(inputCol="numericFeatures", outputCol="scaledFeatures", withStd=True, withMean=False)

        # Combine scaled numeric features into final features vector
        finalAssembler = VectorAssembler(inputCols=["scaledFeatures"], outputCol="features", handleInvalid="skip")

        # Handle class imbalance by adjusting class weights
        class_counts = df.groupBy("Class").count().collect()
        total_count = sum([row['count'] for row in class_counts])
        weight_dict = {str(row['Class']): (total_count / row['count']) if row['count'] > 0 else 1.0 for row in class_counts}
        print(f"Computed class weights: {weight_dict}")
        
        weight_0 = float(weight_dict.get('0', 1.0))
        weight_1 = float(weight_dict.get('1', 1.0))

        # Add class weights to the DataFrame
        df = df.withColumn("weight", when(col("Class") == 0, weight_0).otherwise(weight_1))

        # Repartition and cache in memory for fast multi-pass ML training
        df = df.repartition(16).cache()
        record_count = df.count()
        print(f"Persisted dataset in memory across executor partitions: {record_count} records")

        # Convert Class to numeric label
        indexer = StringIndexer(inputCol="Class", outputCol="label", handleInvalid="skip")

        # Define the RandomForestClassifier with class weights
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=10, 
                                    maxDepth=5, maxBins=16, weightCol="weight", seed=42)

        # Create a pipeline
        pipeline = Pipeline(stages=[assembler, scaler, finalAssembler, indexer, rf])

        # Fit the model
        print("Training Random Forest Classifier on AWS Glue...")
        model = pipeline.fit(df)

        # Modify model saving for Glue (use S3 destination)
        model.write().overwrite().save(model_save_path)
        print(f"Model successfully trained and saved to S3 destination: {model_save_path}")

        # Also save a timestamped version for model lineage & versioning
        timestamp_str = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        versioned_path = f"s3://{s3_bucket}/model/history/fraud_detection_model_{timestamp_str}"
        model.write().overwrite().save(versioned_path)
        print(f"Versioned model artifact archived to: {versioned_path}")

        # Convert DataFrame back to DynamicFrame and return DynamicFrameCollection for Glue
        output_dynamic_frame = DynamicFrame.fromDF(df, glueContext, "output_dynamic_frame")
        return DynamicFrameCollection({"CustomTransform0": output_dynamic_frame}, glueContext)
    except Exception as e:
        print(f"Error in MyTransform: {e}")
        raise e

# AWS Glue Job Entrypoint
if __name__ == "__main__":
    expected_args = ['JOB_NAME']
    optional_args = ['S3_BUCKET', 'S3_INPUT_PATH', 'DATA_URL', 'MODEL_SAVE_PATH']
    
    passed_args = [arg.replace('--', '') for arg in sys.argv if arg.startswith('--')]
    keys_to_resolve = ['JOB_NAME']
    for opt in optional_args:
        if opt in passed_args:
            keys_to_resolve.append(opt)

    try:
        args = getResolvedOptions(sys.argv, keys_to_resolve)
    except Exception:
        args = {'JOB_NAME': 'fraud-detection-model-trainer'}

    job_name = args.get('JOB_NAME', 'fraud-detection-model-trainer')
    s3_bucket = args.get('S3_BUCKET', 'credit-card-fraud-pipeline-117913377161')
    
    # Support both DATA_URL and S3_INPUT_PATH as documented in README
    s3_input = args.get('DATA_URL') or args.get('S3_INPUT_PATH') or f"s3://{s3_bucket}/data/raw/creditcard.csv"
    model_save_path = args.get('MODEL_SAVE_PATH') or f"s3://{s3_bucket}/model/fraud_detection_model_latest"

    # Initialize SparkContext and GlueContext
    sc = SparkContext.getOrCreate()
    try:
        sc._jsc.hadoopConfiguration().set("mapred.output.committer.class", "org.apache.hadoop.mapred.FileOutputCommitter")
        print("Configured Hadoop output committer to FileOutputCommitter")
    except Exception as e:
        print(f"Warning setting Hadoop committer: {e}")

    glueContext = GlueContext(sc)
    spark = glueContext.spark_session
    job = Job(glueContext)
    job.init(job_name, args)

    print(f"Starting AWS Glue model training job: {job_name}")
    print(f"Reading dataset from: {s3_input}")
    print(f"Model will be saved to: {model_save_path}")

    # Read data from S3 using Glue DynamicFrame
    input_dynamic_frame = glueContext.create_dynamic_frame.from_options(
        format_options={"withHeader": True, "separator": ","},
        connection_type="s3",
        format="csv",
        connection_options={"paths": [s3_input], "recurse": True},
        transformation_ctx="input_dynamic_frame"
    )

    dfc = DynamicFrameCollection({"input": input_dynamic_frame}, glueContext)
    result_dfc = MyTransform(glueContext, dfc, s3_bucket=s3_bucket, model_save_path=model_save_path)

    job.commit()
    print("AWS Glue model training job completed successfully.")