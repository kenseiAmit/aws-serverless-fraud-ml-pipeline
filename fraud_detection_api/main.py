import os
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors

class InputData(BaseModel):
    features: list[float]

app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")

# Initialize Spark session
spark = SparkSession.builder.appName("FraudDetectionAPI").getOrCreate()

model = None

def download_s3_directory(bucket_name: str, s3_prefix: str, local_dir: str):
    """
    Downloads all objects under an S3 prefix to a local directory (intermediate storage).
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    os.makedirs(local_dir, exist_ok=True)
    
    downloaded_any = False
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            rel_path = os.path.relpath(s3_key, s3_prefix)
            local_file_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            s3.download_file(bucket_name, s3_key, local_file_path)
            downloaded_any = True
    return downloaded_any

@app.on_event('startup')
async def load_model():
    global model
    # Cloud Storage location in Amazon S3
    s3_bucket = os.environ.get("S3_BUCKET", "credit-card-fraud-pipeline-117913377161")
    s3_prefix = os.environ.get("MODEL_PREFIX", "model/fraud_detection_model_latest")
    
    # Intermediate local storage location on the container instance
    local_model_dir = os.environ.get("LOCAL_MODEL_DIR", "/app/model/fraud_detection_model_latest")
    if not os.path.exists(os.path.dirname(local_model_dir)):
        local_model_dir = os.path.abspath("model/fraud_detection_model_latest")

    print(f"Loading model from S3 bucket '{s3_bucket}', prefix '{s3_prefix}' to intermediate location '{local_model_dir}'...")

    try:
        downloaded = download_s3_directory(s3_bucket, s3_prefix, local_model_dir)
        if downloaded:
            print(f"Model artifacts successfully downloaded from S3 to {local_model_dir}")
        else:
            print(f"No objects downloaded from S3; checking local path {local_model_dir}")
    except Exception as e:
        print(f"Notice: S3 download encountered: {e}. Falling back to existing local model if present.")

    # Load the PySpark model from intermediate storage
    if os.path.exists(local_model_dir):
        model = PipelineModel.load(local_model_dir)
        print("PipelineModel successfully loaded.")
    else:
        raise RuntimeError(f"Model path {local_model_dir} not found.")

@app.get("/")
def root():
    return {"status": "ok", "service": "Credit Card Fraud Detection API", "model_loaded": model is not None}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post('/predict/')
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    # Declaring Feature Columns (30 features matching dataset schema)
    columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
               'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 
               'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

    if len(data.features) != len(columns):
        raise HTTPException(
            status_code=400, 
            detail=f"Expected {len(columns)} features, but received {len(data.features)}"
        )

    # Create a Spark DataFrame for prediction
    df = spark.createDataFrame([data.features], columns)

    # Make prediction
    predictions = model.transform(df)

    prediction = predictions.select("prediction").collect()[0][0]
    
    return {'prediction': float(prediction), 'is_fraud': bool(prediction == 1.0)}
