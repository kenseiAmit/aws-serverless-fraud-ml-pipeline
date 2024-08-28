<div align="center">
  <img src="assets/hero-banner.svg" alt="AWS Serverless Fraud ML Pipeline" width="100%" />
  <br/><br/>
  <p>
    <img src="https://img.shields.io/badge/Training-AWS%20Glue%204.0%20PySpark%20MLlib-FF9900?style=flat-square&amp;logo=amazonaws&amp;logoColor=white" alt="AWS Glue" />
    <img src="https://img.shields.io/badge/Serving-ECS%20Fargate%20%2B%20FastAPI-2563EB?style=flat-square&amp;logo=fastapi&amp;logoColor=white" alt="ECS Fargate" />
    <img src="https://img.shields.io/badge/Spark%20Training-10x%20Faster%20%28121s%29-16A34A?style=flat-square" alt="Speedup" />
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-475569?style=flat-square" alt="License" /></a>
  </p>
</div>

---

Production AWS machine learning platform migrating local PySpark credit card fraud detection (`284,807` transactions, severe `0.17%` fraud imbalance) to **serverless AWS Glue 4.0 distributed MLlib training**, **S3 event-driven AWS Lambda automated retraining & compliance archival**, and an **Amazon ECS Fargate FastAPI + PySpark real-time inference microservice**.

![Glue Spark Optimization & Telemetry](evidence/glue_spark_caching_benchmark.png)

---

## 1. Event-Driven Serverless & Containerized Architecture

```mermaid
flowchart LR
    A["S3 Data Lake<br/>data/incoming/*.csv"] -->|"s3:ObjectCreated:*"| B["AWS Lambda<br/>refresh_function.py"]
    B -->|"Archive + Timestamp"| C["S3 Compliance Archive<br/>data/archive/*_retrain_ts.csv"]
    B -->|"glue:StartJobRun"| D["AWS Glue 4.0 (4x G.1X)<br/>.repartition(16).cache() + RF"]
    D -->|"Write PipelineModel"| E["S3 Model Registry<br/>model/fraud_detection_model_latest/"]
    E -->|"IAM Task Role Startup Pull"| F["Amazon ECS Fargate<br/>FastAPI + OpenJDK 17 + PySpark"]
    G["Client / Payment Gateway"] -->|"POST /predict/"| F
```

---

## 2. Verified AWS Production Execution Gallery

| AWS Glue 4.0 Distributed Training (`2m 01s`, `0.1344 DPU-h`) | Serialized Spark MLlib `PipelineModel` in S3 |
| :---: | :---: |
| ![Glue Training Job](evidence/table2_glue_training_job.png) | ![S3 Model Artifact](evidence/table3_s3_model_artifact.png) |

| Amazon ECS Fargate Service & Live `curl /predict/` | S3 Event-Triggered Lambda Retraining & Archival |
| :---: | :---: |
| ![ECS Fargate Prediction API](evidence/table4_ecs_prediction_api.png) | ![Lambda Retraining](evidence/table5_lambda_retraining.png) |

| IAM Roles: AWS Glue & AWS Lambda | IAM Task & Execution Role: Amazon ECS Fargate |
| :---: | :---: |
| ![IAM Glue & Lambda](evidence/table7_iam_glue_lambda.png) | ![IAM ECS](evidence/table8_iam_ecs.png) |

Full architectural trade-offs and IAM policy matrices are documented in [`docs/ARCHITECTURE_REPORT.md`](docs/ARCHITECTURE_REPORT.md).

---

## 3. Live ECS Fargate Inference Verification

```json
// GET /health -> HTTP 200 OK
{
  "status": "healthy",
  "model_loaded": true,
  "spark_version": "3.3.0"
}

// POST /predict/ -> HTTP 200 OK
{
  "prediction": 0.0,
  "is_fraud": false
}
```

---

## 4. Repository Structure & Local Docker Quick Start

```text
aws-serverless-fraud-ml-pipeline/
├── fraud_detection_pipeline/            # AWS Glue 4.0 PySpark MLlib trainer (VectorAssembler + StandardScaler + RF)
├── refresh_function/                    # AWS Lambda S3 event archival & Glue job trigger
├── fraud_detection_api/                 # FastAPI + PySpark + OpenJDK 17 inference container & serialized model
├── ecs/fraud-detection-task-def.json    # AWS Fargate (1024 CPU / 2048 MB) task definition
├── data/incoming_transactions_batch3.csv# Sample 30-feature streaming batch (Time, V1..V28, Amount)
├── docs/ARCHITECTURE_REPORT.md          # Complete AWS service rationale & IAM role matrix
└── evidence/                            # Verified AWS Console, CloudWatch & live API screenshots
```

```bash
docker build -t fraud-detection-api ./fraud_detection_api
docker run -p 8000:8000 fraud-detection-api
curl http://localhost:8000/health
```

## License

MIT License — see [LICENSE](LICENSE).
