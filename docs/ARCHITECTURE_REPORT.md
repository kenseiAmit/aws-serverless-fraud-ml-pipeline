# Cloud-Native Fraud Detection Architecture & AWS Operational Evidence

## 1. Service Selection & Cloud Migration Rationale

| Pipeline Capability | Managed AWS Service | Architectural Justification |
| :--- | :--- | :--- |
| **Data Lake & Model Registry** | **Amazon S3** | 11-nines durability unifying raw historical ledgers (`data/raw/`), streaming incoming batches (`data/incoming/`), timestamped compliance archives (`data/archive/`), Glue scripts, and serialized Spark MLlib `PipelineModel` parquets (`model/fraud_detection_model_latest/`). |
| **Distributed MLlib Training** | **AWS Glue 4.0 (Spark 3.3)** | Serverless `4 × G.1X` (`16 DPU`) execution handling severe `0.17%` fraud imbalance across `284,807` transactions. `.repartition(16).cache()` eliminated repeated S3 passes across ensemble tree splits, cutting training runtime from `>20m` to **`121–130s` (`0.1344 DPU-hours`)** with zero idle compute cost. |
| **Real-Time Inference API** | **Amazon ECS on AWS Fargate** | Serverless `1024 CPU / 2048 MB` container (`python:3.10-slim` + headless OpenJDK 17 + PySpark 3.3 + FastAPI) hydrating the latest `PipelineModel` dynamically from S3 via IAM Task Role on startup (`GET /health`, `POST /predict/`). |
| **Event-Driven Retraining** | **AWS Lambda + S3 Events** | `s3:ObjectCreated:*` trigger on `data/incoming/` archives new CSV batches to `data/archive/<batch>_retrain_<ts>.csv`, purges the landing key to prevent duplicate runs, and invokes `glue.start_job_run()` asynchronously. |
| **Centralized Telemetry** | **Amazon CloudWatch** | Unified operational logging across `/aws-glue/jobs/*`, `/aws/lambda/credit-card-fraud-retraining-function`, and `/ecs/fraud-detection-api`. |

---

## 2. IAM Least-Privilege Role Matrix

| Workload | IAM Role | Trust Principal & Scoped Capabilities |
| :--- | :--- | :--- |
| **AWS Glue Trainer** | `AWSGlueServiceRole-FraudDetection` | `glue.amazonaws.com` — Read S3 scripts/datasets, write serialized MLlib `PipelineModel` parquets, publish driver/executor telemetry to CloudWatch. |
| **Lambda Retraining Trigger** | `FraudDetectionLambdaRole` | `lambda.amazonaws.com` — Copy/delete S3 `data/incoming/` → `data/archive/`, invoke `glue:StartJobRun`, write CloudWatch logs. |
| **ECS Fargate Inference Service** | `FraudDetectionECSTaskRole` | `ecs-tasks.amazonaws.com` — Pull ECR container image, stream `awslogs`, and authenticate `boto3` S3 model downloads with zero static credentials. |

---

## 3. Verified Production Execution Evidence

### AWS Glue Distributed Spark MLlib Training (`2m 01s`, `0.1344 DPU-Hours`)
![AWS Glue Training Job](../evidence/table2_glue_training_job.png)

### Serialized Spark MLlib `PipelineModel` Stages in Amazon S3
![S3 Model Artifact](../evidence/table3_s3_model_artifact.png)

### Live Amazon ECS Fargate Service & Real-Time `/health` + `/predict/` Verification
![ECS Fargate Prediction API](../evidence/table4_ecs_prediction_api.png)

### S3 Event-Driven AWS Lambda Retraining & Batch Archival
![Lambda Retraining Workflow](../evidence/table5_lambda_retraining.png)
