# Cloud-Native Credit Card Fraud Detection & Retraining Pipeline on AWS

Serverless and containerized AWS machine learning platform migrating an on-premises PySpark credit card fraud workflow (`284,807` transactions, `0.17%` positive fraud class) to **AWS Glue 4.0** distributed Spark MLlib training, **S3-triggered AWS Lambda** automated retraining & compliance archival, and an **Amazon ECS Fargate** FastAPI + PySpark real-time inference microservice.

---

## Key Performance & Engineering Highlights

- **`10×` Distributed Spark Training Speedup (`>20m` → `121s`)**: Repartitioning and in-memory caching (`.repartition(16).cache()`) with single-projection `DoubleType` casting and dynamic class-imbalance weighting across `4 × G.1X` AWS Glue workers (`0.1344 DPU-hours`).
- **Zero-Idle-Cost Event-Driven Retraining**: Uploading a new transaction batch (e.g., [`data/incoming_transactions_batch3.csv`](data/incoming_transactions_batch3.csv)) to `s3://<bucket>/data/incoming/` triggers `refresh_function.lambda_handler`, archiving the batch with an immutable timestamp and launching an asynchronous AWS Glue retraining run.
- **Credential-Less Containerized Inference**: FastAPI microservice (`fraud_detection_api/`) packaged with Python 3.10 + OpenJDK 17 + PySpark 3.3 dynamically downloads `model/fraud_detection_model_latest/` from S3 via its attached ECS IAM Task Role and serves Pydantic-validated `POST /predict/` requests.

See [`docs/ARCHITECTURE_REPORT.md`](docs/ARCHITECTURE_REPORT.md) for full service rationale, IAM role architecture, and verified AWS deployment telemetry.

---

## Repository Layout

```text
aws-serverless-fraud-ml-pipeline/
├── fraud_detection_pipeline/
│   ├── fraud_detector_model_trainer.py  # AWS Glue 4.0 distributed Spark MLlib trainer
│   └── Pipfile
├── refresh_function/
│   ├── refresh_function.py              # S3 ObjectCreated -> archive + Glue trigger Lambda
│   └── Pipfile
├── fraud_detection_api/
│   ├── main.py                          # FastAPI + PySpark real-time inference microservice
│   ├── Dockerfile                       # Python 3.10 + headless OpenJDK 17 container image
│   ├── requirements.txt
│   └── model/fraud_detection_model_latest/ # Serialized Spark PipelineModel artifact
├── ecs/
│   └── fraud-detection-task-def.json    # AWS Fargate (1024 CPU / 2048 MB) task definition
├── data/
│   └── incoming_transactions_batch3.csv # Sample 30-feature streaming batch (Time, V1..V28, Amount)
├── docs/
│   └── ARCHITECTURE_REPORT.md           # Full cloud architecture, IAM matrix & runbook
└── evidence/                            # Verified Glue, S3, ECS Fargate, Lambda & IAM captures
```

---

## Local Container & API Quick Start

```bash
# Build and run the FastAPI + PySpark inference microservice locally
docker build -t fraud-detection-api ./fraud_detection_api
docker run -p 8000:8000 fraud-detection-api

# Health & readiness check
curl http://localhost:8000/health

# Score a real-time credit card transaction
curl -X POST http://localhost:8000/predict/   -H "Content-Type: application/json"   -d '{"Time": 406.0, "V1": -2.3122, "V2": 1.9519, "V3": -1.6098, "V4": 3.9979, "V5": -0.5221, "V6": -1.4265, "V7": -2.5373, "V8": 1.3916, "V9": -2.7700, "V10": -2.7722, "V11": 3.2020, "V12": -2.8999, "V13": -0.5952, "V14": -4.2892, "V15": 0.3897, "V16": -1.1407, "V17": -2.8300, "V18": -0.0168, "V19": 0.4169, "V20": 0.1269, "V21": 0.5172, "V22": -0.0350, "V23": -0.4652, "V24": 0.3201, "V25": 0.0445, "V26": 0.1778, "V27": 0.2611, "V28": -0.1432, "Amount": 0.0}'
```

## License

MIT License — see [LICENSE](LICENSE).
