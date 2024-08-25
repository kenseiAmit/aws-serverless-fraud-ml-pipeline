import os
import datetime
import urllib.parse
import json
import boto3

# Initiate the appropriate Cloud SDK Clients
s3_client = boto3.client("s3")
glue_client = boto3.client("glue")

GLUE_JOB_NAME = os.environ.get("GLUE_JOB_NAME", "fraud-detection-model-trainer")
DEFAULT_S3_BUCKET = os.environ.get("S3_BUCKET", "credit-card-fraud-pipeline-117913377161")

def trigger_retraining_job(bucket_name: str, object_key: str):
    """
    Kicks off the AWS Glue model training job with the specified S3 data location.
    """
    input_s3_path = f"s3://{bucket_name}/{object_key}"
    print(f"Triggering AWS Glue retraining job '{GLUE_JOB_NAME}' on data: {input_s3_path}")
    
    response = glue_client.start_job_run(
        JobName=GLUE_JOB_NAME,
        Arguments={
            "--S3_INPUT_PATH": input_s3_path,
            "--S3_BUCKET": bucket_name
        }
    )
    job_run_id = response.get("JobRunId")
    print(f"Successfully started Glue Job Run: {job_run_id}")
    return job_run_id

def archive_processed_s3_file(bucket_name: str, object_key: str):
    """
    Moves/archives the processed CSV file to an archive directory in the Cloud Storage location.
    """
    file_name = os.path.basename(object_key)
    base_name, ext = os.path.splitext(file_name)
    current_date = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_key = f"data/archive/{base_name}_retrain_{current_date}{ext}"
    
    print(f"Archiving s3://{bucket_name}/{object_key} to s3://{bucket_name}/{archive_key}")
    
    # Copy object to archive location
    s3_client.copy_object(
        Bucket=bucket_name,
        CopySource={"Bucket": bucket_name, "Key": object_key},
        Key=archive_key
    )
    
    # Delete original file to prevent duplicate triggers
    s3_client.delete_object(Bucket=bucket_name, Key=object_key)
    print(f"Successfully moved processed file to archive: {archive_key}")
    return archive_key

def lambda_handler(event, context):
    """
    AWS Lambda entry point triggered by Amazon S3 ObjectCreated events.
    """
    print("Received S3 event: " + json.dumps(event))
    results = []

    # Process all S3 event records
    records = event.get("Records", [])
    if not records:
        # Fallback or manual test payload
        bucket = event.get("bucket", DEFAULT_S3_BUCKET)
        key = event.get("key", "data/raw/creditcard.csv")
        records = [{"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}]

    for record in records:
        s3_info = record.get("s3", {})
        bucket_name = s3_info.get("bucket", {}).get("name")
        raw_key = s3_info.get("object", {}).get("key", "")
        object_key = urllib.parse.unquote_plus(raw_key)

        print(f"Evaluating S3 object: bucket={bucket_name}, key={object_key}")

        # Check if the object is a new CSV data file in data/raw/ and not already in archive
        if object_key.endswith(".csv") and "archive" not in object_key and "_retrain_" not in object_key:
            try:
                # 1. Archive the processed file in the S3 bucket first
                archive_key = archive_processed_s3_file(bucket_name, object_key)

                # 2. Kick off the AWS Glue model training job with the permanent archived dataset
                job_run_id = trigger_retraining_job(bucket_name, archive_key)
                
                results.append({
                    "status": "SUCCESS",
                    "job_run_id": job_run_id,
                    "input_key": object_key,
                    "archived_key": archive_key
                })
            except Exception as e:
                print(f"Error processing {object_key}: {e}")
                results.append({
                    "status": "FAILED",
                    "input_key": object_key,
                    "error": str(e)
                })
        else:
            print(f"Skipping non-target or already processed key: {object_key}")

    return {
        "statusCode": 200,
        "body": json.dumps({"message": "Retraining workflow executed", "results": results})
    }

def watch_directory_and_retrain(bucket_name=DEFAULT_S3_BUCKET, prefix="data/raw/"):
    """
    Polls the cloud storage bucket location for new CSV files (for standalone/local testing).
    """
    print(f"Watching S3 bucket '{bucket_name}', prefix '{prefix}' for retraining triggers...")
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".csv") and "archive" not in key and "_retrain_" not in key:
                print(f"New data detected in S3: s3://{bucket_name}/{key}")
                job_run_id = trigger_retraining_job(bucket_name, key)
                archive_key = archive_processed_s3_file(bucket_name, key)
                print(f"Retraining started: JobRunId={job_run_id}, Archived to={archive_key}")
    except Exception as e:
        print(f"Error watching S3 storage: {e}")

def main():
    watch_directory_and_retrain(DEFAULT_S3_BUCKET, "data/raw/")

if __name__ == "__main__":
    main()
