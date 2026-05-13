"""
Deploy the credit risk API to AWS SageMaker.

What this script does, step by step:
  1. Packages your model artifacts into a tar.gz and uploads to S3
     (SageMaker can't read local files — everything goes through S3)
  2. Builds a Docker image from the Dockerfile
  3. Pushes the image to ECR (AWS's private Docker registry)
  4. Creates a SageMaker Model — tells SageMaker which image to use and
     where the model artifacts live in S3
  5. Creates an Endpoint Configuration — specifies the instance type and count
  6. Deploys an Endpoint — SageMaker starts the container, runs /ping until
     it returns 200, then starts routing traffic

After deployment you get an endpoint name you can call with boto3 or curl.

Prerequisites:
  - AWS CLI configured: `aws configure`
  - Docker running locally
  - Permissions: ECR, S3, SageMaker (attach AmazonSageMakerFullAccess in IAM)

Usage:
  python sagemaker_deploy.py --region us-east-1 --bucket your-s3-bucket
"""

import argparse
import base64
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

import boto3

MODELS_DIR = Path("../models")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_NAME = "credit-risk-api"
IMAGE_TAG = "latest"
MODEL_NAME = "credit-risk-model"
ENDPOINT_CONFIG_NAME = "credit-risk-endpoint-config"
ENDPOINT_NAME = "credit-risk-endpoint"

# ml.m5.xlarge: 4 vCPU, 16 GB RAM — enough for the MLP + DiCE reference data.
# For higher throughput, scale out with initial_instance_count > 1 rather
# than scaling up to a larger instance type.
INSTANCE_TYPE = "ml.t2.medium"
INITIAL_INSTANCE_COUNT = 1


def package_model_artifacts(s3_bucket: str, region: str) -> str:
    """
    Tars up the model directory and uploads to S3.
    SageMaker extracts this tar into /opt/ml/model/ inside the container.
    Returns the S3 URI of the uploaded tar.
    """
    print("Packaging model artifacts...")
    artifacts = [
        "preprocessor.pkl",
        "calibrator.pkl",
        "mlp_model.pth",
        "winsorize_bounds.json",
        "raw_winsorize_bounds.json",
        "training_meta.json",
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = os.path.join(tmp, "model.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tar:
            for artifact in artifacts:
                src = MODELS_DIR / artifact
                if not src.exists():
                    raise FileNotFoundError(f"Missing model artifact: {src}")
                tar.add(src, arcname=artifact)
            # Include training data for DiCE reference and Evidently baseline
            train_csv = PROJECT_ROOT / "data" / "train.csv"
            tar.add(train_csv, arcname="train.csv")

        s3_key = f"credit-risk/model.tar.gz"
        s3 = boto3.client("s3", region_name=region)
        s3.upload_file(tar_path, s3_bucket, s3_key)
        s3_uri = f"s3://{s3_bucket}/{s3_key}"
        print(f"  Uploaded to {s3_uri}")
        return s3_uri


def build_and_push_image(account_id: str, region: str) -> str:
    """
    Builds the Docker image and pushes it to ECR.
    Returns the full ECR image URI.

    ECR is just a private Docker registry hosted by AWS. You push your image
    there so SageMaker can pull it when starting your endpoint.
    """
    ecr_repo = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{IMAGE_NAME}"
    full_tag = f"{ecr_repo}:{IMAGE_TAG}"

    print("Creating ECR repository (if not exists)...")
    ecr = boto3.client("ecr", region_name=region)
    try:
        ecr.create_repository(repositoryName=IMAGE_NAME)
    except ecr.exceptions.RepositoryAlreadyExistsException:
        pass

    print("Authenticating Docker with ECR...")
    token = ecr.get_authorization_token()
    token_data = token["authorizationData"][0]
    username, password = base64.b64decode(token_data["authorizationToken"]).decode().split(":")
    registry = token_data["proxyEndpoint"]
    subprocess.run(
        ["docker", "login", "--username", username, "--password", password, registry],
        check=True, capture_output=True,
    )

    print("Building Docker image...")
    subprocess.run(
        [
            "docker", "build",
            "--platform", "linux/amd64",   # SageMaker instances are x86_64
            "--provenance=false",           # prevents OCI manifest list format that SageMaker rejects
            "-t", full_tag, ".",
        ],
        cwd=Path(__file__).parent,
        check=True,
    )

    print("Pushing image to ECR...")
    subprocess.run(["docker", "push", full_tag], check=True)

    print(f"  Image pushed: {full_tag}")
    return full_tag


def deploy(s3_bucket: str, region: str) -> str:
    """
    Creates or updates the SageMaker model and endpoint.
    Returns the endpoint name.
    """
    sts = boto3.client("sts", region_name=region)
    account_id = sts.get_caller_identity()["Account"]

    sagemaker = boto3.client("sagemaker", region_name=region)

    model_artifact_uri = package_model_artifacts(s3_bucket, region)
    image_uri = build_and_push_image(account_id, region)

    # SageMaker needs an IAM role that allows it to pull from ECR and read S3.
    # Attach: AmazonSageMakerFullAccess + AmazonS3ReadOnlyAccess to this role.
    role_arn = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"

    print("Creating SageMaker model...")
    try:
        sagemaker.delete_model(ModelName=MODEL_NAME)
    except sagemaker.exceptions.ClientError:
        pass

    sagemaker.create_model(
        ModelName=MODEL_NAME,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_artifact_uri,
            # MODEL_DIR env var tells pipeline.py where to find artifacts
            "Environment": {"MODEL_DIR": "/opt/ml/model"},
        },
        ExecutionRoleArn=role_arn,
    )

    print("Creating endpoint configuration...")
    try:
        sagemaker.delete_endpoint_config(EndpointConfigName=ENDPOINT_CONFIG_NAME)
    except sagemaker.exceptions.ClientError:
        pass

    sagemaker.create_endpoint_config(
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
        ProductionVariants=[
            {
                "VariantName": "primary",
                "ModelName": MODEL_NAME,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": INITIAL_INSTANCE_COUNT,
                "InitialVariantWeight": 1,
            }
        ],
    )

    print(f"Deploying endpoint '{ENDPOINT_NAME}' (takes ~5 minutes)...")
    needs_create = False
    try:
        existing = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = existing["EndpointStatus"]
        if status == "Failed":
            print("  Existing endpoint is Failed — deleting before recreating...")
            sagemaker.delete_endpoint(EndpointName=ENDPOINT_NAME)
            sagemaker.get_waiter("endpoint_deleted").wait(EndpointName=ENDPOINT_NAME)
            needs_create = True
        else:
            sagemaker.update_endpoint(
                EndpointName=ENDPOINT_NAME,
                EndpointConfigName=ENDPOINT_CONFIG_NAME,
            )
    except sagemaker.exceptions.ClientError as e:
        if "Could not find endpoint" in str(e):
            needs_create = True
        else:
            raise

    if needs_create:
        sagemaker.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=ENDPOINT_CONFIG_NAME,
        )

    # Wait until the endpoint is InService
    waiter = sagemaker.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=ENDPOINT_NAME, WaiterConfig={"Delay": 30, "MaxAttempts": 20})

    endpoint = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
    print(f"\nEndpoint live: {endpoint['EndpointArn']}")
    print(f"Status: {endpoint['EndpointStatus']}")
    print(f"\nTo call it:\n"
          f"  import boto3, json\n"
          f"  client = boto3.client('sagemaker-runtime', region_name='{region}')\n"
          f"  response = client.invoke_endpoint(\n"
          f"      EndpointName='{ENDPOINT_NAME}',\n"
          f"      ContentType='application/json',\n"
          f"      Body=json.dumps(your_application_dict)\n"
          f"  )\n"
          f"  print(json.loads(response['Body'].read()))")

    return ENDPOINT_NAME


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--bucket", required=True, help="S3 bucket for model artifacts")
    args = parser.parse_args()

    deploy(s3_bucket=args.bucket, region=args.region)
