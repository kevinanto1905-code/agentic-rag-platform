import os
import shutil

import boto3

UPLOAD_DIR = "uploads"


def _bucket():
    return os.getenv("S3_BUCKET")


def save_file(file_obj, filename: str) -> None:
    """Save an upload locally, and copy it to S3 when S3_BUCKET is set."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    local_path = os.path.join(UPLOAD_DIR, filename)
    with open(local_path, "wb") as f:
        shutil.copyfileobj(file_obj, f)
    if _bucket():
        boto3.client("s3").upload_file(local_path, _bucket(), filename)


def get_local_path(filename: str) -> str:
    """Return a local path for the file, downloading it from S3 first if it's missing."""
    local_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(local_path) and _bucket():
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        boto3.client("s3").download_file(_bucket(), filename, local_path)
    return local_path