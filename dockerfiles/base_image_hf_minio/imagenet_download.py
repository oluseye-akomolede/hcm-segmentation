# file_uploader.py MinIO Python SDK example
from minio import Minio
from minio.error import S3Error
import urllib3
import datasets
from datasets import load_dataset, DownloadConfig
from huggingface_hub import login, logout
import os
from pathlib import Path

def main():
    dataset_name = "ILSVRC/imagenet-1k"
    dataset_split = "train"
    dataset_dir = "/data"

    datasets.config.DOWNLOADED_DATASETS_PATH = Path(dataset_dir)
    datasets.config.HF_DATASETS_CACHE = Path(dataset_dir)
    datasets.enable_progress_bars()
    os.environ["HF_DATASETS_CACHE"] = dataset_dir
    d_cfg = DownloadConfig(
            resume_download = True,
            max_retries= 500
            )
    login(
                token=os.environ['HF_TOKEN']
            )
    dataset = load_dataset( 
                            path=dataset_name,
                            #dataset_name=dataset_name, 
                            split='train', 
                            streaming=False, 
                            token=os.environ['HF_TOKEN'], 
                            trust_remote_code=True,
                            #download_config=d_cfg,
                           )
    logout()
    # Create a client with the MinIO server playground, its access key
    # and secret key.
    httpClient = urllib3.PoolManager(
                cert_reqs="CERT_NONE")
    client = Minio(os.environ.get("MINIO_ENDPOINT", "minio.minio.svc.cluster.local:9000"),
        access_key=os.environ["MINIO_ACCESS_KEY"],
        secret_key=os.environ["MINIO_SECRET_KEY"],
        secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
        http_client=httpClient,
    )
    # The destination bucket and filename on the MinIO server
    bucket_name = "hugging-face-bucket-imagenet-1k"

    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
        print("Created bucket", bucket_name)
    else:
        print("Bucket", bucket_name, "already exists")

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            minio_object_path = local_file_path.replace(dataset_dir,"")[1:]
            print(f"Uploading {local_file_path} to {bucket_name}/{minio_object_path}")
            client.fput_object(bucket_name,minio_object_path,local_file_path)


if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)