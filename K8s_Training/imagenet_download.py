# file_uploader.py MinIO Python SDK example
from minio import Minio
from minio.error import S3Error
import urllib3
import datasets
from datasets import load_dataset, DownloadConfig, Dataset
from huggingface_hub import login, logout
import os
from pathlib import Path
import s3fs

def main():
    dataset_name = "imagenet-1k"
    dataset_dir = "/data_local/imagenet_downloader"

    datasets.config.DOWNLOADED_DATASETS_PATH = Path(dataset_dir)
    datasets.config.HF_DATASETS_CACHE = Path(dataset_dir)
    datasets.enable_progress_bars()
    os.environ["HF_DATASETS_CACHE"] = dataset_dir
    split = "train"
    access_key = os.environ["MINIO_ACCESS_KEY"]
    secret_key = os.environ["MINIO_SECRET_KEY"]
    url = os.environ.get("MINIO_URL", "http://minio.minio.svc.cluster.local:9000")
    dataset_path=f"hugging-face-bucket-imagenet-1k-{split}"
    S3_URI = f"s3://{dataset_path}"

    s3_storage_options = {
                "key" : access_key,
                "secret" : secret_key,
                "endpoint_url" : url,
                "client_kwargs" : {"verify" : False},
            }

    login(
                token=os.environ['HF_TOKEN']
            )
    dataset = load_dataset( 
                            path=dataset_name,
                            split=split, 
                            streaming=False, 
                            token=os.environ['HF_TOKEN'], 
                            trust_remote_code=True,
                           )
    logout()

    dataset.save_to_disk(dataset_path=S3_URI,storage_options=s3_storage_options)
    print(f"Dataset {dataset_name} split {split} saved to {S3_URI}")
if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)