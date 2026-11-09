FROM ghcr.io/mlflow/mlflow:latest
EXPOSE 9091

RUN pip install psycopg2-binary

CMD ["/bin/sh","-c","python -m mlflow server --host 0.0.0.0 --port 9091 --serve-artifacts --artifacts-destination /mlflow_artifacts --backend-store-uri $MLFLOW_BACKEND_STORE_URI"]