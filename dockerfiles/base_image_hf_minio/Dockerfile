FROM ubuntu:latest

USER root

RUN apt-get update && apt-get install -y git

RUN apt-get install build-essential -y

RUN apt-get update && apt-get install -y python3 && apt-get install python-is-python3

RUN apt-get update && apt-get install -y curl

RUN apt install -y python3.12-venv

RUN python -m venv .venv

RUN bash -c "source .venv/bin/activate"

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

RUN bash -c "source .venv/bin/activate && python get-pip.py"

RUN bash -c "source .venv/bin/activate && python -m pip install minio"

RUN bash -c "source .venv/bin/activate && python -m pip install transformers --root-user-action=ignore"

RUN bash -c "source .venv/bin/activate && python -m pip install datasets --root-user-action=ignore"

RUN bash -c "source .venv/bin/activate && python -m pip install huggingface_hub --root-user-action=ignore"

RUN export PATH=$PATH:.

RUN export PATH=$PATH:/

ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:.:/

COPY ./imagenet_download.sh .

COPY ./imagenet_download.py .

RUN chmod -R 777 ./imagenet_download.sh

RUN chmod -R 777 ./imagenet_download.py

ENV HF_DATASETS_CACHE="/data"

ENTRYPOINT [ "./imagenet_download.sh"]