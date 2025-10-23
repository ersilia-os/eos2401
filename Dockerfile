FROM bentoml/model-server:0.11.0-py312
MAINTAINER ersilia

RUN pip install safe-mol==0.1.14


WORKDIR /repo
COPY . /repo
