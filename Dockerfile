FROM bentoml/model-server:0.11.0-py312
MAINTAINER ersilia

RUN pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers==4.57.6 tokenizers==0.22.2 huggingface-hub==0.36.2 datasets==2.14.4 rdkit==2026.3.6 safe-mol==0.1.14


WORKDIR /repo
COPY . /repo
