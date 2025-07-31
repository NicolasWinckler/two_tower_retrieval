FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG https_proxy
ARG http_proxy
ARG USER_NAME
ARG USER_ID
ARG GROUP_NAME
ARG GROUP_ID
ARG WORK_DIR

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential \
      git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# PyTorch 2.4.0 (cu121) – keep as-is
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.4.0+cu121" "torchvision==0.19.0+cu121" "torchaudio==2.4.0+cu121"

# Upgrade TorchRec + FBGEMM-GPU to versions that include the op
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torchrec==0.8.0+cu121" "fbgemm-gpu==0.8.0" "torchmetrics==1.0.3"

# Utilities (incl. FAISS CPU and click)
RUN python3 -m pip install \
  "faiss-cpu>=1.8.0" click==8.1.7 \
  pandas numpy pyarrow scikit-learn tqdm tensorboard matplotlib jupyterlab kaggle hnswlib

RUN python3 -m pip install iopath pyre-extensions

COPY ./entrypoint.sh /entrypoint.sh

RUN chmod 755 /entrypoint.sh

RUN groupadd -g "${GROUP_ID}" "${GROUP_NAME}" \
    && useradd -m -u "${USER_ID}" "${USER_NAME}" -G "${GROUP_NAME}"
#RUN mkdir -p /root/.kaggle && chmod 700 /root/.kaggle
WORKDIR "/home/${USER_NAME}"
