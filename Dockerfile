FROM nvidia/cuda:12.9.1-devel-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:0.8.19 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirror.kakao.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirror.kakao.com/ubuntu|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates curl wget git ffmpeg unzip p7zip-full neovim openjdk-11-jre-headless fonts-noto fonts-noto-cjk fonts-noto-color-emoji\
    libnvtoolsext1 cuda-nvtx-12-9 && \
    rm -rf /var/lib/apt/lists/* && apt-get clean

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
# COPY extra/vllm/ ./extra/vllm/
# ENV VLLM_USE_PRECOMPILED=1 \
# VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f/vllm-0.9.2-cp38-abi3-manylinux1_x86_64.whl
RUN uv sync --frozen
RUN uv sync --project envs/inference --frozen
RUN uv sync --project envs/evaluation --frozen

COPY . .

ENTRYPOINT ["bash", "-c", "scripts/install_vllm.sh; scripts/download_datasets.sh; exec \"$@\"", "bash"]
CMD ["bash"]