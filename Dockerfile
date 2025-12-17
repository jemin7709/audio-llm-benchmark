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

# Copy environment configs only (for layer caching)
COPY envs/inference/pyproject.toml envs/inference/uv.lock ./envs/inference/
COPY envs/evaluation/pyproject.toml envs/evaluation/uv.lock ./envs/evaluation/

# Sync both environments (no root sync)
RUN uv sync --project envs/inference --frozen
RUN uv sync --project envs/evaluation --frozen

# Copy entire source tree
COPY . .

# Set Python path
ENV PYTHONPATH=/app

ENTRYPOINT ["bash", "-c", "scripts/install_vllm.sh; scripts/download_datasets.sh; exec \"$@\"", "bash"]
CMD ["bash"]