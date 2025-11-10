# uv add -r extra/vllm/requirements/build.txt
# uv add -r extra/vllm/requirements/cuda.txt
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f/vllm-0.9.2-cp38-abi3-manylinux1_x86_64.whl
VLLM_USE_PRECOMPILED=1 uv add --editable extra/vllm/ -v --no-build-isolation