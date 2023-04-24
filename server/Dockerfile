FROM nvcr.io/nvidia/tritonserver:22.01-pyt-python-py3 as base

COPY . .

WORKDIR /app/triton_server


CMD tritonserver --model-repository inference_triton/model_repository
