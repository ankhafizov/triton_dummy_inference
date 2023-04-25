FROM nvcr.io/nvidia/tritonserver:22.01-pyt-python-py3
RUN /usr/bin/python3 -m pip install --upgrade pip
CMD sleep 500

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-get update
RUN apt install ffmpeg libsm6 libxext6 -y
RUN pip install opencv-python pandas pydantic Pillow psutil PyYAML tqdm matplotlib seaborn
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

COPY . /app/triton_server

WORKDIR /app/triton_server

CMD tritonserver --model-repository /app/triton_server/model_repository
