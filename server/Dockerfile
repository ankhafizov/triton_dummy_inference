FROM nvcr.io/nvidia/tritonserver:22.01-pyt-python-py3
RUN /usr/bin/python3 -m pip install --upgrade pip
CMD sleep 500

COPY . /app/triton_server

WORKDIR /app/triton_server
RUN pip install --no-cache-dir -r requirements.txt

CMD tritonserver --model-repository /app/triton_server/model_repository
