FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR /workspace
COPY . /workspace

# 安装依赖
RUN pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
RUN pip install --no-cache-dir -r requirements.txt

# 设置启动指令
ENTRYPOINT ["python", "main.py"]