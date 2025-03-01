ARG BASE_IMAGE=anylearn/anylearn-runtime-base:pytorch2.0.1-cuda11.7-python3.11
FROM $BASE_IMAGE

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

ARG PYPI_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
ENV PIP_I=${PYPI_INDEX_URL:+"-i $PYPI_INDEX_URL"}

RUN pip install -r /tmp/requirements.txt $PIP_I

COPY . /app/

ENV GRADIO_SERVER_NAME="0.0.0.0"

EXPOSE 7860

CMD ["python", "app.py"]

