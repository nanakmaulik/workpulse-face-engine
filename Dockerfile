FROM python:3.11-slim

WORKDIR /srv

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

RUN mkdir -p /srv/models

RUN curl -fL --retry 3 \
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx?download=true" \
    -o /srv/models/face_detection_yunet_2023mar.onnx

RUN curl -fL --retry 3 \
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx?download=true" \
    -o /srv/models/face_recognition_sface_2021dec.onnx

ENV MODEL_DIR=/srv/models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
