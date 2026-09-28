# ---------------------------------------------------------------
# Hugging Face Space - FaceFusion 3.9.0 (GPU - CUDA 12.4 + cuDNN)
# Kod menbe: github.com/eliko0102/facefusion (master)
# ---------------------------------------------------------------
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    FACEFUSION_VERSION=3.9.0 \
    FACEFUSION_DIR=/facefusion \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Python 3.12 + pip + sistem paketleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv \
    git curl wget ca-certificates xz-utils \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && python -m ensurepip --upgrade \
    && python -m pip install --upgrade pip

# FFMPEG 8.1 statik build
RUN curl -L --fail -o /tmp/ffmpeg.tar.xz \
      https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz \
    && test $(stat -c%s /tmp/ffmpeg.tar.xz) -gt 10000000 \
    && mkdir -p /opt/ffmpeg \
    && tar -xJf /tmp/ffmpeg.tar.xz -C /opt/ffmpeg --strip-components=1 \
    && ln -sf /opt/ffmpeg/bin/ffmpeg /usr/bin/ffmpeg \
    && ln -sf /opt/ffmpeg/bin/ffprobe /usr/bin/ffprobe \
    && ln -sf /opt/ffmpeg/bin/ffmpeg /usr/local/bin/ffmpeg \
    && ln -sf /opt/ffmpeg/bin/ffprobe /usr/local/bin/ffprobe \
    && rm -f /tmp/ffmpeg.tar.xz \
    && ffmpeg -version | head -n 1

# ---------------------------------------------------------------
# FaceFusion - GitHub-dan (Drive ZIP yerine)
# diffusion_swapper processor da bu repodadir (3d1389e)
# ---------------------------------------------------------------
WORKDIR /facefusion
RUN git clone --depth 1 --branch master https://github.com/eliko0102/facefusion.git /facefusion \
    && python install.py cuda@12 --skip-conda \
    && rm -rf /facefusion/.git /root/.cache/pip

# ---------------------------------------------------------------
# diffusion_swapper modelleri (SD 1.5 inpainting, ONNX)
# QEYD: ONNX external data -> "model.onnx_data" unet .onnx-un
# YANINDA bu adla qalmalidir, adini deyismek olmaz.
# ---------------------------------------------------------------
RUN mkdir -p /facefusion/.assets/models/sd_1_5_inpainting \
    && cd /facefusion/.assets/models/sd_1_5_inpainting \
    && curl -L --fail --retry 3 --retry-delay 3 \
        https://huggingface.co/modularai/stable-diffusion-1.5-onnx/resolve/main/unet/model.onnx \
        -o sd_1_5_inpainting_unet.onnx \
    && curl -L --fail --retry 3 --retry-delay 3 \
        https://huggingface.co/modularai/stable-diffusion-1.5-onnx/resolve/main/unet/model.onnx_data \
        -o model.onnx_data \
    && curl -L --fail --retry 3 --retry-delay 3 \
        https://huggingface.co/modularai/stable-diffusion-1.5-onnx/resolve/main/vae_encoder/model.onnx \
        -o sd_1_5_inpainting_vae_encoder.onnx \
    && curl -L --fail --retry 3 --retry-delay 3 \
        https://huggingface.co/modularai/stable-diffusion-1.5-onnx/resolve/main/vae_decoder/model.onnx \
        -o sd_1_5_inpainting_vae_decoder.onnx \
    && printf '5b35084c' > sd_1_5_inpainting_unet.onnx.hash \
    && printf 'b07d38c2' > sd_1_5_inpainting_vae_encoder.onnx.hash \
    && printf '096d4fdc' > sd_1_5_inpainting_vae_decoder.onnx.hash \
    && test $(stat -c%s sd_1_5_inpainting_unet.onnx) -eq 1206595 \
    && test $(stat -c%s model.onnx_data) -eq 3438083840 \
    && test $(stat -c%s sd_1_5_inpainting_vae_encoder.onnx) -eq 136760313 \
    && test $(stat -c%s sd_1_5_inpainting_vae_decoder.onnx) -eq 198078206

# Tetbiq qati
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

COPY app.py facefusionswapper.py /app/

EXPOSE 7860
CMD ["python", "/app/app.py"]
