# For running the pack on x86 Linux. On Apple Silicon you do not need Docker at all --
# `pip install -r requirements.txt` is enough and the Metal kernels are much faster.
#
# Note what this image can and cannot do. The pack's quantized matmul has kernels for
# Metal and CPU only; mlx-cuda has none ("QuantizedMatmul has no CUDA implementation"),
# so this is a CPU image and a forward pass of this 27B model takes minutes. It is
# useful for checking behaviour, not for serving.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
# mlx-cpu carries libmlx.so, which the plain mlx wheel omits on this platform.
RUN pip install -r /tmp/requirements.txt "mlx-cpu==0.32.0"

WORKDIR /app
