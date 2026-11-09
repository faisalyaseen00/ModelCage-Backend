# ModelCage ML runner: the controlled pickle harness plus the libraries that
# real model pickles reference. Dependencies are pinned so every execution of a
# given profile sees the same interpreter and import-time syscall behaviour.
FROM python:3.12.7-slim-bookworm

# CPU-only PyTorch keeps the image free of CUDA libraries the sandbox cannot use.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1 \
    && pip install --no-cache-dir \
        numpy==2.1.3 \
        scikit-learn==1.5.2 \
        transformers==4.46.3 \
        safetensors==0.4.5 \
    && find /usr/local/lib/python3.12 -name '__pycache__' -prune -exec rm -rf {} +

# The root filesystem is read-only at runtime and the network is disabled, so
# every library cache points at the sandbox's private /tmp tmpfs and hub access
# is switched off. One thread matches the sandbox's single CPU and PID budget.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    HF_HOME=/tmp/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TORCH_HOME=/tmp/torch \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
