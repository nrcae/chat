FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for llama-cpp-python compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    git \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Set environment variables for llama-cpp-python compilation on ARM64
ENV CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_NATIVE=OFF -DGGML_NATIVE=OFF"
ENV FORCE_CMAKE=1

RUN pip install --no-cache-dir -r requirements.txt

COPY chat.py .
RUN mkdir -p /app/model

# Set environment variables with defaults
ENV MODEL_PATH=/app/model/model.gguf
ENV N_GPU_LAYERS=0
ENV N_THREADS=4
ENV N_CTX=2048
ENV MAX_TOKENS_RESPONSE=350
ENV CHAT_FORMAT=mistral-instruct

# Run the chatbot
CMD ["python", "chat.py", \
     "--model-path", "/app/model/model.gguf", \
     "--n-gpu-layers", "0", \
     "--n-threads", "4", \
     "--n-ctx", "2048", \
     "--max-tokens-response", "350", \
     "--chat-format", "mistral-instruct"]