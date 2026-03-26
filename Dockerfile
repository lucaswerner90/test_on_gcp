# ==============================================================================
# Google Artifact Registry (GAR) Build & Push Instructions
# ==============================================================================
# 1. Build the local image:
#    docker build -t cats-dogs-jax .
#
# 2. Tag the image for GAR:
#    docker tag cats-dogs-jax us-central1-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPO_NAME/cats-dogs-jax:latest
#
# 3. Authenticate Docker with GAR (Run once per machine):
#    gcloud auth configure-docker us-central1-docker.pkg.dev
#
# 4. Push the image to GAR:
#    docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/YOUR_REPO_NAME/cats-dogs-jax:latest
# ==============================================================================

FROM python:3.12-slim

# Set the Keras 3 backend to JAX globally
ENV KERAS_BACKEND=jax
# Keep Python from buffering stdout/stderr to ensure logs flush immediately to GCP Cloud Logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# CRITICAL MLOps TRICK: Strip standard tensorflow from requirements to prevent massive GPU binary downloads.
# Install tensorflow-cpu for CPU-bound tf.data pipelines, and jax[cuda12] for GPU matrix multiplication.
RUN sed -i '/^tensorflow/d' requirements.txt && \
    pip install --no-cache-dir --upgrade pip && \
    pip install "dvc[gs]" && \
    pip install --no-cache-dir tensorflow-cpu "jax[cuda12]" && \
    pip install --no-cache-dir -r requirements.txt

# Do NOT copy the raw data/ directory! Data rehydration will organically pull it via DVC at runtime.
COPY .dvc/ /app/.dvc/
COPY *.dvc /app/

COPY src/ src/
COPY tests/ tests/

ENTRYPOINT ["python", "src/task.py"]