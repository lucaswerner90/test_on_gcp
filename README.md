# Cats vs Dogs: Enterprise ML Pipeline

Welcome to the internal engineering manifesto for our Google MLOps Level 2 Maturity "Cats vs Dogs" project.

## 1. System Architecture (The ML Factory)

Our object classification pipeline leverages a heavily decoupled architecture that separates high-throughput data processing from intensive matrix multiplications:

- **`tf.data` (CPU loading):** Streams and prefetches massive image pools directly from Google Cloud Storage into local RAM.
- **TFDV (Data Gatekeeper):** Validates raw image bytes prior to training to ensure clean data input.
- **Google Vizier (Parallel Hyperparameter Tuning):** Dynamically hunts for the best learning rates and convolutional filters.
- **Keras 3 / JAX (GPU Training):** Fully delegates graph execution via XLA (Accelerated Linear Algebra) for unparalleled GPU utilization.
- **Golden Dataset (Behavioral Gatekeeper):** Curated edge cases designed to stress test Challenger models against the production Champion.
- **Vertex AI Model Registry:** Manages the deployment lifecycle, supercharged with Integrated Gradients (XAI) for visual decision heatmaps.

We maintain **TWO** distinct orchestrations:
1. **Training & Tuning Pipeline:** An active learning loop scaling from data preprocessing and hyperparameter tuning to evaluation and model deployment.
2. **`batch_scoring_pipeline`:** A massive-scale asynchronous inference pipeline designed to score 100k+ unlabelled images rapidly using our latest Champion model.

## 2. Advanced Enterprise Features (MLOps Level 2)

Our most recent infrastructure upgrade grants the following capabilities:

### Data Validation
We use **TensorFlow Data Validation (TFDV)** to act as an aggressive gatekeeper to the GPU cluster. TFDV generates statistics on input batches to assert that images strictly conform to RGB channels and standard dimension sizes. If anomalies are detected (e.g., corrupted bytes, grayscale encodings), the pipeline halts immediately, preventing garbage data from wasting expensive GPU time.

### Hyperparameter Optimization (HPO)
Instead of static runs, the pipeline utilizes **Google Vizier**. Vizier dynamically hunts for the optimal hyperparameters in parallel using Bayesian optimization. It searches learning rates (e.g., 0.0001 to 0.01) and convolutional filters (e.g., 32 vs. 64) concurrently, ensuring maximum validation accuracy across the defined max trials.

### Explainable AI (XAI)
To demystify deep learning decisions, the Vertex Registry utilizes **Integrated Gradients**. This feature allows Vertex AI to generate pixel-level heatmaps explaining exactly *why* the JAX model predicted Cat or Dog, pinpointing specific morphological features (like ear shapes or fur patterns) that swayed the sigmoid output.

<<<<<<< HEAD
=======
### Advanced Binary Metrics Evaluation
Instead of solely tracking Accuracy (which can obscure dataset imbalances), the training loop natively measures and exports several advanced metrics to the Vertex AI UI for detailed tradeoff analysis:

- **Precision:** Answers *"Out of all the images the model predicted as a 'Dog', how many were actually 'Dogs'?"* High precision means you can trust the model when it claims a positive match, as it minimizes False Positives.
- **Recall (Sensitivity):** Answers *"Out of all the actual 'Dogs' in the dataset, how many did the model successfully find?"* High recall means the model is aggressively searching and catching all positive cases, minimizing False Negatives.
- **AUC (Area Under the ROC Curve):** This is the gold standard metric for evaluating binary architecture. It measures how well the model separates the Cats from the Dogs across *all* possible confidence thresholds (not just the strict `0.5` cutoff) and is entirely immune to extreme class imbalances.

Because the model utilizes a Keras multi-output dictionary format, these metrics are perfectly bound to the `label` tensor and render beautiful, simultaneous training curves directly into TensorBoard!

>>>>>>> 02384d4 (Implement a complete MLOps pipeline for a Cats vs Dogs classifier using DVC, Docker, GitHub Actions, and Keras 3/JAX on Vertex AI.)
## 3. Day 0 Infrastructure Setup

Before any pipeline code executes natively, the Google Cloud environment must be initialized. Run these commands from an authenticated terminal to establish the infrastructure.

**1. Set the active GCP project:**
```bash
gcloud config set project YOUR_PROJECT_ID
```

**2. Enable required Google Cloud APIs:**
```bash
gcloud services enable compute.googleapis.com \
                       storage.googleapis.com \
                       aiplatform.googleapis.com
```

**3. Create the centralized Cloud Storage bucket:**
```bash
gcloud storage buckets create gs://cats-dogs-mlops-artifacts --location=us-central1
```

**4. Create the Service Account and map core IAM roles:**
```bash
# Create the service account
gcloud iam service-accounts create pipeline-runner \
    --description="Service Account for Vertex AI KFP Pipelines" \
    --display-name="Vertex Pipeline Runner"

# Assign Vertex AI User role
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:pipeline-runner@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

# Assign Storage Admin role
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:pipeline-runner@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"
```

## 4. Local Setup & Version Control

Configure your daily-driver hardware to interact correctly with the new GCP architecture and the JAX-backed Keras environment.

**1. Authenticate locally:**
```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

**2. Create Python environment & install requirements:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Initialize DVC & link remote storage:**
```bash
dvc init
dvc remote add -d gcs-remote gs://cats-dogs-mlops-artifacts/dvc-store
```

<<<<<<< HEAD
=======
**4. Download the Dataset:**
In order for the project to work locally, you need to download and extract the dataset. Execute the bash script which uses the Kaggle API to automatically pull it into the `data/` directory:
```bash
./download_data.sh
```

>>>>>>> 02384d4 (Implement a complete MLOps pipeline for a Cats vs Dogs classifier using DVC, Docker, GitHub Actions, and Keras 3/JAX on Vertex AI.)
## 5. Quality Assurance & The Data Flywheel

The **Golden Dataset** is an immutable test set composed entirely of edge cases. We actively hunt down:
- **Spurious Correlations (The "Clever Hans" Effect):** Cats playing outside in green grass, dogs sleeping indoors on couches.
- **Biological Lookalikes:** Maine Coons (dog-like cats) and Pomeranians (cat-like dogs).
- **Environmental Nightmares:** Severe occlusions, bad lighting, and heavy motion blur.

### The Automated Active Learning Loop
<<<<<<< HEAD
Our Data Flywheel automatically traps the model's blind spots. The automated active learning loop via `src/mine_hard_negatives.py` continually iterates over unlabelled image dumps, flagging inferences where the confidence falls perfectly between `0.40 - 0.60`. These high-entropy anomalies are extracted, manually reviewed, and continuously tracked via DVC for the next CI/CD run, ensuring the model's domain authority monotonically increases.
=======
Our Data Flywheel automatically traps the model's blind spots. The automated active learning loop is built directly into our Kubeflow Pipeline (`src/pipeline.py` via the `mine_hard_negatives_op` component). It continually iterates over unlabelled image dumps during pipeline execution, flagging inferences where the confidence falls perfectly between `0.40 - 0.60`. These high-entropy anomalies are copied to a review bucket, manually reviewed, and continuously tracked via DVC for the next CI/CD run, ensuring the model's domain authority monotonically increases.
>>>>>>> 02384d4 (Implement a complete MLOps pipeline for a Cats vs Dogs classifier using DVC, Docker, GitHub Actions, and Keras 3/JAX on Vertex AI.)

## 6. Local Pre-Flight Execution (Testing Without the Cloud)

To prevent spending money on GCP Vertex AI compute prematurely, developers must test the entire ML system locally on their machines before any cloud submission.

### 1. Local Unit & Invariance Testing
The first step is always running the local test suite to verify tensor shapes and ensure the fundamental model graph integrity holds under JAX compilation.
```bash
pytest tests/
```

### 2. Local Model Training
Run the core Keras 3 / JAX training script locally by passing a local directory path instead of a `gs://` bucket path. This delegates the intense matrix multiplications to your local CPU (or a local GPU if properly configured).

**Tip for faster iteration:** You can append the `--images-per-class N` argument to dynamically instantly build a perfectly balanced micro-subset of the data using zero-cost symlinks, rather than training on all 25,000 images natively!
```bash
python src/task.py --data-dir data/PetImages/ --learning-rate 0.001 --conv-filters 32 --images-per-class 50 --epochs 5
```

### 3. Pipeline Graph Compilation
We do not execute Kubeflow Pipeline steps natively on our local machines (this includes the Hard Negative Mining, which runs exclusively as a pipeline component). Instead, we compile the `src/pipeline.py` script locally to catch any DAG (Directed Acyclic Graph) structural errors, missing input artifacts, or Python syntax issues. If the process successfully generates a `pipeline.json` file, the graph is structurally sound and ready for Vertex AI.
```bash
python src/pipeline.py --compile-only
```

## 7. Day-to-Day Workflow

Daily operations revolve around launching remote pipelines and verifying live inferences.

**Step 1: Run Local Code Tests:**
```bash
pytest tests/
```

**Step 2: Trigger the Vertex Pipeline:**
```bash
python src/pipeline.py \
    --project-id YOUR_PROJECT_ID \
    --region us-central1 \
    --data-dir gs://cats-dogs-mlops-artifacts/dataset
```

**Step 3: Ping the Live Endpoint:**
```bash
python client.py \
    --project-id YOUR_PROJECT_ID \
    --region us-central1 \
    --endpoint-id YOUR_ENDPOINT_ID \
    --image-path tests/sample_dog.jpg
```

## 8. The Survival Guide (Troubleshooting)

When deep learning systems fail securely in the cloud, finding the exact error is half the battle.

### Vertex AI Pipeline Fails Immediately
If your component instantly enters a failed state (usually within 3-5 seconds of dispatch), it is almost exclusively an IAM or base-image boot error.
*Fix:* Ensure the `pipeline-runner` Service Account was correctly attached during submission and has the `roles/storage.admin` and `roles/aiplatform.user` IAM roles assigned.

### My model is training on CPU instead of GPU locally
JAX operates with a silent fallback mode—if it cannot find CUDA/CUDNN, it will silently map matrices to the CPU.
*Verification:* Execute `python -c "import jax; print(jax.devices())"`.
*Fix:* If the output lists `CpuDevice`, confirm your system has `CUDA>=12` installed. You may need to pip install explicit wheel variants for XLA via the official JAX CUDA releases.

### XLA Compilation Errors
Because Keras 3 routes the background graph through XLA on the JAX backend, dynamic tensor shapes are fundamentally incompatible.
*Fix:* If you see XLA recompilation warnings or outright crashes on batch iteration, you are likely passing variably sized images through the pipeline. Ensure your inputs have a strict bounds check so XLA can confidently allocate static GPU RAM.
