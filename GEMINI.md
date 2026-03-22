# Gemini / AI Context Guide: Cats vs Dogs MLOps

This file serves as the system instruction and architectural context guide for AI assistants (like Gemini) interacting with this codebase.

## 🏗 System Architecture & Tooling
- **Orchestration**: Kubeflow Pipelines (KFP > 2.0) compiling to Google Cloud Vertex AI.
- **Deep Learning Framework**: Keras 3 strictly utilizing the **JAX** backend (`os.environ["KERAS_BACKEND"] = "jax"`).
- **Core Script**: `src/task.py` executes the single-node GPU matrix math. `src/pipeline.py` defines the macro DAG orchestration.
- **Components Location**: To adhere to enterprise standards, pipeline steps are modularized inside `src/components/` (`evaluate.py`, `review.py`, `validation.py`).

## 📂 Key Project Files
- **`src/task.py`**: The core GPU training script. It handles the Kaggle dataset corruption cleanup, creates fast-iteration datasets via symlinks (`--images-per-class`), and compiles the dynamically generated Keras 3 MIMO model (tracking `AUC`, `Precision`, `Recall`).
- **`src/pipeline.py`**: The Kubeflow (KFP) execution DAG. It orchestrates Google Vizier Hyperparameter Tuning, Model Registry deployment, and XAI Explanations using Integrated Gradients mapped to `images` and `label` tensors.
- **`src/components/validation.py`**: Uses TensorFlow Data Validation (TFDV) to aggressively reject non-RGB or structurally corrupted batches before wasting GPU compute.
- **`src/components/evaluate.py`**: Extracts dictionary metrics post-training to decide if the new model structurally beats the existing Champion model on the Golden Dataset.
- **`src/components/review.py`**: The Hard Negative Mining component (`mine_hard_negatives_op`). Autonomously scans unlabelled cloud buckets, flags 0.40-0.60 confidence inferences, and exports them directly back to GCS (bypassing the lack of Git SSH keys inside Vertex containers).
- **`.github/workflows/ml-pipeline.yml`**: The CI/CD workflow. Secured via Workload Identity Federation and strictly uses `--compile-only` to validate DAG integrity on PRs without spending Cloud credits.
- **`download_data.sh`**: A quick-start Kaggle API script essential for setting up the local `data/PetImages/` workspace.

## 🧠 Architectural Paradigms & Constraints

### 1. Ephemeral Cloud Containers (Hard Negative Mining)
Because Vertex AI containers spin up and die dynamically, they do not possess Git SSH keys. Therefore, the "Hard Negative Mining" active learning loop (`mine_hard_negatives_op`) cannot use `git commit` or `dvc push`. Instead, it identifies confusing inferences (confidence between 0.40 - 0.60) and writes them directly to a Google Cloud Storage (GCS) review bucket.

### 2. Multi-Input Multi-Output (MIMO) Dictionaries
The Keras Sequential API was completely refactored into the Functional API to support explicit dictionary naming.
- **Inputs**: Expects `{"images": x}`
- **Outputs**: Expects `{"label": y}`
- **Implication**: Any script running `tf.data` (e.g., `model.fit`, `model.evaluate`, `model.predict`) absolutely MUST map the generator tuples into dictionaries before passing them to Keras 3.

### 3. Advanced Metric Tracking
The training script does not just measure Accuracy. It actively exports `Precision`, `Recall`, and `AUC` (Area Under the ROC Curve) explicitly mapped to the `label` output. These metrics heavily safeguard against disguised class imbalances.

### 4. Continuous Integration & Workload Identity
GitHub Actions (`.github/workflows/ml-pipeline.yml`) uses strictly keyless authentication via Workload Identity Federation. The `gcloud auth` action is strictly condition-bound to `lucaswerner90/test_on_gcp`. 
- The workflow runs a DAG compilation check on every push using `python src/pipeline.py --compile-only`.
- The actual Vertex AI pipeline submission is hidden behind a manual `workflow_dispatch` gate.

## ⚠️ Known Quirks & Specific Workarounds

- **Kaggle Dataset Corruption**: The Microsoft Cats vs Dogs dataset is famously riddled with 0-byte images and fake JPEGs (e.g., GIFs renamed to `.jpg`). Keras' `decode_image` will crash dynamically if these are read. `src/task.py` possesses a critical `clean_dataset()` function that scans and deletes any image missing a valid `JFIF` binary header. Do not remove this.
- **Fast Local Iteration**: Because standard testing on 25k images is too slow, `src/task.py` implements a custom `--images-per-class` parameter. It mathematically isolates a tiny micro-subset of the data using zero-cost filesystem `symlinks`. Always use this for local debugging (along with `--epochs`!).

## 🧑‍💻 Primary Development Commands
* **Local Fast-Test**: `python src/task.py --data-dir data/PetImages/ --learning-rate 0.001 --conv-filters 32 64 --images-per-class 50 --epochs 5`
* **Local DAG Compilation**: `python src/pipeline.py --compile-only`
* **Download Data**: `source .venv/bin/activate && ./download_data.sh`
