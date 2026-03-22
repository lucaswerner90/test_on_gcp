import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

from kfp import dsl
from kfp.v2.dsl import (
    component,
    Input,
    Model,
    Output,
    Metrics,
    ClassificationMetrics,
    Condition
)
from google_cloud_pipeline_components.v1.hyperparameter_tuning_job import HyperparameterTuningJobRunOp
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp as ModelImportOp


# Component 1: Data Validation
@component(
    base_image="python:3.10",
    packages_to_install=["tensorflow", "tensorflow-data-validation", "pandas"]
)
def data_validation_op(unlabelled_data_gcs_path: str):
    import tensorflow as tf
    import tensorflow_data_validation as tfdv
    import pandas as pd
    import logging
    
    logging.info(f"Validating imagery at {unlabelled_data_gcs_path}...")
    
    file_paths = tf.io.gfile.glob(f"{unlabelled_data_gcs_path}/*/*")
    if not file_paths:
        file_paths = tf.io.gfile.glob(f"{unlabelled_data_gcs_path}/*")
    
    records = []
    # Asserting up to 100 images for speed in validation gate
    for path in file_paths[:100]:
        img_raw = tf.io.read_file(path)
        img = tf.image.decode_image(img_raw)
        shape = img.shape
        records.append({
            "channels": shape[2] if len(shape) > 2 else 1,
            "filename": path
        })
        
    df = pd.DataFrame(records)
    stats = tfdv.generate_statistics_from_dataframe(df)
    
    schema = tfdv.infer_schema(stats)
    
    # Assert RGB
    tfdv.get_feature(schema, 'channels').presence.min_fraction = 1.0
    tfdv.set_domain(schema, 'channels', tfdv.IntDomain(min=3, max=3))
    
    anomalies = tfdv.validate_statistics(statistics=stats, schema=schema)
    
    if anomalies.anomaly_info:
        for feature_name, anomaly_info in anomalies.anomaly_info.items():
            logging.error(f"TFDV Anomaly in {feature_name}: {anomaly_info.description}")
        raise ValueError("Data Validation Failed: Anomalies found in image features (not RGB or standard format). Halting pipeline.")
        
    logging.info("TFDV: Data Validation passed. Images conform to RGB representations.")


# Custom evaluation component using JAX and Keras 3
@component(
    base_image="python:3.10",
    packages_to_install=[
        "keras>=3.0.0", 
        "jax[cuda12]", 
        "tensorflow", 
        "scikit-learn", 
        "google-cloud-storage"
    ]
)
def evaluate_model(
    model_dir: str,
    test_data_dir: str,
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics]
) -> float:
    import os
    os.environ["KERAS_BACKEND"] = "jax"
    
    import keras
    import tensorflow as tf
    import numpy as np
    from sklearn.metrics import confusion_matrix
    
    model_path = f"{model_dir}/model.keras" if tf.io.gfile.exists(f"{model_dir}/model.keras") else model_dir
    model = keras.saving.load_model(model_path)

    test_ds = keras.utils.image_dataset_from_directory(
        test_data_dir,
        image_size=(256, 256),
        batch_size=32,
        label_mode='binary',
        shuffle=False
    )
    
    loss, accuracy = model.evaluate(test_ds)
    metrics.log_metric("accuracy", float(accuracy))
    metrics.log_metric("loss", float(loss))
    
    predictions = model.predict(test_ds)
    y_pred = (predictions > 0.5).astype("int32").flatten()
    y_true = np.concatenate([y for x, y in test_ds], axis=0).flatten()
    
    cm = confusion_matrix(y_true, y_pred)
    classification_metrics.log_confusion_matrix(["Cat", "Dog"], cm.tolist())
    
    return float(accuracy)


# Champion vs Challenger Evaluation Component
@component(
    base_image="python:3.10",
    packages_to_install=[
        "keras>=3.0.0", 
        "jax[cuda12]", 
        "tensorflow", 
        "google-cloud-aiplatform",
        "google-cloud-storage"
    ]
)
def champion_vs_challenger(
    new_model_dir: str,
    project_id: str,
    region: str,
    golden_dataset_dir: str,
) -> bool:
    import os
    os.environ["KERAS_BACKEND"] = "jax"
    
    import logging
    import keras
    import tensorflow as tf
    from google.cloud import aiplatform

    def evaluate_accuracy(model_dir):
        model_path = f"{model_dir}/model.keras" if tf.io.gfile.exists(f"{model_dir}/model.keras") else model_dir
        model = keras.saving.load_model(model_path)
        test_ds = keras.utils.image_dataset_from_directory(
            golden_dataset_dir,
            image_size=(256, 256),
            batch_size=32,
            label_mode='binary',
            shuffle=False
        )
        _, accuracy = model.evaluate(test_ds)
        return float(accuracy)

    new_acc = evaluate_accuracy(new_model_dir)

    aiplatform.init(project=project_id, location=region)
    models = aiplatform.Model.list(filter='display_name="production-cats-dogs"', order_by="create_time desc")
    
    if not models:
        logging.warning("No Champion model found in Registry. Challenger wins by default.")
        return True

    old_acc = evaluate_accuracy(models[0].uri)
    return bool(new_acc > old_acc)


# Hard Negative Mining Component
@component(
    base_image="python:3.10",
    packages_to_install=[
        "keras>=3.0.0", 
        "jax[cuda12]", 
        "tensorflow"
    ]
)
def mine_hard_negatives_op(
    model_artifact: Input[Model],
    unlabelled_data_gcs_path: str,
    output_review_path: str
):
    import os
    os.environ["KERAS_BACKEND"] = "jax"
    
    import keras
    import tensorflow as tf
    import numpy as np
    
    model_path = f"{model_artifact.path}/model.keras" if tf.io.gfile.exists(f"{model_artifact.path}/model.keras") else model_artifact.path
    model = keras.saving.load_model(model_path)
    
    unlabelled_ds = tf.keras.utils.image_dataset_from_directory(
        unlabelled_data_gcs_path,
        shuffle=False
    )
    
    predictions = model.predict(unlabelled_ds)
    confidences = predictions.flatten()
    
    hard_negative_indices = np.where((confidences >= 0.40) & (confidences <= 0.60))[0]
    
    file_paths = unlabelled_ds.file_paths
    
    if not tf.io.gfile.exists(output_review_path):
        tf.io.gfile.makedirs(output_review_path)
        
    for idx in hard_negative_indices:
        original_path = file_paths[idx]
        conf = float(confidences[idx])
        filename = os.path.basename(original_path)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_{conf:.4f}{ext}"
        new_path = f"{output_review_path.rstrip('/')}/{new_filename}"
        tf.io.gfile.copy(original_path, new_path, overwrite=True)


@dsl.pipeline(
    name="cats-vs-dogs-jax-pipeline",
    description="Pipeline to train and deploy Cats vs Dogs classifier using Keras 3 and JAX"
)
def cats_dogs_pipeline(
    project_id: str,
    region: str,
    staging_bucket: str,
    training_data_dir: str,
    test_data_dir: str,
    task_image_uri: str,  # Image must support JAX and Keras 3
    unlabelled_data_gcs_path: str,
    review_gcs_path: str,
):
    # Component 1: TFDV Data Validation Gatekeeper
    validation_task = data_validation_op(
        unlabelled_data_gcs_path=unlabelled_data_gcs_path
    )

    # Component 2: HPO with Google Vizier
    tuning_op = HyperparameterTuningJobRunOp(
        display_name="cats_dogs_hpo_vizier",
        project=project_id,
        location=region,
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": "n1-standard-4",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": task_image_uri,
                "args": ["--data-dir", training_data_dir]
            }
        }],
        study_spec_metrics=[
            {"metric_id": "val_accuracy", "goal": "MAXIMIZE"}
        ],
        study_spec_parameters=[
            {
                "parameter_id": "learning-rate",
                "double_value_spec": {"min_value": 0.0001, "max_value": 0.01},
                "scale_type": "UNIT_LOG_SCALE"
            },
            {
                "parameter_id": "conv-filters",
                "discrete_value_spec": {"values": [32, 64]}
            }
        ],
        max_trial_count=6,
        parallel_trial_count=2,
        base_output_directory=staging_bucket
    ).after(validation_task)

    # We assume the user creates a stable path for the best model or tuning job updates base_output_directory
    model_dir = f"{staging_bucket}/model"

    # Evaluate the trained model
    eval_op = evaluate_model(
        model_dir=model_dir,
        test_data_dir=test_data_dir
    ).after(tuning_op)

    # Import the trained model directory as a dsl.Model artifact
    import_model_op = dsl.importer(
        artifact_uri=model_dir,
        artifact_class=dsl.Model
    ).after(tuning_op)

    # Hard Negative Mining (Unconditional)
    mine_hard_negatives_task = mine_hard_negatives_op(
        model_artifact=import_model_op.output,
        unlabelled_data_gcs_path=unlabelled_data_gcs_path,
        output_review_path=review_gcs_path
    ).after(tuning_op)

    # Champion vs Challenger Evaluation
    champ_vs_chall_op = champion_vs_challenger(
        new_model_dir=model_dir,
        project_id=project_id,
        region=region,
        golden_dataset_dir=test_data_dir
    ).after(eval_op)

    # Condition Check (Challenger Accuracy > Champion Accuracy)
    with Condition(champ_vs_chall_op.output == True, name="champion_vs_challenger_check"):
        
        # Component 3: XAI & Model Registry Import with Integrated Gradients
        model_upload_op = ModelImportOp(
            project=project_id,
            location=region,
            display_name="production-cats-dogs",
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest", 
            artifact_uri=model_dir,
            explanation_metadata={
                "inputs": {"image": {"input_tensor_name": "input_1"}},
                "outputs": {"prediction": {"output_tensor_name": "dense_1"}}
            },
            explanation_parameters={
                "integrated_gradients_attribution": {"step_count": 50}
            }
        )


# Component 4: Secondary Asynchronous Batch Prediction Pipeline
@dsl.pipeline(
    name="batch_scoring_pipeline",
    description="Massive-scale asynchronous batch inference for Cats vs Dogs"
)
def batch_scoring_pipeline(
    project_id: str,
    region: str,
    unlabelled_data_gcs_path: str,
    model_resource_name: str,
    batch_predict_gcs_destination: str,
):
    model_artifact = dsl.importer(
        artifact_uri=model_resource_name,
        artifact_class=dsl.Model,
        metadata={"resourceName": model_resource_name}
    )
    
    batch_predict_op = ModelBatchPredictOp(
        project=project_id,
        location=region,
        model=model_artifact.output,
        job_display_name="cats-dogs-batch-scoring",
        gcs_source_uris=[unlabelled_data_gcs_path],
        gcs_destination_output_uri_prefix=batch_predict_gcs_destination,
        instances_format="file-list",
        predictions_format="jsonl",
        machine_type="n1-standard-4",
        starting_replica_count=1,
        max_replica_count=4
    )


if __name__ == "__main__":
    import os
    from kfp import compiler
    from google.cloud import aiplatform

    logging.info("Compiling the Kubeflow Pipelines...")
    try:
        # Compile Main Pipeline
        compiler.Compiler().compile(
            pipeline_func=cats_dogs_pipeline,
            package_path="pipeline.json"
        )
        
        # Compile Batch Scoring Pipeline
        compiler.Compiler().compile(
            pipeline_func=batch_scoring_pipeline,
            package_path="batch_scoring_pipeline.json"
        )
        logging.info("Pipelines compiled successfully.")

        project_id = os.environ.get("PROJECT_ID", "YOUR_PROJECT_ID")
        region = os.environ.get("REGION", "us-central1")

        logging.info("Submitting PipelineJob to Vertex AI...")
        aiplatform.init(project=project_id, location=region)
        job = aiplatform.PipelineJob(
            display_name="cats-dogs-jax-pipeline-job",
            template_path="pipeline.json",
            parameter_values={
                "project_id": project_id,
                "region": region,
                "staging_bucket": "gs://cats-dogs-mlops-artifacts/staging",
                "training_data_dir": "gs://cats-dogs-mlops-artifacts/data/train",
                "test_data_dir": "gs://cats-dogs-mlops-artifacts/data/golden_set",
                "task_image_uri": f"{region}-docker.pkg.dev/{project_id}/cats-dogs-repo/cats-dogs-jax:latest",
                "unlabelled_data_gcs_path": "gs://cats-dogs-mlops-artifacts/data/unlabelled",
                "review_gcs_path": "gs://cats-dogs-mlops-artifacts/data/review"
            }
        )
        job.submit()
        logging.info("Pipeline submission successful.")
    except Exception as e:
        logging.warning(f"Pipeline submission skipped or failed: {e}")
