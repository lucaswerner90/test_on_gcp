import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

from kfp import dsl
from kfp.dsl import (
    component,
    Input,
    Model,
    Output,
    Metrics,
    ClassificationMetrics,
    If
)
from google_cloud_pipeline_components.v1.hyperparameter_tuning_job import HyperparameterTuningJobRunOp
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp as ModelImportOp
from google_cloud_pipeline_components.types import artifact_types

# Modularized Components
from components.validation import data_validation_op
from components.evaluate import evaluate_model, champion_vs_challenger
from components.review import mine_hard_negatives_op


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
    with If(champ_vs_chall_op.output == True, name="champion_vs_challenger_check"):
        
        unmanaged_model_importer = dsl.importer(
            artifact_uri=model_dir,
            artifact_class=artifact_types.UnmanagedContainerModel,
            metadata={
                "containerSpec": {
                    "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"
                }
            }
        )
        
        # Component 3: XAI & Model Registry Import with Integrated Gradients
        model_upload_op = ModelImportOp(
            project=project_id,
            location=region,
            display_name="production-cats-dogs",
            unmanaged_container_model=unmanaged_model_importer.output,
            explanation_metadata={
                "inputs": {"image": {"input_tensor_name": "images"}},
                "outputs": {"prediction": {"output_tensor_name": "label"}}
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
        artifact_class=artifact_types.VertexModel,
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
    import argparse
    from kfp import compiler
    from google.cloud import aiplatform

    parser = argparse.ArgumentParser(description="Compile and execute the Kubeflow Pipeline.")
    parser.add_argument("--compile-only", action="store_true", help="Only generate the JSON artifacts without submitting to Vertex AI.")
    args = parser.parse_args()

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
        
        if args.compile_only:
            logging.info("Exiting early: --compile-only flag passed. No jobs will be submitted to Vertex AI.")
            exit(0)

        project_id = os.environ.get("GCP_PROJECT_ID", None)
        region = os.environ.get("GCP_REGION", None)
        bucket_url = os.environ.get("GCP_BUCKET_URL", None)

        if not project_id or not region or not bucket_url:
            raise ValueError("GCP_PROJECT_ID, GCP_REGION and GCP_BUCKET_URL must be set in environment variables")

        logging.info("Submitting PipelineJob to Vertex AI...")
        aiplatform.init(project=project_id, location=region)
        job = aiplatform.PipelineJob(
            display_name="cats-dogs-jax-pipeline-job",
            template_path="pipeline.json",
            parameter_values={
                "project_id": project_id,
                "region": region,
                "staging_bucket": f"{bucket_url}/staging",
                "training_data_dir": f"{bucket_url}/data/train",
                "test_data_dir": f"{bucket_url}/data/golden_set",
                "task_image_uri": f"{region}-docker.pkg.dev/{project_id}/cats-dogs-repo/cats-dogs-jax:latest",
                "unlabelled_data_gcs_path": f"{bucket_url}/data/unlabelled",
                "review_gcs_path": f"{bucket_url}/data/review"
            }
        )
        job.submit()
        logging.info("Pipeline submission successful.")
    except Exception as e:
        logging.warning(f"Pipeline submission skipped or failed: {e}")
