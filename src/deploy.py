import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

import argparse
from google.cloud import aiplatform

def deploy_model(project_id: str, location: str):
    # Initialize the Vertex AI SDK
    aiplatform.init(project=project_id, location=location)

    # 1. Fetch the latest model from the Vertex AI Registry named "production-cats-dogs"
    # Note: "production-cats-dogs" is the literal hardcoded display_name assigned to the 
    # model by the `ModelImportOp` during the final step of the training pipeline (see src/pipeline.py).
    logging.info("Fetching the latest model 'production-cats-dogs' from Registry...")
    
    # List models with the given display name, ordered by creation time descending
    models = aiplatform.Model.list(
        filter='display_name="production-cats-dogs"',
        order_by="create_time desc"
    )

    if not models:
        logging.error("No model named 'production-cats-dogs' found in the registry.")
        raise ValueError("No model named 'production-cats-dogs' found in the registry.")

    latest_model = models[0]
    logging.info(f"Found latest model: {latest_model.resource_name} (Version: {latest_model.version_id})")

    # 2. Create a Vertex AI Endpoint
    logging.info("Creating a new Vertex AI Endpoint...")
    endpoint = aiplatform.Endpoint.create(
        display_name="cats-dogs-jax-endpoint",
        project=project_id,
        location=location,
    )
    logging.info(f"Endpoint created successfully: {endpoint.resource_name}")

    # 3. Deploy the model to the created endpoint
    logging.info("Deploying the model to the endpoint using n1-standard-2 machine type...")
    logging.info("This process may take several minutes.")
    
    latest_model.deploy(
        endpoint=endpoint,
        deployed_model_display_name="cats-dogs-jax-deployed-model",
        machine_type="n1-standard-2",
        sync=True
    )

    logging.info("Deployment complete!")
    logging.info(f"You can now send predictions to endpoint: {endpoint.resource_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Cats vs Dogs JAX model to a Vertex AI Endpoint")
    parser.add_argument("--project-id", required=True, type=str, help="GCP Project ID")
    parser.add_argument("--region", required=True, type=str, help="GCP Region (e.g., us-central1)")
    
    args = parser.parse_args()
    deploy_model(args.project_id, args.region)
