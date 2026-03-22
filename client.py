import os
# Set the JAX backend for Keras 3 BEFORE importing keras
os.environ["KERAS_BACKEND"] = "jax"

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

import argparse
import keras
import numpy as np
from google.cloud import aiplatform

def predict_image(project_id: str, location: str, endpoint_id: str, image_path: str):
    # Initialize the Vertex AI SDK
    aiplatform.init(project=project_id, location=location)

    logging.info(f"Loading and preprocessing image: {image_path}")
    
    # Load a local image and resize it to 150x150
    # Keras 3 utils handles image loading directly
    img = keras.utils.load_img(image_path, target_size=(150, 150))
    
    # Convert it to an array and expand dimensions to create a batch of 1
    # Output shape: (1, 150, 150, 3)
    img_array = keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0) 
    
    # Convert numpy array to JSON serializable list structure
    instances = img_batch.tolist()

    # Get the endpoint
    logging.info(f"Connecting to Endpoint ID or resource name: {endpoint_id}...")
    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)

    logging.info("Sending payload...")
    
    # Send the payload using endpoint.predict()
    response = endpoint.predict(instances=instances)
    
    # The output format from the sigmoid layer is a list of [probability]
    prediction = response.predictions[0]
    prob_dog = float(prediction[0])
    
    # Using the same mapping: 0=Cat, 1=Dog
    if prob_dog > 0.5:
        class_name = "Dog"
        confidence = prob_dog * 100
    else:
        class_name = "Cat"
        confidence = (1.0 - prob_dog) * 100
        
    # Log a human-readable result
    logging.info("--- Prediction Result ---")
    logging.info(f"Image: {image_path}")
    logging.info(f"Classification: {class_name}")
    logging.info(f"Confidence: {confidence:.2f}%")
    logging.info("-------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Vertex AI Endpoint Predict locally (JAX model)")
    parser.add_argument("--project-id", required=True, type=str, help="GCP Project ID")
    parser.add_argument("--region", required=True, type=str, help="GCP Region (e.g., us-central1)")
    parser.add_argument("--endpoint-id", required=True, type=str, help="Vertex AI Endpoint ID or full resource name")
    parser.add_argument("--image-path", required=True, type=str, help="Path to local image to predict")
    
    args = parser.parse_args()
    predict_image(args.project_id, args.region, args.endpoint_id, args.image_path)
