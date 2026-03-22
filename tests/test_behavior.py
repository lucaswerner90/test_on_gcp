import os
# Set the JAX backend for Keras 3 BEFORE importing keras
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import keras
import tensorflow as tf
import numpy as np
from src.task import build_model

@pytest.fixture
def dummy_model_and_image(tmp_path):
    """Helper fixture to create/load a dummy saved model and a dummy image tensor."""
    # 1. Initialize and save a compiled model to mock an enterprise artifact
    model = build_model()
    model_path = tmp_path / "dummy_model.keras"
    model.save(model_path)
    
    # 2. Load the dummy saved model
    loaded_model = keras.saving.load_model(model_path)
    
    # 3. Create a dummy image tensor (batch_size=1, 150x150 RGB)
    image_tensor = tf.random.uniform(shape=(1, 150, 150, 3))
    
    return loaded_model, image_tensor

def test_model_invariance_to_flip(dummy_model_and_image):
    """Asserts the model's prediction difference between original and flipped images is < 5%."""
    model, image = dummy_model_and_image
    
    pred_original = np.array(model(image))
    pred_flipped = np.array(model(tf.image.flip_left_right(image)))
    
    diff = np.abs(pred_original - pred_flipped)
    
    assert np.all(diff < 0.05), f"Prediction difference exceeded 5% after flipping: {diff}"