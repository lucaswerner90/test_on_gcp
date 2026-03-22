
import os
# Set the JAX backend for Keras 3 BEFORE importing keras
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import keras
import tensorflow as tf
import numpy as np

# Importing build_model from our task module
from src.task import build_model

def test_image_resizing():
    """Test that the standard image resizing scales a dummy tensor to exactly 150x150."""
    dummy_tensor = tf.random.uniform(shape=(200, 200, 3))
    
    resizer = keras.layers.Resizing(150, 150)
    output = resizer(dummy_tensor)
    
    assert output.shape == (150, 150, 3), f"Expected shape (150, 150, 3), got {output.shape}"

def test_model_architecture():
    """Test model initialization, forward pass output shape, and valid sigmoid bounds."""
    model = build_model()
    dummy_batch = tf.random.uniform(shape=(4, 150, 150, 3))
    
    preds = model(dummy_batch)
    
    assert preds.shape == (4, 1), f"Expected shape (4, 1), got {preds.shape}"
    
    # Convert predictions to numpy array to assert bounds
    preds_np = np.array(preds)
    assert np.all((preds_np >= 0.0) & (preds_np <= 1.0)), "Predictions must be strictly between 0.0 and 1.0"
