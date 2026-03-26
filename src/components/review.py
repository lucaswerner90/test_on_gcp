from kfp.dsl import component, Input, Model

"""
Hard Negative Mining (Active Learning) Component

Goal: To autonomously identify "confusing" examples from mass unlabelled data buckets where 
the model is highly uncertain (confidence between 0.40 and 0.60). By exporting these 
specific images to a Human Review bucket, we create an Active Learning data-flywheel.
This component specifically uses tf.io.gfile to write directly to GCS because ephemeral 
Vertex AI containers securely lack the Git SSH keys needed to push via DVC during runtime.
"""
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
    
    def map_to_dict(x, y):
        return {"images": x}, {"label": y}
    unlabelled_ds = unlabelled_ds.map(map_to_dict)
    
    predictions = model.predict(unlabelled_ds)
    confidences = predictions["label"].flatten()
    
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
