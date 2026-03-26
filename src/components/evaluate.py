from kfp.dsl import component, Output, Metrics, ClassificationMetrics

"""
Model Evaluation Component

Goal: To compute offline validation metrics against the "Golden" test dataset 
using the trained Keras 3 (JAX backend) model. This logs essential metrics 
such as Accuracy, Loss, and the detailed Confusion Matrix directly to the 
Kubeflow UI for human review before any deployment decisions are made.
"""
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
    
    def map_to_dict(x, y):
        return {"images": x}, {"label": y}
    test_ds = test_ds.map(map_to_dict)
    
    # model.evaluate returns [loss, accuracy]
    results = model.evaluate(test_ds)
    loss, accuracy = results[0], results[1]
    metrics.log_metric("accuracy", float(accuracy))
    metrics.log_metric("loss", float(loss))
    
    predictions = model.predict(test_ds)
    # predictions is now a dictionary because of the MIMO architecture
    y_pred = (predictions["label"] > 0.5).astype("int32").flatten()
    y_true = np.concatenate([y["label"] for x, y in test_ds], axis=0).flatten()
    
    cm = confusion_matrix(y_true, y_pred)
    classification_metrics.log_confusion_matrix(["Cat", "Dog"], cm.tolist())
    
    return float(accuracy)


"""
Champion vs Challenger (Model Promotion) Component

Goal: To act as a deployment safeguard. It compares the newly trained model ("Challenger")
against the existing production model ("Champion") found in the Vertex AI Model Registry.
Both models are evaluated on the exact same Golden Dataset. If the new model performs better,
this component returns True, allowing the pipeline to proceed with deploying the new model.
Otherwise, it returns False and halts deployment.
"""
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
        def map_to_dict(x, y):
            return {"images": x}, {"label": y}
        test_ds = test_ds.map(map_to_dict)
        
        results = model.evaluate(test_ds)
        accuracy = results[1]
        return float(accuracy)

    new_acc = evaluate_accuracy(new_model_dir)

    aiplatform.init(project=project_id, location=region)
    models = aiplatform.Model.list(filter='display_name="production-cats-dogs"', order_by="create_time desc")
    
    if not models:
        logging.warning("No Champion model found in Registry. Challenger wins by default.")
        return True

    old_acc = evaluate_accuracy(models[0].uri)
    return bool(new_acc > old_acc)
