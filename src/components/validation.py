from kfp.dsl import component

"""
Data Validation Gatekeeper Component

Goal: To aggressively reject corrupted or structurally invalid datasets BEFORE spinning up 
expensive GPU compute resources. The Microsoft Cats vs Dogs dataset notoriously contains 
0-byte files, grayscale images, and fake JPEGs (e.g., GIFs renamed to .jpg). 
This component uses TensorFlow Data Validation (TFDV) to scan a sample of images 
and assert that they strictly conform to a 3-channel RGB format. If anomalies are found, 
the Kubeflow pipeline halts early, saving cloud credits.
"""
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
    # We only sample 100 images to keep this pipeline step fast and lightweight.
    # The goal is to safely catch systemic data corruption, not exhaustively check every file.
    for path in file_paths[:100]:
        img_raw = tf.io.read_file(path)
        
        # tf.image.decode_image will crash if the image is a fake JPEG (e.g. GIF) or 0-byte,
        # naturally catching severe structural corruption right here.
        img = tf.image.decode_image(img_raw)
        shape = img.shape
        
        # We record the number of color channels. Standard RGB images should have exactly 3.
        # Grayscale images would only have 1 (which would break our Keras 3 model shape expectations).
        records.append({
            "channels": shape[2] if len(shape) > 2 else 1,
            "filename": path
        })
        
    df = pd.DataFrame(records)
    stats = tfdv.generate_statistics_from_dataframe(df)
    
    schema = tfdv.infer_schema(stats)
    
    # We explicitly define the expected schema: Every image MUST be strictly 3-channel RGB.
    # This proactively prevents the downstream JAX model from crashing during GPU matrix 
    # math if it unexpectedly receives a 1-channel grayscale image.
    tfdv.get_feature(schema, 'channels').presence.min_fraction = 1.0
    tfdv.set_domain(schema, 'channels', tfdv.IntDomain(min=3, max=3))
    
    anomalies = tfdv.validate_statistics(statistics=stats, schema=schema)
    
    if anomalies.anomaly_info:
        for feature_name, anomaly_info in anomalies.anomaly_info.items():
            logging.error(f"TFDV Anomaly in {feature_name}: {anomaly_info.description}")
        raise ValueError("Data Validation Failed: Anomalies found in image features (not RGB or standard format). Halting pipeline.")
        
    logging.info("TFDV: Data Validation passed. Images conform to RGB representations.")
