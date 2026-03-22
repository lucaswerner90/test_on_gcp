from kfp.dsl import component

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
