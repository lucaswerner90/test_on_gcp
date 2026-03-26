import os
# Set the JAX backend for Keras 3 BEFORE importing keras
os.environ["KERAS_BACKEND"] = "jax"

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

import argparse
import keras
import tensorflow as tf
from keras import layers, models
import subprocess
import hypertune

def pull_dvc_data():
    logging.info("Starting DVC Data Rehydration: Pulling dataset from GCS...")
    try:
        subprocess.run(["dvc", "pull"], check=True)
        logging.info("DVC pull completed successfully. Data is ready in local path.")
    except subprocess.CalledProcessError as e:
        logging.error(f"DVC pull failed: {e}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description="Cats vs Dogs Vertex AI Training Task (Keras 3 + JAX)")
    parser.add_argument(
        '--data-dir', 
        type=str, 
        required=True, 
        help='GCS path to the dataset directory'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer'
    )
    parser.add_argument(
        '--conv-filters',
        type=int,
        nargs='+',
        default=[32, 64, 128],
        help='List of filters per convolutional layer. Computes N layers.'
    )
    parser.add_argument(
        '--images-per-class',
        type=int,
        default=None,
        help='Subset the dataset to a specific number of images per class for fast iteration'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    return parser.parse_args()

def build_model(learning_rate, conv_filters):
    inputs = keras.Input(shape=(None, None, 3), name="images")
    
    # Baked-in preprocessing and augmentation
    x = layers.Resizing(150, 150)(inputs)
    x = layers.Rescaling(1./255)(x)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.2)(x)
    
    # Iterate over the provided list to dynamically stack CNN layers
    for filters in conv_filters:
        x = layers.Conv2D(filters, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D(2, 2)(x)
        
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid', name="label")(x)
    
    model = keras.Model(inputs={"images": inputs}, outputs={"label": outputs})
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
        loss={"label": "binary_crossentropy"}, 
        metrics={
            "label": [
                "accuracy",
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        }
    )
    return model

def get_datasets(data_dir):
    # Using keras.utils.image_dataset_from_directory for data loading
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(256, 256),
        batch_size=32,
        label_mode='binary'
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(256, 256),
        batch_size=32,
        label_mode='binary'
    )
    
    def map_to_dict(x, y):
        return {"images": x}, {"label": y}
        
    # Map the default tuples to the new dictionary structure the Multi-Input model expects
    train_ds = train_ds.map(map_to_dict, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(map_to_dict, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Use tf.data exclusively for the preprocessing pipelines optimization
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def clean_dataset(data_dir):
    import os
    import logging
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.exists(folder_path):
            continue
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                # Check if the file is a valid JFIF (JPEG) image
                with open(fpath, "rb") as fobj:
                    is_jfif = b"JFIF" in fobj.peek(10)
            except Exception:
                is_jfif = False
                
            if not is_jfif:
                num_skipped += 1
                os.remove(fpath)
                
    if num_skipped > 0:
        logging.warning(f"Deleted {num_skipped} corrupted images from the dataset.")

def create_subset_dataset(data_dir, images_per_class):
    import os
    import logging
    subset_dir = f"{data_dir.rstrip('/')}_subset_{images_per_class}"
    if not os.path.exists(subset_dir):
        logging.info(f"Creating fast-iteration subset directory ({images_per_class} images per class) via symlinks...")
        os.makedirs(subset_dir)
        for folder_name in ("Cat", "Dog"):
            src_folder = os.path.join(data_dir, folder_name)
            if not os.path.exists(src_folder): 
                continue
            dst_folder = os.path.join(subset_dir, folder_name)
            os.makedirs(dst_folder, exist_ok=True)
            
            files = os.listdir(src_folder)[:images_per_class]
            for f in files:
                try:
                    os.symlink(os.path.abspath(os.path.join(src_folder, f)), os.path.join(dst_folder, f))
                except Exception:
                    pass
                    
    # Verification check
    for folder_name in ("Cat", "Dog"):
        dst_folder = os.path.join(subset_dir, folder_name)
        if os.path.exists(dst_folder):
            actual_count = len(os.listdir(dst_folder))
            assert actual_count == images_per_class, f"Subset error: Expected {images_per_class} images in {dst_folder}, but found {actual_count}."
            
    return subset_dir

def main():
    pull_dvc_data()
    args = parse_args()
    
    # DVC pulls to the local "data" directory, dynamically overriding GCS parameters
    data_dir = "data"
    
    logging.info("Scanning for corrupted JFIF images...")
    clean_dataset(data_dir)
    
    if args.images_per_class:
        data_dir = create_subset_dataset(data_dir, args.images_per_class)
    
    logging.info(f"Loading data from {data_dir}...")
    train_ds, val_ds = get_datasets(data_dir)
    
    logging.info(f"Building model (Keras 3 w/ JAX backend) with LR={args.learning_rate}, Filters={args.conv_filters}...")
    model = build_model(learning_rate=args.learning_rate, conv_filters=args.conv_filters)
    model.summary(print_fn=logging.info)
    
    logging.info("Starting training...")
    # Training loop using standard Keras 3
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs
    )
    
    # Dictionary outputs change the metric name automatically in Keras 3
    val_accuracy = history.history.get('val_label_accuracy', history.history.get('val_accuracy'))[-1]
    logging.info(f"Reporting val_accuracy to HyperTune: {val_accuracy}")
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='val_accuracy',
        metric_value=val_accuracy,
        global_step=10
    )
    
    # Save the model to AIP_MODEL_DIR for Vertex AI compatibility
    model_dir = os.environ.get("AIP_MODEL_DIR", "saved_model")
    logging.info(f"Saving model to: {model_dir}")
    # Under Keras 3, the generic .keras extension is standard, but you can save to a dir using SavedModel format if desired.
    model.save(f"{model_dir}/model.keras")
    logging.info(f"Model saved successfully to {model_dir}")

if __name__ == "__main__":
    main()
