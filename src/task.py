import os
# Set the JAX backend for Keras 3 BEFORE importing keras
os.environ["KERAS_BACKEND"] = "jax"

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

import argparse
import keras
import tensorflow as tf
from keras import layers, models
import hypertune

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
        default=32,
        help='Base number of convolutional filters'
    )
    return parser.parse_args()

def build_model(learning_rate, conv_filters):
    model = models.Sequential([
        # Baked-in preprocessing and augmentation
        keras.Input(shape=(None, None, 3)),
        layers.Resizing(150, 150),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        
        # CNN block
        layers.Conv2D(conv_filters, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(conv_filters * 2, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(conv_filters * 4, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
        loss='binary_crossentropy', 
        metrics=['accuracy']
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
    
    # Use tf.data exclusively for the preprocessing pipelines optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds

def main():
    args = parse_args()
    
    logging.info(f"Loading data from {args.data_dir}...")
    train_ds, val_ds = get_datasets(args.data_dir)
    
    logging.info(f"Building model (Keras 3 w/ JAX backend) with LR={args.learning_rate}, Filters={args.conv_filters}...")
    model = build_model(learning_rate=args.learning_rate, conv_filters=args.conv_filters)
    model.summary(print_fn=logging.info)
    
    logging.info("Starting training...")
    # Training loop using standard Keras 3
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )
    
    val_accuracy = history.history['val_accuracy'][-1]
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
