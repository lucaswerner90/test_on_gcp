import os
# Set the JAX backend for Keras 3 BEFORE importing keras
os.environ["KERAS_BACKEND"] = "jax"

import argparse
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Cats vs Dogs Error Analysis (Keras 3 + JAX)")
    parser.add_argument('--model-dir', type=str, required=True, help='Path to the saved model directory (.keras or SavedModel)')
    parser.add_argument('--val-data-dir', type=str, required=True, help='Path to the validation dataset directory')
    return parser.parse_args()

def analyze_errors():
    args = parse_args()
    
    print(f"Loading Keras 3 model from {args.model_dir}...")
    
    # Handle both directory (containing model.keras) and direct file path
    model_path = f"{args.model_dir}/model.keras" if os.path.exists(f"{args.model_dir}/model.keras") else args.model_dir
    model = keras.saving.load_model(model_path)
    
    # 0 = Cats, 1 = Dogs (based on alphabetical sorting in image_dataset_from_directory)
    class_names = ["Cat", "Dog"]
    
    print(f"Loading validation dataset from {args.val_data_dir}...")
    # The dataset MUST NOT be shuffled so predictions map 1:1 with the loaded image order
    val_ds = keras.utils.image_dataset_from_directory(
        args.val_data_dir,
        image_size=(256, 256),
        batch_size=32,
        label_mode='binary',
        shuffle=False
    )
    
    print("Generating predictions via JAX backend...")
    predictions = model.predict(val_ds)
    y_pred_probs = predictions.flatten()
    y_pred = (y_pred_probs > 0.5).astype("int32")
    
    print("Collecting true labels and raw images...")
    y_true = []
    images = []
    
    # Load all images and labels into memory for plotting
    for batch_images, batch_labels in val_ds:
        images.append(batch_images.numpy())
        y_true.append(batch_labels.numpy().flatten())
        
    images = np.concatenate(images, axis=0)
    y_true = np.concatenate(y_true, axis=0).astype("int32")
    
    # Find indices where the prediction doesn't match the true label
    incorrect_indices = np.where(y_pred != y_true)[0]
    print(f"Found {len(incorrect_indices)} mistakes out of {len(y_true)} validation samples.")
    
    if len(incorrect_indices) == 0:
        print("No errors found! The model is 100% accurate on this validation set.")
        return
        
    print("Plotting the first 9 mistakes...")
    plt.figure(figsize=(10, 10))
    
    # Plot up to 9 mistakes
    num_to_plot = min(9, len(incorrect_indices))
    for i in range(num_to_plot):
        idx = incorrect_indices[i]
        
        # keras load images as float32 in range [0, 255], need uint8 for matplotlib
        raw_img = images[idx].astype("uint8")
        
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        
        # Confidence is the probability of the *predicted* class
        confidence = y_pred_probs[idx] if y_pred[idx] == 1 else 1.0 - y_pred_probs[idx]
        
        plt.subplot(3, 3, i + 1)
        plt.imshow(raw_img)
        
        title_text = f"True: {true_label}\nPred: {pred_label} ({confidence*100:.1f}%)"
        plt.title(title_text, color='red' if true_label != pred_label else 'green')
        plt.axis("off")
        
    plt.tight_layout()
    plt.savefig("error_analysis_grid.png")
    print("Saved error grid to 'error_analysis_grid.png'.")
    plt.show()

if __name__ == '__main__':
    analyze_errors()
