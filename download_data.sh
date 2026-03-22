#!/bin/bash
# Script to download the Microsoft Cats vs Dogs dataset using the Kaggle CLI

# Ensure we are working inside the virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: .venv directory not found."
fi

# Ensure the Kaggle CLI is installed
if ! command -v kaggle &> /dev/null
then
    echo "Kaggle CLI could not be found. Please ensure your virtual environment is active."
    echo "Installing kaggle..."
    pip install kaggle
fi

# Create the target directory
TARGET_DIR="data"
mkdir -p ${TARGET_DIR}

echo "Downloading and unzipping 'shaunthesheep/microsoft-catsvsdogs-dataset' to '${TARGET_DIR}'..."
echo "Note: This requires you to have your Kaggle API credentials configured."
echo "      (e.g., KAGGLE_USERNAME and KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json file)"

# Download and unzip the dataset directly to the target folder
kaggle datasets download -d shaunthesheep/microsoft-catsvsdogs-dataset -p ${TARGET_DIR}/ --unzip

echo "Dataset download complete!"