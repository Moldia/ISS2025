#!/usr/bin/env bash

# Exit if any command fails
set -e

# Initialize conda so it works inside the script
source "$(conda info --base)/etc/profile.d/conda.sh"

# Function to automate environment setup
setup_env() {
    DIR=$1
    YAML=$2
    ENV_NAME=$3

    echo "----------------------------------------"
    echo "Setting up environment: $ENV_NAME"
    echo "Directory: $DIR"
    echo "----------------------------------------"

    cd "$DIR"

    echo "Creating conda environment from $YAML..."
    conda env create --name "$ENV_NAME" --file "$YAML"

    echo "Activating environment $ENV_NAME..."
    conda activate "$ENV_NAME"

    echo "Installing package..."
    python setup.py install

    echo "Registering Jupyter kernel..."
    python -m ipykernel install --user --name="$ENV_NAME"

    echo "Deactivating environment..."
    conda deactivate

    cd ..
}

# Set up environments
setup_env "ISS_preprocessing" "preprocessing.yml" "ISS_preprocessing"
setup_env "ISS_decoding" "decoding.yml" "ISS_decoding"
setup_env "ISS_postprocessing" "postprocessing.yml" "ISS_postprocessing"
setup_env "ISS_CARE" "ISS_CARE.yml" "ISS_CARE"

echo ""
echo "All environments set up successfully."