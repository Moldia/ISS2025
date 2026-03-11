#!/usr/bin/env bash

set -eo pipefail

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is not installed or not available in PATH."
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "ISS_preprocessing" ] || [ ! -d "ISS_decoding" ] || [ ! -d "ISS_postprocessing" ]; then
    echo "Error: run this script from the ISS2025 repository root."
    exit 1
fi

setup_env() {
    DIR="$1"
    YAML="$2"
    ENV_NAME="$3"

    echo "----------------------------------------"
    echo "Setting up environment: $ENV_NAME"
    echo "Module directory: $DIR"
    echo "----------------------------------------"

    if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        echo "Environment $ENV_NAME already exists. Updating from $DIR/$YAML ..."
        conda env update -n "$ENV_NAME" -f "$DIR/$YAML"
    else
        echo "Environment $ENV_NAME does not exist. Creating from $DIR/$YAML ..."
        conda env create -n "$ENV_NAME" -f "$DIR/$YAML"
    fi

    echo "Activating $ENV_NAME ..."
    conda activate "$ENV_NAME"

    echo "Installing/updating Python package in $DIR ..."
    (
        cd "$DIR"
        pip install .
    )

    echo "Registering Jupyter kernel for $ENV_NAME ..."
    python -m ipykernel install --user --name="$ENV_NAME"

    conda deactivate
    echo "Finished $ENV_NAME"
    echo
}

setup_env "ISS_preprocessing" "ISS_preprocessing.yml" "ISS_preprocessing"
setup_env "ISS_decoding" "ISS_decoding.yml" "ISS_decoding"
setup_env "ISS_postprocessing" "ISS_postprocessing.yml" "ISS_postprocessing"
setup_env "ISS_CARE" "ISS_CARE.yml" "ISS_CARE"

echo "All environments installed or updated successfully."