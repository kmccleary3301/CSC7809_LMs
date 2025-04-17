#!/bin/bash

# Determine the base directory
if [ "$(basename "$PWD")" = "final_submission" ]; then
  BASE_DIR="./"
  PARENT_DIR="../"
else
  BASE_DIR="final_submission/"
  PARENT_DIR="./"
fi

# Activate the conda environment
echo "Activating conda environment csc7809..."
# source /opt/conda/etc/profile.d/conda.sh
# conda activate csc7809

# Check if the environment was activated successfully
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment. Please make sure csc7809 environment exists."
  exit 1
fi

# Install required packages
echo "Installing required packages..."
pip install sentencepiece nltk tqdm pandas matplotlib

# Create necessary directories
mkdir -p "${BASE_DIR}data" \
         "${BASE_DIR}project_results/models" \
         "${BASE_DIR}project_results/plots" \
         "${BASE_DIR}project_results/generated_texts" \
         "${BASE_DIR}architecture_diagrams"

# Check if data files exist
if [ ! -f "${BASE_DIR}data/train.jsonl" ] || [ ! -f "${BASE_DIR}data/test.jsonl" ]; then
  echo "Copying dataset files..."
  cp "${PARENT_DIR}CSC7809_FoundationModels/Project2/data/train.jsonl" "${BASE_DIR}data/"
  cp "${PARENT_DIR}CSC7809_FoundationModels/Project2/data/test.jsonl" "${BASE_DIR}data/"
fi

# Navigate to the final_submission directory if we're not already there
if [ "$BASE_DIR" != "./" ]; then
  cd "$BASE_DIR"
fi

# Generate architecture diagrams (assuming they are needed in the final report)
echo "Generating architecture diagrams..."
cd architecture_diagrams
python create_diagrams.py
# Copy diagrams to results folder for easy collection
mkdir -p ../project_results/architecture_diagrams
cp *.png ../project_results/architecture_diagrams/
cd ..

# Run the main script to perform all steps
echo "Running the main workflow (tokenize, train, evaluate)..."
python main.py --tokenize --train --evaluate

echo "Workflow completed. All results saved in ${BASE_DIR}project_results/"

# Optional: Run generation for demonstration
# echo "Running text generation for demonstration..."
# python main.py --generate --prompt "Which do you prefer? Dogs or cats?" 