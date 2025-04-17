# Sequential Neural Networks for Text Generation

This project implements three different sequential neural network architectures (RNN, LSTM, and Transformer) for text generation using PyTorch. The models are trained on a text corpus from Project Gutenberg and evaluated using perplexity and BLEU score.

## Requirements

- Python 3.10+
- PyTorch 2.x
- SentencePiece
- NLTK
- pandas
- matplotlib
- tqdm
- numpy

To set up the environment, use the conda environment `csc7809`:

```bash
# Activate the provided conda environment
conda activate csc7809

# Install any missing dependencies
pip install sentencepiece nltk tqdm pandas matplotlib
```

## Project Structure

- `tokenizer.py`: Implements BPE tokenization using SentencePiece
- `models.py`: Defines the RNN, LSTM, and Transformer model architectures
- `train.py`: Training script for all models
- `evaluate.py`: Evaluation script for calculating perplexity and BLEU score
- `generate.py`: Script for generating text from prompts
- `main.py`: Main script for running the entire workflow
- `run.sh`: Shell script for automating the entire process
- `report.md`: Project report
- `data/`: Directory containing the dataset
- `models/`: Directory containing saved model weights
- `plots/`: Directory containing training loss curves
- `generated_texts/`: Directory containing generated text samples
- `architecture_diagrams/`: Directory containing model architecture diagrams

## Running the Code

### Option 1: Using the Shell Script

For the simplest execution, run the shell script which will:
1. Activate the conda environment
2. Install required packages
3. Copy dataset files if needed
4. Generate architecture diagrams
5. Run the entire workflow

```bash
# Make sure the script is executable
chmod +x run.sh

# Run the script
./run.sh
```

### Option 2: Using the Main Script

The main script provides a unified interface for all steps:

```bash
# Activate the conda environment first
conda activate csc7809

# Display help
python main.py

# Run the entire workflow (tokenize, train, evaluate, generate)
python main.py --all

# Run individual steps
python main.py --tokenize  # Train the tokenizer
python main.py --train     # Train the models
python main.py --evaluate  # Evaluate the models

# Generate text with a specific model and prompt
python main.py --generate --model lstm --prompt "The meaning of life is"
```

### Option 3: Running Individual Scripts

Alternatively, you can run each script individually:

#### 1. Dataset Preparation

The dataset files (`train.jsonl` and `test.jsonl`) should be placed in the `data/` directory.

#### 2. Training the Tokenizer

To train the BPE tokenizer on the training data:

```bash
python tokenizer.py
```

This will train a tokenizer with vocabulary size of 10,000 and save it to `data/bpe_tokenizer.model`.

#### 3. Training the Models

To train all three models (RNN, LSTM, and Transformer):

```bash
python train.py
```

This will train each model and save the best weights to the `models/` directory. Training progress and loss curves will be saved to the `plots/` directory.

#### 4. Evaluating the Models

To evaluate the models on the test dataset:

```bash
python evaluate.py
```

This will calculate perplexity and BLEU score for each model and save the results to `evaluation_results.csv` and `model_evaluation.md`.

#### 5. Generating Text

To generate text using a trained model:

```bash
python generate.py --model rnn --prompt "Your prompt text" --temperature 1.0
```

Options:
- `--model`: Model type (rnn, lstm, or transformer)
- `--prompt`: Text prompt for generation
- `--max_length`: Maximum length of generated text (default: 100)
- `--temperature`: Temperature for sampling (default: 1.0)

## Results

See `report.md` for a detailed analysis of the results, including:
- Training and validation loss curves
- Perplexity and BLEU score for each model
- Examples of generated text
- Discussion of the trade-offs between different model architectures 