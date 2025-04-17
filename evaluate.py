import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch.utils.data import DataLoader
import argparse
import yaml

from tokenizer import TextDataset, TextTokenizer, tokenize_dataset, get_base_dir
from models import RNNModel, LSTMModel, TransformerModel
from pretrainer import TextGenerationDataset, Pretrainer

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def calculate_perplexity(model, data_loader, model_type, criterion):
    """
    Calculate perplexity on a dataset.
    Perplexity = exp(average cross-entropy loss)
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    pad_id = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc=f"Calculating perplexity for {model_type}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            hidden = None
            if model_type in ["rnn", "lstm"]:
                if isinstance(hidden, tuple): hidden = tuple(h.detach() for h in hidden)
                elif hidden is not None: hidden = hidden.detach()
                
                outputs, hidden = model(inputs, hidden)
            else:  # Transformer
                outputs = model(inputs)
            
            # Reshape outputs and targets for loss calculation
            outputs_flat = outputs.reshape(-1, outputs.shape[-1])
            targets_flat = targets.reshape(-1)

            # Calculate loss (summed over batch, ignoring padding)
            loss = criterion(outputs_flat, targets_flat)
            
            # Accumulate loss and token count (only non-padded tokens)
            total_loss += loss.item()
            non_pad_tokens = (targets_flat != pad_id).sum().item()
            total_tokens += non_pad_tokens
    
    if total_tokens == 0:
        return float('inf')
        
    # Calculate average negative log-likelihood per token
    avg_nll = total_loss / total_tokens
    
    # Perplexity = exp(avg_nll)
    try:
        perplexity = math.exp(avg_nll)
    except OverflowError:
        perplexity = float('inf')
        
    return perplexity


def generate_text_samples(model, tokenizer, model_type, num_samples=10, max_length=100, temperature=1.0):
    """Generate text samples using the model"""
    base_dir = get_base_dir()
    model.eval()
    
    # Load a few samples from the test dataset for prompts
    test_dataset = TextDataset(f'{base_dir}data/test.jsonl')
    samples = []
    
    for i in range(min(num_samples, len(test_dataset))):
        text = test_dataset[i]
        prompt = ' '.join(text.split()[:5])  # Take first 5 words as prompt
        
        # Generate text using the model
        if model_type in ["rnn", "lstm"]:
            generated_text = model.prompt(tokenizer, prompt, max_seq_length=max_length, temperature=temperature)
        else:  # Transformer
            generated_text = model.prompt(tokenizer, prompt, max_seq_length=max_length, temperature=temperature)
        
        samples.append({
            'prompt': prompt,
            'original': text,
            'generated': generated_text
        })
    
    return samples


def calculate_bleu_score(text_samples):
    """Calculate BLEU score for the generated samples vs. original texts"""
    references = []
    hypotheses = []
    
    for sample in text_samples:
        # Tokenize reference (original) and hypothesis (generated) texts
        reference = sample['original'].split()
        hypothesis = sample['generated'].split()
        
        # BLEU expects a list of reference sentences
        references.append([reference])
        hypotheses.append(hypothesis)
    
    # Calculate BLEU score with smoothing
    smoothing = SmoothingFunction().method1
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=smoothing)
    
    return bleu_score


def evaluate_model(config_path):
    """Evaluate a single model based on its config file."""
    base_dir = get_base_dir()
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from: {config_path}")
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return None

    model_type = config.get('model_type')
    if not model_type:
        print(f"Error: 'model_type' not found in {config_path}")
        return None

    results_dir = config.get('save_dir', os.path.join(base_dir, 'project_results', model_type))
    models_dir = os.path.join(results_dir, 'models')
    model_path = os.path.join(models_dir, f'{model_type}_model.pth')
    results_texts_dir = os.path.join(results_dir, 'generated_texts')
    os.makedirs(results_texts_dir, exist_ok=True)

    # Load the tokenizer
    tokenizer_path = config.get('tokenizer_path', f'{base_dir}data/bpe_tokenizer.model')
    tokenizer = TextTokenizer(tokenizer_path)
    vocab_size = config.get('vocab_size', tokenizer.get_vocab_size())
    pad_id = tokenizer.pad_id

    # Load test dataset
    test_data_path = config.get('test_data_path', f'{base_dir}data/test.jsonl')
    max_samples_test = config.get('val_dataset', {}).get('max_samples', None)
    test_dataset = TextDataset(test_data_path, max_samples=max_samples_test)
    
    # Tokenize test data with correct max_seq_length
    max_seq_length = config.get('max_seq_length', 512)
    test_tokenized = tokenize_dataset(test_dataset, tokenizer, max_length=max_seq_length)
    
    # Create data loader
    batch_size = config.get('batch_size', 128)
    test_data = TextGenerationDataset(test_tokenized, max_seq_length=max_seq_length)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Initialize model using Pretrainer's logic (which reads from config)
    model_params = {
        'vocab_size': vocab_size,
        'embedding_dim': config.get('embedding_dim', 256),
        'hidden_dim': config.get('hidden_dim', 512),
        'num_layers': config.get('num_layers', 2),
        'dropout': config.get('dropout', 0.5)
    }
    if model_type == 'transformer':
        model_params['num_heads'] = config.get('num_heads', 2)

    if model_type == 'rnn': model_class = RNNModel
    elif model_type == 'lstm': model_class = LSTMModel
    elif model_type == 'transformer': model_class = TransformerModel
    else: 
        print(f"Unknown model type in config: {model_type}")
        return None
        
    model = model_class(**model_params)

    # Load model weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Error: Model weights not found at {model_path}. Cannot evaluate.")
        return None

    # Define Loss criterion for perplexity (needs ignore_index)
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=pad_id)

    # Calculate perplexity
    print(f"Calculating Perplexity...")
    perplexity = calculate_perplexity(model, test_loader, model_type, criterion)
    print(f"Perplexity: {perplexity:.4f}")

    # Generate text samples for BLEU score
    print(f"Generating samples for BLEU score...")
    samples = generate_text_samples(model, tokenizer, model_type, num_samples=100, max_length=max_seq_length, temperature=1.0)
    
    # Calculate BLEU score
    bleu = calculate_bleu_score(samples)
    print(f"BLEU Score: {bleu:.4f}")

    # Save generated samples to file
    samples_save_path = os.path.join(results_texts_dir, f'{model_type}_best_samples.json')
    with open(samples_save_path, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"Generated samples saved to {samples_save_path}")

    # Store results
    model_results = {
        'Model': model_type.upper(),
        'Config': config_path,
        'Perplexity': perplexity,
        'BLEU Score': bleu
    }

    return model_results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Language Models")
    parser.add_argument(
        '--configs', 
        nargs='+', 
        default=[
            'configs/rnn_best.yaml',
            'configs/lstm_best.yaml',
            'configs/transformer_best.yaml'
        ],
        help='Paths to the best model configuration YAML files to evaluate.'
    )
    args = parser.parse_args()

    all_results = []
    for config_path in args.configs:
        print(f"\n===== Evaluating configuration: {config_path} =====")
        result = evaluate_model(config_path)
        if result:
            all_results.append(result)

    # Display results in a table
    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n===== Evaluation Summary =====")
        print(results_df.to_string(index=False))

        # Save results to CSV
        summary_path = os.path.join(get_base_dir(), 'project_results', 'evaluation_summary.csv')
        results_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
    else:
        print("\nNo models were successfully evaluated.")

if __name__ == "__main__":
    main() 