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

from tokenizer import TextDataset, TextTokenizer, tokenize_dataset, get_base_dir
from models import RNNModel, LSTMModel, TransformerModel
from train import TextGenerationDataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def calculate_perplexity(model, data_loader, model_type):
    """
    Calculate perplexity on a dataset.
    Perplexity = exp(average cross-entropy loss)
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, desc=f"Calculating perplexity for {model_type}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            hidden = None # Initialize hidden state for RNN/LSTM
            if model_type in ["rnn", "lstm"]:
                if isinstance(hidden, tuple): # LSTM hidden state
                    hidden = (h.detach() for h in hidden)
                elif hidden is not None: # RNN hidden state
                    hidden = hidden.detach()
                outputs, hidden = model(inputs, hidden)
                # Reshape outputs and targets for loss calculation
                outputs = outputs.reshape(-1, outputs.shape[-1])
                targets = targets.reshape(-1)
            else:  # Transformer
                outputs = model(inputs)
                # Reshape outputs and targets for loss calculation
                outputs = outputs.reshape(-1, outputs.shape[-1])
                targets = targets.reshape(-1)
            
            # Calculate loss (cross-entropy)
            loss = criterion(outputs, targets)
            
            # Accumulate loss and token count
            total_loss += loss.item()
            total_tokens += targets.numel()
    
    # Calculate average negative log-likelihood per token
    avg_nll = total_loss / total_tokens
    
    # Perplexity = exp(avg_nll)
    perplexity = math.exp(avg_nll)
    
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


def evaluate_models():
    """Evaluate all models and generate a report"""
    base_dir = get_base_dir()
    results_dir = os.path.join(base_dir, 'project_results')
    results_texts_dir = os.path.join(results_dir, 'generated_texts')
    os.makedirs(results_texts_dir, exist_ok=True)
    
    # Load the tokenizer
    tokenizer = TextTokenizer(f'{base_dir}data/bpe_tokenizer.model')
    vocab_size = tokenizer.get_vocab_size()
    
    # Load test dataset
    test_dataset = TextDataset(f'{base_dir}data/test.jsonl')
    test_tokenized = tokenize_dataset(test_dataset, tokenizer)
    
    # Create data loader
    seq_length = 64
    batch_size = 128
    test_data = TextGenerationDataset(test_tokenized, seq_length)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    # Model hyperparameters (should match training)
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2
    dropout = 0.5
    
    # Models to evaluate
    models_info = {
        'rnn': {
            'class': RNNModel,
            'path': os.path.join(results_dir, 'models/rnn_model.pth')
         },
        'lstm': {
            'class': LSTMModel,
            'path': os.path.join(results_dir, 'models/lstm_model.pth')
         },
        'transformer': {
            'class': TransformerModel,
            'path': os.path.join(results_dir, 'models/transformer_model.pth'),
            'params': {'num_heads': 2}
        }
    }
    
    # Results dictionary
    results = {
        'Model': [],
        'Perplexity': [],
        'BLEU Score': []
    }
    
    # Custom prompts for all models
    standard_prompt = "Which do you prefer? Dogs or cats?"
    custom_prompt = "The meaning of life is"
    
    # Dictionary to store generated texts for the report
    generated_texts = {
        'standard': {},
        'custom': {}
    }
    
    # Evaluate each model
    for model_name, info in models_info.items():
        print(f"\n===== Evaluating {model_name.upper()} Model =====")
        
        # Initialize model
        model_params = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout
        }
        if 'params' in info:
            model_params.update(info['params'])
        model = info['class'](**model_params)
        
        # Load model weights
        if os.path.exists(info['path']):
            model.load_state_dict(torch.load(info['path'], map_location=device))
            model = model.to(device)
            model.eval()
            print(f"Loaded model weights from {info['path']}")
        else:
            print(f"Warning: Model weights not found at {info['path']}. Skipping evaluation for {model_name}.")
            results['Model'].append(model_name.upper())
            results['Perplexity'].append(float('nan'))
            results['BLEU Score'].append(float('nan'))
            generated_texts['standard'][model_name] = "Model not found"
            generated_texts['custom'][model_name] = "Model not found"
            continue
        
        # Calculate perplexity
        perplexity = calculate_perplexity(model, test_loader, model_name)
        print(f"Perplexity: {perplexity:.4f}")
        
        # Generate text samples for BLEU score
        samples = generate_text_samples(model, tokenizer, model_name, max_length=100)
        
        # Calculate BLEU score
        bleu = calculate_bleu_score(samples)
        print(f"BLEU Score: {bleu:.4f}")
        
        # Save generated samples to file
        samples_save_path = os.path.join(results_texts_dir, f'{model_name}_samples.json')
        with open(samples_save_path, 'w') as f:
            json.dump(samples, f, indent=2)
        print(f"Generated samples saved to {samples_save_path}")
        
        # Generate text using standard prompt
        standard_response = ""
        if model_name in ["rnn", "lstm"]:
            standard_response = model.prompt(tokenizer, standard_prompt, max_seq_length=100, temperature=1.0)
        else:  # Transformer
            standard_response = model.prompt(tokenizer, standard_prompt, max_seq_length=100, temperature=1.0)
        
        # Generate text using custom prompt
        custom_response = ""
        if model_name in ["rnn", "lstm"]:
            custom_response = model.prompt(tokenizer, custom_prompt, max_seq_length=100, temperature=1.0)
        else:  # Transformer
            custom_response = model.prompt(tokenizer, custom_prompt, max_seq_length=100, temperature=1.0)
        
        # Store responses
        generated_texts['standard'][model_name] = standard_response
        generated_texts['custom'][model_name] = custom_response
        
        # Save results
        results['Model'].append(model_name.upper())
        results['Perplexity'].append(perplexity)
        results['BLEU Score'].append(bleu)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    metrics_save_path = os.path.join(results_dir, 'evaluation_metrics.csv')
    results_df.to_csv(metrics_save_path, index=False)
    print(f"\nEvaluation metrics saved to {metrics_save_path}")
    
    # Save generated texts to JSON
    prompts_save_path = os.path.join(results_texts_dir, 'prompt_responses.json')
    with open(prompts_save_path, 'w') as f:
        json.dump(generated_texts, f, indent=2)
    print(f"Prompt responses saved to {prompts_save_path}")
    
    # Note: Report generation is now separate
    print("\nEvaluation completed.")


if __name__ == "__main__":
    # This block only runs if evaluate.py is executed directly
    # Import DataLoader here to avoid circular imports if needed when running directly
    # from torch.utils.data import DataLoader
    evaluate_models() 