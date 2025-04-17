import torch
import argparse
import os
from tokenizer import TextTokenizer, get_base_dir
from models import RNNModel, LSTMModel, TransformerModel

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_type, model_path, vocab_size):
    """Load a trained model"""
    # Model hyperparameters (should match training)
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2
    dropout = 0.5

    if model_type == 'rnn':
        model = RNNModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    elif model_type == 'lstm':
        model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    elif model_type == 'transformer':
        model = TransformerModel(
            vocab_size, embedding_dim, hidden_dim,
            num_layers=num_layers,
            num_heads=2,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load model weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Error: Model weights not found at {model_path}")
        return None

    return model

def generate_text(model, model_type, tokenizer, prompt, max_length=100, temperature=1.0):
    """Generate text using a trained model"""
    if model is None:
        return "Model not loaded."

    if model_type in ['rnn', 'lstm']:
        return model.prompt(tokenizer, prompt, max_length, temperature)
    else:  # transformer
        return model.prompt(tokenizer, prompt, max_length, temperature)

def main():
    base_dir = get_base_dir()
    results_dir = os.path.join(base_dir, 'project_results')
    results_models_dir = os.path.join(results_dir, 'models')

    parser = argparse.ArgumentParser(description='Generate text using trained models')
    parser.add_argument('--model', type=str, required=True, choices=['rnn', 'lstm', 'transformer'],
                        help='Model type (rnn, lstm, or transformer)')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = TextTokenizer(f'{base_dir}data/bpe_tokenizer.model')
    vocab_size = tokenizer.get_vocab_size()

    # Load model
    model_path = os.path.join(results_models_dir, f'{args.model}_model.pth')
    model = load_model(args.model, model_path, vocab_size)

    # Generate text
    generated_text = generate_text(
        model, args.model, tokenizer, args.prompt, args.max_length, args.temperature
    )

    # Print prompt and generated text
    print(f"\nPrompt: {args.prompt}")
    print(f"\nGenerated text: {generated_text}")

if __name__ == "__main__":
    main() 