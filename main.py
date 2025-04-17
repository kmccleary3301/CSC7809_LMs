import os
import sys
import argparse

try:
    file_dir = globals()['_dh'][0]
except:
	file_dir = os.path.dirname(__file__)

def print_header(text):
    """Print a header with the given text"""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80 + "\n")

def get_base_dir():
    """Get the base directory for file paths"""
    # Check if we're running from inside final_submission directory
    return file_dir+"/"

def main():
    parser = argparse.ArgumentParser(description='Run the sequential neural networks project')
    parser.add_argument('--tokenize', action='store_true', help='Train the tokenizer')
    parser.add_argument('--train', action='store_true', help='Train the models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the models')
    parser.add_argument('--generate', action='store_true', help='Generate text from models')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    parser.add_argument('--prompt', type=str, default="The meaning of life is", help='Prompt for text generation')
    parser.add_argument('--model', type=str, choices=['rnn', 'lstm', 'transformer'], default='all',
                        help='Model to use for text generation')

    args = parser.parse_args()

    # If no arguments are provided, print help
    if len(sys.argv) == 1:
        parser.print_help()
        return

    # Run all steps if --all is provided
    if args.all:
        args.tokenize = True
        args.train = True
        args.evaluate = True
        args.generate = True

    # Get base directory for file paths
    base_dir = get_base_dir()
    results_dir = os.path.join(base_dir, 'project_results')
    results_models_dir = os.path.join(results_dir, 'models')
    results_plots_dir = os.path.join(results_dir, 'plots')
    results_texts_dir = os.path.join(results_dir, 'generated_texts')

    # Check if required directories exist, create if not
    for dir_path in ['data', 'project_results', results_models_dir, results_plots_dir, results_texts_dir, 'architecture_diagrams']:
        os.makedirs(f'{base_dir}{dir_path}', exist_ok=True)

    # Step 1: Train the tokenizer
    if args.tokenize:
        print_header("Training the BPE Tokenizer")
        from tokenizer import prepare_data_for_training
        prepare_data_for_training()

    # Step 2: Train the models
    if args.train:
        print_header("Training the Models")
        from train import train_all_models
        train_all_models()

    # Step 3: Evaluate the models
    if args.evaluate:
        print_header("Evaluating the Models")
        from torch.utils.data import DataLoader
        from evaluate import evaluate_models
        evaluate_models()

    # Step 4: Generate text (This part is mostly for demonstration/quick checks)
    if args.generate:
        print_header("Generating Text from Models (Demonstration)")
        from generate import load_model, generate_text
        from tokenizer import TextTokenizer

        # Load tokenizer
        tokenizer_path = os.path.join(base_dir, 'data/bpe_tokenizer.model')
        if not os.path.exists(tokenizer_path):
             print(f"Error: Tokenizer not found at {tokenizer_path}. Please run --tokenize first.")
             return
        tokenizer = TextTokenizer(tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()

        if args.model == 'all':
            models_to_generate = ['rnn', 'lstm', 'transformer']
        else:
            models_to_generate = [args.model]

        for model_type in models_to_generate:
            print(f"\nGenerating text with {model_type.upper()} model:")
            print(f"Prompt: {args.prompt}")
            model_path = os.path.join(results_models_dir, f'{model_type}_model.pth')
            if not os.path.exists(model_path):
                print(f"Model file for {model_type} not found at {model_path}. Please train the model first.")
                continue

            model = load_model(model_type, model_path, vocab_size)
            generated_text = generate_text(model, model_type, tokenizer, args.prompt)
            print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main() 