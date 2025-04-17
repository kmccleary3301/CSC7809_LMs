import os
import json
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader

def get_base_dir():
    """Get the base directory for file paths"""
    # Check if we're running from inside final_submission directory
    if os.path.basename(os.getcwd()) == 'final_submission':
        return ""
    else:
        return "final_submission/"

class TextDataset(Dataset):
    def __init__(
        self, 
        file_path, 
        max_length=512,
        max_samples=None
    ):
        self.max_length = max_length
        self.texts = []
        
        # Load JSONL file and extract text
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_samples is not None and len(self.texts) >= max_samples:
                    break
                data = json.loads(line)
                if 'text' in data:
                    self.texts.append(data['text'])
                elif 'prompt' in data and 'completion' in data:
                    # Combine prompt and completion
                    text = data['prompt'] + ' ' + data['completion']
                    self.texts.append(text)
                else:
                    # Skip lines with unknown format
                    continue
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]
    
    def save_for_tokenizer_training(self, output_path):
        """Save all texts to a single file for tokenizer training"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in self.texts:
                f.write(text + '\n')


def train_tokenizer(input_file, model_prefix, vocab_size=10000, character_coverage=1.0):
    """Train a SentencePiece BPE tokenizer"""
    train_args = [
        f'--input={input_file}',
        f'--model_prefix={model_prefix}',
        f'--vocab_size={vocab_size}',
        f'--model_type=bpe',
        f'--character_coverage={character_coverage}',
        '--pad_id=0',
        '--unk_id=1',
        '--bos_id=2',
        '--eos_id=3',
        '--pad_piece=[PAD]',
        '--unk_piece=[UNK]',
        '--bos_piece=[BOS]',
        '--eos_piece=[EOS]',
    ]
    spm.SentencePieceTrainer.train(' '.join(train_args))
    print(f"Tokenizer trained and saved with prefix: {model_prefix}")


class TextTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        # Special token IDs
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        
        # Vocabulary size
        self.vocab_size = self.sp.get_piece_size()
    
    def encode(self, text, add_special_tokens=True):
        """Convert text to token IDs"""
        if add_special_tokens:
            return [self.bos_id] + self.sp.encode_as_ids(text) + [self.eos_id]
        else:
            return self.sp.encode_as_ids(text)
    
    def decode(self, token_ids):
        """Convert token IDs back to text"""
        # Filter out special tokens if present
        filtered_ids = [id for id in token_ids if id != self.pad_id and id != self.bos_id and id != self.eos_id]
        return self.sp.decode_ids(filtered_ids)
    
    def get_vocab_size(self):
        """Return the vocabulary size"""
        return self.vocab_size


def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenize a dataset of texts and convert to PyTorch tensors"""
    tokenized_texts = []
    
    for text in dataset:
        token_ids = tokenizer.encode(text)
        
        # Truncate if necessary
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Convert to PyTorch tensor
        tokenized_texts.append(torch.tensor(token_ids, dtype=torch.long))
    
    return tokenized_texts


def prepare_data_for_training():
    base_dir = get_base_dir()
    
    # Create temp directory if it doesn't exist
    os.makedirs(f'{base_dir}data/temp', exist_ok=True)
    
    # Load raw training data
    train_dataset = TextDataset(f'{base_dir}data/train.jsonl')
    
    # Save text to temporary file for tokenizer training
    temp_file = f'{base_dir}data/temp/train_texts.txt'
    train_dataset.save_for_tokenizer_training(temp_file)
    
    # Train tokenizer
    model_prefix = f'{base_dir}data/bpe_tokenizer'
    train_tokenizer(temp_file, model_prefix)
    
    # Load the trained tokenizer
    tokenizer = TextTokenizer(f'{model_prefix}.model')
    
    print(f"Tokenizer trained with vocabulary size: {tokenizer.get_vocab_size()}")
    return tokenizer


if __name__ == "__main__":
    tokenizer = prepare_data_for_training()
    print(f"Tokenizer trained successfully with vocab size: {tokenizer.get_vocab_size()}") 