import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from tokenizer import TextDataset, TextTokenizer, tokenize_dataset, get_base_dir
from models import RNNModel, LSTMModel, TransformerModel

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TextGenerationDataset(Dataset):
    def __init__(self, tokenized_texts, seq_length=64):
        self.tokenized_texts = tokenized_texts
        self.seq_length = seq_length
        
        # Create input-output pairs for language modeling
        self.input_sequences = []
        self.target_sequences = []
        
        for tokens in self.tokenized_texts:
            if len(tokens) <= 1:  # Skip texts that are too short
                continue
            
            # Create overlapping sequences
            for i in range(0, len(tokens) - 1, seq_length // 2):
                if i + seq_length + 1 > len(tokens):
                    # Add the last sequence if it's not long enough
                    if len(tokens) > seq_length:
                        input_seq = tokens[:seq_length]
                        target_seq = tokens[1:seq_length + 1]
                        self.input_sequences.append(input_seq)
                        self.target_sequences.append(target_seq)
                    break
                
                input_seq = tokens[i:i + seq_length]
                target_seq = tokens[i + 1:i + seq_length + 1]
                self.input_sequences.append(input_seq)
                self.target_sequences.append(target_seq)
    
    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self, idx):
        return self.input_sequences[idx], self.target_sequences[idx]


def collate_batch(batch):
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets


def train_model(model, train_loader, val_loader, tokenizer, model_type, epochs=30, lr=0.001, patience=3, weight_decay=1e-5):
    """Train the model and return training history"""
    base_dir = get_base_dir()
    results_models_dir = os.path.join(base_dir, 'project_results/models')
    os.makedirs(results_models_dir, exist_ok=True)
    
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )
    
    # Initialize variables for training
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Start training
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        
        # Training loop
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)")
        for i, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Initialize hidden state for RNN/LSTM for each batch
            hidden = None
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            if model_type in ["rnn", "lstm"]:
                # Pass hidden=None, model.forward will init based on batch size
                outputs, hidden = model(inputs, hidden)
                
                # Reshape outputs and targets for loss calculation
                outputs = outputs.reshape(-1, outputs.shape[-1])
                targets = targets.reshape(-1)
            else:  # Transformer
                outputs = model(inputs)
                
                # Reshape outputs and targets for loss calculation
                outputs = outputs.reshape(-1, outputs.shape[-1])
                targets = targets.reshape(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update progress bar
            epoch_train_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_train_loss / (i + 1)})
        
        # Calculate average training loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Validation)")
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Initialize hidden state for each batch
                hidden_val = None
                
                # Forward pass
                if model_type in ["rnn", "lstm"]:
                    # Pass hidden_val=None, model.forward will initialize correctly based on batch size
                    outputs, hidden_val = model(inputs, hidden_val)
                    
                    # Reshape outputs and targets for loss calculation
                    outputs = outputs.reshape(-1, outputs.shape[-1])
                    targets = targets.reshape(-1)
                else:  # Transformer
                    outputs = model(inputs)
                    
                    # Reshape outputs and targets for loss calculation
                    outputs = outputs.reshape(-1, outputs.shape[-1])
                    targets = targets.reshape(-1)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({"loss": epoch_val_loss / (i + 1)})
        
        # Calculate average validation loss for the epoch
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        
        # Check if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save the best model
            model_save_path = os.path.join(results_models_dir, f'{model_type}_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss didn't improve. Patience: {patience_counter}/{patience}")
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    # Load the best model
    best_model_path = os.path.join(results_models_dir, f'{model_type}_model.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    else:
        print("No best model was saved during training.")
    
    # Prepare training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    
    return model, history


def plot_training_history(history, model_type):
    """Plot training and validation loss curves"""
    base_dir = get_base_dir()
    results_plots_dir = os.path.join(base_dir, 'project_results/plots')
    os.makedirs(results_plots_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss for {model_type.upper()} Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_save_path = os.path.join(results_plots_dir, f'{model_type}_loss_curve.png')
    plt.savefig(plot_save_path)
    plt.close()
    print(f"Loss curve saved to {plot_save_path}")


def train_all_models():
    """Train RNN, LSTM, and Transformer models"""
    base_dir = get_base_dir()
    results_dir = os.path.join(base_dir, 'project_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load the tokenizer
    tokenizer = TextTokenizer(f'{base_dir}data/bpe_tokenizer.model')
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")
    
    # Load datasets
    train_dataset = TextDataset(f'{base_dir}data/train.jsonl')
    val_dataset = TextDataset(f'{base_dir}data/test.jsonl')  # Using test as validation for simplicity
    
    # Tokenize datasets
    train_tokenized = tokenize_dataset(train_dataset, tokenizer)
    val_tokenized = tokenize_dataset(val_dataset, tokenizer)
    
    # Create data loaders
    seq_length = 64  # Sequence length for training
    batch_size = 128  # Batch size as recommended
    
    train_data = TextGenerationDataset(train_tokenized, seq_length)
    val_data = TextGenerationDataset(val_tokenized, seq_length)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_batch)
    
    print(f"Training dataset size: {len(train_data)}")
    print(f"Validation dataset size: {len(val_data)}")
    
    # Model hyperparameters
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2
    dropout = 0.5
    weight_decay = 1e-5  # <--- Easy to tune
    
    all_histories = {}
    
    # Train RNN model
    print("\n===== Training RNN Model =====")
    rnn_model = RNNModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    rnn_model, rnn_history = train_model(rnn_model, train_loader, val_loader, tokenizer, "rnn", weight_decay=weight_decay)
    plot_training_history(rnn_history, "rnn")
    all_histories['rnn'] = rnn_history
    
    # Train LSTM model
    print("\n===== Training LSTM Model =====")
    lstm_model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    lstm_model, lstm_history = train_model(lstm_model, train_loader, val_loader, tokenizer, "lstm", weight_decay=weight_decay)
    plot_training_history(lstm_history, "lstm")
    all_histories['lstm'] = lstm_history
    
    # Train Transformer model
    print("\n===== Training Transformer Model =====")
    transformer_model = TransformerModel(
        vocab_size, embedding_dim, hidden_dim, 
        num_layers=num_layers, 
        num_heads=2,  # 2 attention heads as specified in assignment
        dropout=dropout
    )
    transformer_model, transformer_history = train_model(transformer_model, train_loader, val_loader, tokenizer, "transformer", weight_decay=weight_decay)
    plot_training_history(transformer_history, "transformer")
    all_histories['transformer'] = transformer_history
    
    # Save combined training history
    history_save_path = os.path.join(results_dir, 'training_history.json')
    with open(history_save_path, 'w') as f:
        json.dump(all_histories, f, indent=2)
    print(f"Combined training history saved to {history_save_path}")


if __name__ == "__main__":
    train_all_models() 