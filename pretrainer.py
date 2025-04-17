"""
This is a wandb pretrainer.
Borrowed from the network intrusion codebase.

It let's me hyperparameter sweep with wandb.
"""

import os, sys

# Change to directory of the script, whether it's a notebook or a .py file
try:
    file_dir = globals()['_dh'][0]
except:
	file_dir = os.path.dirname(__file__)
 
sys.path.append(os.path.dirname(file_dir))
os.chdir(file_dir)

import argparse
import json
import yaml
import time
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from tqdm import tqdm
from wandb.sdk.wandb_run import Run
from collections import deque

from models import RNNModel, LSTMModel, TransformerModel
from tokenizer import TextDataset, TextTokenizer, tokenize_dataset, get_base_dir
from wandb_manager import WandbTrainingManager


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

class Pretrainer:
    def __init__(
        self, 
        config, 
        premade_run : Run | None = None, 
        locked_params : list[str] | None = None
    ):
        """
        Initialize a model pretrainer with the given configuration.
        
        Args:
            config: Dictionary containing configuration parameters including:
                - model_type: Type of model to train ('rnn', 'lstm', or 'transformer')
                - vocab_size: Size of the vocabulary
                - embedding_dim: Dimension of the embedding vectors
                - hidden_dim: Number of hidden units
                - num_layers: Number of layers in the model
                - dropout: Dropout probability
                - num_heads: Number of attention heads (for transformer only)
                - lr: Learning rate
                - weight_decay: Weight decay for optimizer
                - epochs: Number of training epochs
                - patience: Patience for early stopping
                - save_dir: Directory to save models and plots
                - moving_average_window_size: Window size for moving average calculations
                - gradient_accumulation_steps: Number of steps to accumulate gradients
                - log_interval: Interval for logging metrics
                - use_amp: Whether to use automatic mixed precision
            wandb_manager: WandbTrainingManager instance for logging (optional)
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if "device" in self.config:
            self.device = torch.device(self.config["device"])
        print(f"Using device: {self.device}")
        self._setup()
        self.wandb_manager = WandbTrainingManager(
            self.config,
            self.model,
            premade_run=premade_run,
            locked_params=locked_params
        )
        
    def _setup(self):
        """Set up parameters from config and initialize the model"""
        # Model parameters
        self.model_type = self.config.get('model_type', 'lstm')
        self.vocab_size = self.config.get('vocab_size')
        if self.vocab_size is None:
            raise ValueError("vocab_size must be provided in config")
        
        self.embedding_dim = self.config.get('embedding_dim', 256)
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout = self.config.get('dropout', 0.5)
        self.num_heads = self.config.get('num_heads', 2)
        
        # Training parameters
        self.lr = self.config.get('lr', 0.001)
        self.weight_decay = self.config.get('weight_decay', 1e-5)
        self.epochs = self.config.get('epochs', 30)
        self.patience = self.config.get('patience', 3)
        self.use_amp = self.config.get('use_amp', False)
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        self.moving_average_window_size = self.config.get('moving_average_window_size', 100)
        self.log_interval = self.config.get('log_interval', 10)
        
        # File management
        self.base_dir = get_base_dir()
        self.save_dir = self.config.get('save_dir', os.path.join(self.base_dir, 'project_results'))
        self.models_dir = os.path.join(self.save_dir, 'models')
        self.plots_dir = os.path.join(self.save_dir, 'plots')
        
        
        
        self.tokenizer = TextTokenizer(f'{self.base_dir}data/bpe_tokenizer.model')
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        # Load datasets
        self.train_dataset = TextDataset(
            f'{self.base_dir}data/train.jsonl', 
            max_samples=self.config.get('train_dataset', {}).get('max_samples', None) 
        )
        print(f"Train dataset size: {len(self.train_dataset)}")
        self.val_dataset = TextDataset(
            f'{self.base_dir}data/test.jsonl', 
            max_samples=self.config.get('val_dataset', {}).get('max_samples', None)
        )
        print(f"Val dataset size: {len(self.val_dataset)}")
        
        # Tokenize datasets
        self.train_tokenized = tokenize_dataset(self.train_dataset, self.tokenizer)
        self.val_tokenized = tokenize_dataset(self.val_dataset, self.tokenizer)
        
        # Create data loaders
        self.seq_length = self.config.get('seq_length', 64)  # Sequence length for training
        self.batch_size = self.config.get('batch_size', 128)  # Batch size as recommended
        
        self.train_data = TextGenerationDataset(self.train_tokenized, self.seq_length)
        self.val_data = TextGenerationDataset(self.val_tokenized, self.seq_length)
        
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=collate_batch)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=collate_batch)    
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Metrics tracking
        self.metrics_tracker = {
            "train": {
                "loss": {"epochs": [], "values": []},
                "perplexity": {"epochs": [], "values": []},
                "accuracy": {"epochs": [], "values": []},
            },
            "val": {
                "loss": {"epochs": [], "values": []},
                "perplexity": {"epochs": [], "values": []},
                "accuracy": {"epochs": [], "values": []},
            },
            "test": {
                "loss": {"epochs": [], "values": []},
                "perplexity": {"epochs": [], "values": []},
                "accuracy": {"epochs": [], "values": []},
            }
        }
        
        # Create model based on model_type
        self._create_model()
        
        # Initialize AMP scaler if using mixed precision
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
    def _create_model(self):
        """Create model based on model_type specified in config"""
        if self.model_type == 'rnn':
            self.model = RNNModel(
                self.vocab_size,
                self.embedding_dim,
                self.hidden_dim,
                self.num_layers,
                self.dropout
            )
        elif self.model_type == 'lstm':
            self.model = LSTMModel(
                self.vocab_size,
                self.embedding_dim,
                self.hidden_dim,
                self.num_layers,
                self.dropout
            )
        elif self.model_type == 'transformer':
            self.model = TransformerModel(
                self.vocab_size,
                self.embedding_dim,
                self.hidden_dim,
                self.num_layers,
                self.num_heads,
                self.dropout
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
    def train(self):
        """
        Train the model using the provided data loaders
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data (optional)
            tokenizer: Tokenizer object (optional, for generating examples during training)
            
        Returns:
            history: Dictionary containing training and validation losses
        """
        train_loader = self.train_loader
        val_loader = self.val_loader
        test_loader = self.val_loader
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1, verbose=True
        )
        
        # Initialize variables for training
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        global_step = 0
        
        # Set up validation iteration tracking
        total_val_batches = len(val_loader)
        current_val_batch = 0
        best_perplexity = float('inf')
        
        # Set up moving average window size for validation
        self.val_moving_average_window_size = max(
            5,  # Minimum window size of 5 to ensure some smoothing
            int(self.moving_average_window_size * self.gradient_accumulation_steps)
        )
        
        if self.wandb_manager:
            print(f"Using moving average window of {self.moving_average_window_size} for training metrics")
            print(f"Using moving average window of {self.val_moving_average_window_size} for validation metrics")
            print(f"This averages over approximately {self.moving_average_window_size * self.config.get('batch_size', 32)} samples in both cases")
        
        # Initialize moving averages for metrics
        moving_averages = {
            "train": {
                "loss": deque(maxlen=self.moving_average_window_size),
                "perplexity": deque(maxlen=self.moving_average_window_size),
                "accuracy": deque(maxlen=self.moving_average_window_size),
            },
            "val": {
                "loss": deque(maxlen=self.val_moving_average_window_size),
                "perplexity": deque(maxlen=self.val_moving_average_window_size),
                "accuracy": deque(maxlen=self.val_moving_average_window_size),
            }
        }
        
        # For tracking gradient statistics
        grad_norms = deque(maxlen=self.moving_average_window_size)
        
        # For tracking timing statistics
        epoch_start_time = time.time()
        batch_start_time = time.time()
        forward_times = deque(maxlen=100)
        backward_times = deque(maxlen=100)
        optimizer_times = deque(maxlen=100)
        
        # Start training
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.model.train()
            total_tokens = 0
            correct_predictions = 0
            running_loss = []
            optimizer.zero_grad(set_to_none=True)
            
            # For continuous validation tracking
            epoch_val_loss = 0
            epoch_val_tokens = 0
            epoch_val_correct = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                batch_start_time = time.time()
                # Calculate current epoch progress
                epoch_progress = epoch + (batch_idx / len(train_loader))
                global_step += 1
                
                # Move tensors to GPU
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Initialize attention mask (all ones for non-padded tokens)
                attention_mask = torch.ones_like(targets, dtype=torch.bool)
                
                # Forward pass timing
                forward_start_time = time.time()
                
                # Forward pass with mixed precision if configured
                if self.use_amp:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        # Initialize hidden state for RNN/LSTM for each batch
                        hidden = None
                        
                        # Forward pass
                        if self.model_type in ["rnn", "lstm"]:
                            outputs, hidden = self.model(inputs, hidden)
                        else:  # Transformer
                            outputs = self.model(inputs)
                        
                        # Reshape outputs and targets for loss calculation
                        outputs_flat = outputs.reshape(-1, outputs.shape[-1])
                        targets_flat = targets.reshape(-1)
                        mask_flat = attention_mask.reshape(-1)
                        
                        # Compute loss on non-padding tokens only
                        active_loss = mask_flat.bool()
                        active_logits = outputs_flat[active_loss]
                        active_targets = targets_flat[active_loss]
                        
                        # Calculate loss
                        loss = criterion(active_logits, active_targets)
                        scaled_loss = loss / self.gradient_accumulation_steps
                        
                        # Calculate accuracy
                        with torch.no_grad():
                            pred_tokens = torch.argmax(active_logits, dim=1)
                            correct = (pred_tokens == active_targets).sum().item()
                            total = active_targets.numel()
                            correct_predictions += correct
                            total_tokens += total
                else:
                    # Initialize hidden state for RNN/LSTM for each batch
                    hidden = None
                    
                    # Forward pass
                    if self.model_type in ["rnn", "lstm"]:
                        outputs, hidden = self.model(inputs, hidden)
                    else:  # Transformer
                        outputs = self.model(inputs)
                    
                    # Reshape outputs and targets for loss calculation
                    outputs_flat = outputs.reshape(-1, outputs.shape[-1])
                    targets_flat = targets.reshape(-1)
                    mask_flat = attention_mask.reshape(-1)
                    
                    # Compute loss on non-padding tokens only
                    active_loss = mask_flat.bool()
                    active_logits = outputs_flat[active_loss]
                    active_targets = targets_flat[active_loss]
                    
                    # Calculate loss
                    loss = criterion(active_logits, active_targets)
                    scaled_loss = loss / self.gradient_accumulation_steps
                    
                    # Calculate accuracy
                    with torch.no_grad():
                        pred_tokens = torch.argmax(active_logits, dim=1)
                        correct = (pred_tokens == active_targets).sum().item()
                        total = active_targets.numel()
                        correct_predictions += correct
                        total_tokens += total
                
                # Record forward pass time
                forward_time = time.time() - forward_start_time
                forward_times.append(forward_time)
                
                # Backward pass timing
                backward_start_time = time.time()
                
                # Backward pass with mixed precision if configured
                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                # Record backward pass time
                backward_time = time.time() - backward_start_time
                backward_times.append(backward_time)
                
                running_loss.append(scaled_loss.item() * self.gradient_accumulation_steps)
                
                # Update optimizer
                if ((batch_idx + 1) % self.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                    # Optimizer step timing
                    optimizer_start_time = time.time()
                    
                    # Calculate gradient norm for tracking
                    with torch.no_grad():
                        grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) 
                                                          for p in self.model.parameters() if p.grad is not None]), 2)
                        grad_norms.append(grad_norm.item())
                    
                    # Perform optimizer step with gradient clipping
                    if self.use_amp:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                    
                    # Record optimizer step time
                    optimizer_time = time.time() - optimizer_start_time
                    optimizer_times.append(optimizer_time)
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Continuous validation evaluation
                    # Perform validation on small chunks after weights are updated
                    self.model.eval()
                    num_val_batches_to_process = 1
                    
                    with torch.no_grad():
                        for val_batch_idx in range(num_val_batches_to_process):
                            # Get the validation batch circularly
                            val_iter = iter(val_loader)
                            try:
                                val_batch_data = next(val_iter)
                            except StopIteration:
                                val_iter = iter(val_loader)
                                val_batch_data = next(val_iter)
                            
                            current_val_batch = (current_val_batch + 1) % total_val_batches
                            
                            val_inputs, val_targets = val_batch_data
                            val_inputs = val_inputs.to(self.device)
                            val_targets = val_targets.to(self.device)
                            
                            # Initialize validation attention mask (all ones for non-padded tokens)
                            val_attention_mask = torch.ones_like(val_targets, dtype=torch.bool)
                            
                            # Initialize hidden state for RNN/LSTM
                            val_hidden = None
                            
                            # Forward pass
                            if self.use_amp:
                                with torch.amp.autocast('cuda', dtype=torch.float16):
                                    if self.model_type in ["rnn", "lstm"]:
                                        val_outputs, val_hidden = self.model(val_inputs, val_hidden)
                                    else:  # Transformer
                                        val_outputs = self.model(val_inputs)
                            else:
                                if self.model_type in ["rnn", "lstm"]:
                                    val_outputs, val_hidden = self.model(val_inputs, val_hidden)
                                else:  # Transformer
                                    val_outputs = self.model(val_inputs)
                            
                            # Reshape outputs and targets for loss calculation
                            val_outputs_flat = val_outputs.reshape(-1, val_outputs.shape[-1])
                            val_targets_flat = val_targets.reshape(-1)
                            val_mask_flat = val_attention_mask.reshape(-1)
                            
                            # Compute loss on non-padding tokens only
                            val_active_loss = val_mask_flat.bool()
                            val_active_logits = val_outputs_flat[val_active_loss]
                            val_active_targets = val_targets_flat[val_active_loss]
                            
                            if val_active_targets.numel() > 0:
                                # Calculate loss
                                val_loss = criterion(val_active_logits, val_active_targets)
                                
                                # Calculate accuracy
                                val_pred_tokens = torch.argmax(val_active_logits, dim=1)
                                val_correct = (val_pred_tokens == val_active_targets).sum().item()
                                
                                # Update metrics
                                epoch_val_loss += val_loss.item() * val_active_targets.numel()
                                epoch_val_correct += val_correct
                                epoch_val_tokens += val_active_targets.numel()
                    
                    self.model.train()
                    
                    # Calculate and record continuous validation metrics
                    if epoch_val_tokens > 0:
                        val_avg_loss = epoch_val_loss / epoch_val_tokens
                        val_accuracy = epoch_val_correct / epoch_val_tokens
                        val_perplexity = torch.exp(torch.tensor(val_avg_loss)).item()
                        
                        # Update moving averages for validation
                        moving_averages["val"]["loss"].append(val_avg_loss)
                        moving_averages["val"]["accuracy"].append(val_accuracy)
                        moving_averages["val"]["perplexity"].append(val_perplexity)
                        
                        # Calculate moving average values
                        val_ma_loss = sum(moving_averages["val"]["loss"]) / len(moving_averages["val"]["loss"])
                        val_ma_accuracy = sum(moving_averages["val"]["accuracy"]) / len(moving_averages["val"]["accuracy"])
                        val_ma_perplexity = sum(moving_averages["val"]["perplexity"]) / len(moving_averages["val"]["perplexity"])
                        
                        # Record validation metrics with moving average
                        self.metrics_tracker["val"]["loss"]["epochs"].append(epoch_progress)
                        self.metrics_tracker["val"]["loss"]["values"].append(float(val_ma_loss))
                        self.metrics_tracker["val"]["perplexity"]["epochs"].append(epoch_progress)
                        self.metrics_tracker["val"]["perplexity"]["values"].append(float(val_ma_perplexity))
                        self.metrics_tracker["val"]["accuracy"]["epochs"].append(epoch_progress)
                        self.metrics_tracker["val"]["accuracy"]["values"].append(float(val_ma_accuracy))
                        
                        # Log validation metrics to wandb
                        if self.wandb_manager:
                            self.wandb_manager.log_validation_metrics(
                                {
                                    "loss": val_ma_loss,
                                    "perplexity": val_ma_perplexity,
                                    "accuracy": val_ma_accuracy
                                },
                                global_step,
                                epoch_progress
                            )
                
                # Calculate current batch metrics
                batch_loss = scaled_loss.item() * self.gradient_accumulation_steps
                batch_accuracy = correct / total if total > 0 else 0
                batch_perplexity = torch.exp(torch.tensor(batch_loss)).item()
                
                # Calculate total batch time
                batch_time = time.time() - batch_start_time
                
                # Update moving averages for training metrics
                moving_averages["train"]["loss"].append(batch_loss)
                moving_averages["train"]["accuracy"].append(batch_accuracy)
                moving_averages["train"]["perplexity"].append(batch_perplexity)
                
                # Calculate moving average values
                ma_loss = sum(moving_averages["train"]["loss"]) / len(moving_averages["train"]["loss"])
                ma_accuracy = sum(moving_averages["train"]["accuracy"]) / len(moving_averages["train"]["accuracy"])
                ma_perplexity = sum(moving_averages["train"]["perplexity"]) / len(moving_averages["train"]["perplexity"])
                
                # Record training metrics with moving average
                self.metrics_tracker["train"]["loss"]["epochs"].append(epoch_progress)
                self.metrics_tracker["train"]["loss"]["values"].append(float(ma_loss))
                self.metrics_tracker["train"]["perplexity"]["epochs"].append(epoch_progress)
                self.metrics_tracker["train"]["perplexity"]["values"].append(float(ma_perplexity))
                self.metrics_tracker["train"]["accuracy"]["epochs"].append(epoch_progress)
                self.metrics_tracker["train"]["accuracy"]["values"].append(float(ma_accuracy))
                
                # Log training metrics to wandb
                if self.wandb_manager and (batch_idx + 1) % self.log_interval == 0:
                    # Calculate average timing statistics
                    avg_forward_time = sum(forward_times) / len(forward_times) if forward_times else 0
                    avg_backward_time = sum(backward_times) / len(backward_times) if backward_times else 0
                    avg_optimizer_time = sum(optimizer_times) / len(optimizer_times) if optimizer_times else 0
                    
                    # Get GPU memory usage
                    if torch.cuda.is_available():
                        gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
                        gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # Convert to GB
                        gpu_max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)   # Convert to GB
                    else:
                        gpu_memory_allocated = 0
                        gpu_memory_reserved = 0
                        gpu_max_memory = 0
                    
                    # Get average gradient norm
                    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
                    
                    # Sequence lengths statistics
                    avg_seq_length = inputs.shape[1]
                    tokens_per_second = total / batch_time if batch_time > 0 else 0
                    
                    # Create metrics dictionary for logging
                    train_metrics = {
                        "loss": ma_loss,
                        "perplexity": ma_perplexity,
                        "accuracy": ma_accuracy,
                        "total_tokens": total,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "gpu_memory_allocated": gpu_memory_allocated,
                        "gpu_memory_reserved": gpu_memory_reserved,
                        "gpu_max_memory": gpu_max_memory,
                        "tokens_per_second": tokens_per_second,
                        "avg_sequence_length": avg_seq_length,
                        "forward_time_ms": avg_forward_time * 1000,
                        "backward_time_ms": avg_backward_time * 1000,
                        "optimizer_time_ms": avg_optimizer_time * 1000,
                        "batch_time_ms": batch_time * 1000,
                        "grad_norm": avg_grad_norm
                    }
                    
                    # Log metrics with WandbTrainingManager
                    self.wandb_manager.log_train_batch_metrics(train_metrics, global_step, epoch_progress)
                
                # Update progress bar with moving averages
                if epoch_val_tokens > 0:
                    progress_bar.set_postfix({
                        'loss': f'{ma_loss:.4f}',
                        'ppl': f'{ma_perplexity:.2f}',
                        'acc': f'{ma_accuracy:.4f}',
                        'val_loss': f'{val_ma_loss:.4f}'
                    })
                else:
                    progress_bar.set_postfix({
                        'loss': f'{ma_loss:.4f}',
                        'ppl': f'{ma_perplexity:.2f}',
                        'acc': f'{ma_accuracy:.4f}'
                    })
            
            # End of epoch timing
            epoch_time = time.time() - epoch_start_time
            
            # Evaluate on the full validation set at the end of each epoch
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(self.device), val_targets.to(self.device)
                    val_hidden = None
                    
                    # Forward pass
                    if self.model_type in ["rnn", "lstm"]:
                        val_outputs, val_hidden = self.model(val_inputs, val_hidden)
                    else:  # Transformer
                        val_outputs = self.model(val_inputs)
                    
                    # Reshape outputs and targets for loss calculation
                    val_outputs = val_outputs.reshape(-1, val_outputs.shape[-1])
                    val_targets = val_targets.reshape(-1)
                    
                    # Calculate loss
                    batch_loss = criterion(val_outputs, val_targets)
                    val_loss += batch_loss.item() * val_targets.size(0)
                    
                    # Calculate accuracy
                    pred = val_outputs.argmax(dim=1)
                    val_correct += (pred == val_targets).sum().item()
                    val_total += val_targets.size(0)
            
            # Calculate average validation loss and metrics
            avg_val_loss = val_loss / val_total
            val_accuracy = val_correct / val_total
            val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
            
            # Add to validation losses
            val_losses.append(avg_val_loss)
            
            # Print epoch results
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print(f"Time: {epoch_time:.2f}s, Train Loss: {ma_loss:.4f}, Train Accuracy: {ma_accuracy:.4f}, Train Perplexity: {ma_perplexity:.2f}")
            print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Perplexity: {val_perplexity:.2f}")
            
            # Log epoch metrics to wandb
            if self.wandb_manager:
                self.wandb_manager.log_test_metrics(
                    {
                        "loss": avg_val_loss,
                        "perplexity": val_perplexity,
                        "accuracy": val_accuracy
                    },
                    epoch + 1,
                    global_step,
                    epoch_time
                )
            
            # Update learning rate based on validation loss
            scheduler.step(avg_val_loss)
            
            # Check if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save the best model
                self.save_model()
                print(f"Model saved with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"Validation loss didn't improve. Patience: {patience_counter}/{self.patience}")
            
            # Early stopping
            if patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
        
        # Load the best model
        self.load_model()
        
        # Evaluate on test set if provided
        if test_loader is not None:
            print("\nEvaluating on test set...")
            self.model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for test_inputs, test_targets in test_loader:
                    test_inputs, test_targets = test_inputs.to(self.device), test_targets.to(self.device)
                    test_hidden = None
                    
                    # Forward pass
                    if self.model_type in ["rnn", "lstm"]:
                        test_outputs, test_hidden = self.model(test_inputs, test_hidden)
                    else:  # Transformer
                        test_outputs = self.model(test_inputs)
                    
                    # Reshape outputs and targets for loss calculation
                    test_outputs = test_outputs.reshape(-1, test_outputs.shape[-1])
                    test_targets = test_targets.reshape(-1)
                    
                    # Calculate loss
                    batch_loss = criterion(test_outputs, test_targets)
                    test_loss += batch_loss.item() * test_targets.size(0)
                    
                    # Calculate accuracy
                    pred = test_outputs.argmax(dim=1)
                    test_correct += (pred == test_targets).sum().item()
                    test_total += test_targets.size(0)
            
            # Calculate average test loss and metrics
            avg_test_loss = test_loss / test_total
            test_accuracy = test_correct / test_total
            test_perplexity = torch.exp(torch.tensor(avg_test_loss)).item()
            
            print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Perplexity: {test_perplexity:.2f}")
            
            # Log final test metrics to wandb
            if self.wandb_manager:
                test_results = {
                    "loss": avg_test_loss,
                    "perplexity": test_perplexity,
                    "accuracy": test_accuracy
                }
                self.wandb_manager.log_test_metrics(test_results, self.epochs, global_step, 0)
                
                # Save metrics tracker to JSON if wandb is available
                self.wandb_manager.save_metrics_tracker(self.metrics_tracker)
        
        # Prepare training history
        history = {
            'train_loss': self.metrics_tracker["train"]["loss"]["values"],
            'val_loss': self.metrics_tracker["val"]["loss"]["values"],
        }
        
        # Plot history
        self.plot_training_history(history)
        
        # Save history
        history_save_path = os.path.join(self.save_dir, f'{self.model_type}_history.json')
        with open(history_save_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        
        self.run_id = self.wandb_manager.run.id if self.wandb_manager.run else None # Store run id
        
        return history
    
    def save_model(self, custom_path=None):
        """
        Save the model to the specified path or default path
        
        Args:
            custom_path: Optional custom path to save the model
        """
        if custom_path:
            model_save_path = custom_path
        else:
            model_save_path = os.path.join(self.models_dir, f'{self.model_type}_model.pth')
        
        torch.save(self.model.state_dict(), model_save_path)
        return model_save_path
        
    def load_model(self, custom_path=None):
        """
        Load the model from the specified path or default path
        
        Args:
            custom_path: Optional custom path to load the model from
        """
        if custom_path:
            model_load_path = custom_path
        else:
            model_load_path = os.path.join(self.models_dir, f'{self.model_type}_model.pth')
        
        if os.path.exists(model_load_path):
            self.model.load_state_dict(torch.load(model_load_path))
            print(f"Loaded model from {model_load_path}")
            return True
        else:
            print(f"No model found at {model_load_path}")
            return False
        
    def plot_training_history(self, history):
        """
        Plot training and validation loss curves
        
        Args:
            history: Dictionary containing training and validation losses
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'Training and Validation Loss for {self.model_type.upper()} Model')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_save_path = os.path.join(self.plots_dir, f'{self.model_type}_loss_curve.png')
        plt.savefig(plot_save_path)
        plt.close()
        print(f"Loss curve saved to {plot_save_path}")
        
    def generate_text(self, tokenizer, prompt_text, max_seq_length=100, temperature=1.0):
        """
        Generate text using the trained model
        
        Args:
            tokenizer: Tokenizer object for encoding/decoding text
            prompt_text: Text to start generation with
            max_seq_length: Maximum sequence length to generate
            temperature: Temperature for sampling
            
        Returns:
            generated_text: The generated text
        """
        return self.model.prompt(tokenizer, prompt_text, max_seq_length, temperature)


def main():
    parser = argparse.ArgumentParser(description="Train a Language Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/transformer_sweep.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    
    # Load configuration from YAML file
    try:
        with open(args.config, 'r') as f:
            loaded_config = yaml.safe_load(f)
        print("Loaded configuration from:", args.config)
        print(json.dumps(loaded_config, indent=4))
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        return
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return
    

    # --- Seed everything for reproducibility ---
    # random.seed(config.get('seed', 42))
    # np.random.seed(config.get('seed', 42))
    # torch.manual_seed(config.get('seed', 42))
    # torch.cuda.manual_seed_all(config.get('seed', 42))
    # torch.backends.cudnn.deterministic = True # May impact performance
    # torch.backends.cudnn.benchmark = False    # May impact performance
    # -----------------------------------------


    # Initialize and run the trainer
    
    config_base_reference = deepcopy(loaded_config)
    
    if not loaded_config.get('wandb_project'):
        loaded_config['wandb_project'] = "CSC7809 Project 2" + \
            (" [DEBUG]" if loaded_config.get('debug_mode', False) else "")
    
    all_sweep_run_ids = []
    project_name = loaded_config["wandb_project"]
    
    def sweep_objective(hyper_parameters : dict, premade_run : Run):
        nonlocal config_base_reference, project_name, all_sweep_run_ids
        
        print("hyper_parameters:", type(hyper_parameters), hyper_parameters)
        
        for key, value in hyper_parameters.items():
            print(key, "->", value)

        # hyper_params_2 = json.loads(str(hyper_parameters))
        # hyper_params_2 = deepcopy(hyper_parameters) # Deepcopy on hyper_parameters triggers a maximum recursion depth  error
        new_config = deepcopy(config_base_reference)
        new_config.update(hyper_parameters)
        
        trainer = Pretrainer(
            new_config,
            premade_run=premade_run,
            locked_params=list(hyper_parameters.keys())
        )
        trainer.train()
        
        
        get_run_id = trainer.run_id
        
        if not get_run_id is None:
            all_sweep_run_ids.append(get_run_id)
        
        if project_name is None:
            project_name = trainer.wandb_manager.config["wandb_project"]
        
        del trainer
    
    def sweep_main():
        nonlocal project_name
        premade_run = wandb.init(project=project_name)
        
        sweep_params = wandb.config
        
        sweep_objective(sweep_params, premade_run=premade_run)
        
    # Perform a sweep
    if "sweep" in loaded_config:
        
        sweep_config = loaded_config["sweep"]
        sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
        sweep_count = None
        
        if "total_runs" in sweep_config:
            sweep_count = sweep_config.pop("total_runs")
        
        wandb.agent(sweep_id, function=sweep_main, count=sweep_count)
    
    # If we're not performing sweeps, we can just run the trainer directly
    else:
        trainer = Pretrainer(loaded_config)
        trainer.train()
    
    print("--- Pretraining Finished ---")

if __name__ == "__main__":
    main()