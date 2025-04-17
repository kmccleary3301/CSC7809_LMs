import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os # Added for path joining


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.5):
        super(RNNModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Embedding layer: maps token IDs to embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer(s)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected output layer: maps hidden states to token probabilities
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        # Initialize embedding weights with small random values
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        # Initialize linear layer weights with small random values
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x, hidden=None):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of token IDs (batch_size, seq_len)
            hidden: Initial hidden state (optional)
            
        Returns:
            output: Predicted token probabilities (batch_size, seq_len, vocab_size)
            hidden: Final hidden state
        """
        # Get batch size and sequence length
        batch_size, seq_len = x.size()
        
        # Determine the device from the input tensor
        device = x.device
        
        # If no hidden state is provided, initialize with zeros on the correct device
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)
        
        # Convert token IDs to embeddings
        embeds = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Process through RNN
        # Ensure hidden state is on the same device as embeds
        output, hidden = self.rnn(embeds, hidden.to(device))  # output: (batch_size, seq_len, hidden_dim)
        
        # Map hidden states to token probabilities
        output = self.fc(output)  # (batch_size, seq_len, vocab_size)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state with zeros on the specified device"""
        # Use a parameter's device to ensure consistency
        # device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
    
    def predict_next_token(self, x, hidden=None, temperature=1.0):
        """
        Predict the next token given the input sequence
        
        Args:
            x: Input tensor of token IDs (batch_size, seq_len)
            hidden: Hidden state (optional)
            temperature: Temperature for sampling (optional)
            
        Returns:
            next_token: Predicted next token ID
            hidden: Updated hidden state
        """
        # Get predictions
        output, hidden = self.forward(x, hidden)
        
        # Get probabilities for the last token in the sequence
        logits = output[:, -1, :] / temperature
        
        # Get the token with the highest probability
        next_token = torch.argmax(logits, dim=-1)
        
        return next_token, hidden
    
    def prompt(self, tokenizer, text, max_seq_length=100, temperature=1.0):
        """
        Generate text given a prompt
        
        Args:
            tokenizer: Tokenizer object for encoding/decoding text
            text: The prompt text
            max_seq_length: Maximum sequence length to generate
            temperature: Temperature for sampling (optional)
            
        Returns:
            generated_text: The generated text
        """
        self.eval()  # Set the model to evaluation mode
        device = next(self.parameters()).device # Get device from model
        
        # Tokenize the prompt
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens = tokens[:-1]  # Remove EOS token to keep generating
        
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        generated_tokens = tokens.copy()
        
        # Initialize hidden state
        hidden = None
        
        # Generate text until EOS token or max length
        with torch.no_grad(): # Disable gradient calculation for generation
            while len(generated_tokens) < max_seq_length:
                # Get predictions
                output, hidden = self.forward(input_tensor, hidden)
                
                # Sample next token
                logits = output[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                # For highest probability (undergrad version)
                # next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                
                # Add the predicted token to the generated tokens
                next_token_item = next_token.item()
                generated_tokens.append(next_token_item)
                
                # If EOS token is generated, stop
                if next_token_item == tokenizer.eos_id:
                    break
                
                # Update input tensor for next iteration
                input_tensor = next_token.to(device)
        
        # Decode the generated tokens
        generated_text = tokenizer.decode(generated_tokens)
        
        return generated_text


class LSTMModel(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embedding_dim, 
        hidden_dim, 
        num_layers=1, 
        dropout=0.5
    ):
        """
        LSTM-based language model with embedding, dropout, layer normalization, and output layer.
        :param vocab_size: Size of the vocabulary
        :param embedding_dim: Dimension of the embedding vectors
        :param hidden_dim: Number of hidden units in the LSTM
        :param num_layers: Number of LSTM layers
        :param dropout: Dropout probability
        """
        super(LSTMModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Dropout after embedding
        self.dropout = nn.Dropout(dropout)
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        # Layer normalization after LSTM
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights for embedding and output layers."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, hidden=None):
        """
        Forward pass through the LSTM model.
        :param x: Input tensor of token IDs (batch_size, seq_len)
        :param hidden: Tuple of (hidden state, cell state) (optional)
        :return: Output logits (batch_size, seq_len, vocab_size), hidden state tuple
        """
        batch_size, seq_len = x.size()
        device = x.device
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)
        # Embedding + dropout
        embeds = self.dropout(self.embedding(x))
        # LSTM
        hidden = (hidden[0].to(device), hidden[1].to(device))
        output, hidden = self.lstm(embeds, hidden)
        # Layer normalization
        output = self.layer_norm(output)
        # Output layer
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size, device):
        """
        Initializes hidden and cell states to zeros on the correct device.
        :param batch_size: Number of samples in the batch
        :param device: Device to create tensors on
        :return: Tuple (h0, c0)
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    def predict_next_token(self, x, hidden=None, temperature=1.0):
        output, hidden = self.forward(x, hidden)
        logits = output[:, -1, :] / temperature
        next_token = torch.argmax(logits, dim=-1)
        return next_token, hidden

    def prompt(self, tokenizer, text, max_seq_length=100, temperature=1.0):
        """
        Generate text given a prompt.
        :param tokenizer: Tokenizer object
        :param text: Prompt text
        :param max_seq_length: Maximum sequence length to generate
        :param temperature: Sampling temperature
        :return: Generated text string
        """
        self.eval()
        device = next(self.parameters()).device
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens = tokens[:-1]
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
        generated_tokens = tokens.copy()
        hidden = None
        with torch.no_grad():
            while len(generated_tokens) < max_seq_length:
                output, hidden = self.forward(input_tensor, hidden)
                logits = output[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token_item = next_token.item()
                generated_tokens.append(next_token_item)
                if next_token_item == tokenizer.eos_id:
                    break
                input_tensor = next_token.to(device)
        generated_text = tokenizer.decode(generated_tokens)
        return generated_text


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        # pe shape: (1, max_len, d_model)
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


class TransformerModel(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embedding_dim, 
        hidden_dim, 
        num_layers=2, 
        num_heads=2, 
        dropout=0.5, 
        max_seq_len=512
    ):
        super(TransformerModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Embedding layer: maps token IDs to embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, max_seq_len)
        
        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Fully connected output layer: maps hidden states to token probabilities
        self.fc = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        # Initialize embedding weights with small random values
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        # Initialize linear layer weights with small random values
        nn.init.uniform_(self.fc.weight, -0.1, 0.1)
        nn.init.zeros_(self.fc.bias)
    
    def generate_square_subsequent_mask(self, sz, device):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf')."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, x, mask=None):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of token IDs (batch_size, seq_len)
            mask: Attention mask (optional)
            
        Returns:
            output: Predicted token probabilities (batch_size, seq_len, vocab_size)
        """
        # Get sequence length and device
        seq_len = x.size(1)
        device = x.device
        
        # Create mask if not provided
        if mask is None:
            mask = self.generate_square_subsequent_mask(seq_len, device)
        else:
            mask = mask.to(device)
        
        # Convert token IDs to embeddings and add positional encoding
        embeds = self.embedding(x) * math.sqrt(self.embedding_dim)  # Scale embeddings
        embeds = self.pos_encoder(embeds)
        
        # Process through transformer encoder
        output = self.transformer_encoder(embeds, mask)
        
        # Map hidden states to token probabilities
        output = self.fc(output)
        
        return output
    
    def predict_next_token(self, x, temperature=1.0):
        """
        Predict the next token given the input sequence
        
        Args:
            x: Input tensor of token IDs (batch_size, seq_len)
            temperature: Temperature for sampling (optional)
            
        Returns:
            next_token: Predicted next token ID
        """
        # Get device from input
        device = x.device
        
        # Create attention mask
        mask = self.generate_square_subsequent_mask(x.size(1), device)
        
        # Get predictions
        with torch.no_grad():
            output = self.forward(x, mask)
        
        # Get probabilities for the last token in the sequence
        logits = output[:, -1, :] / temperature
        
        # Get the token with the highest probability
        next_token = torch.argmax(logits, dim=-1)
        
        return next_token
    
    def prompt(self, tokenizer, text, max_seq_length=100, temperature=1.0):
        """
        Generate text given a prompt
        
        Args:
            tokenizer: Tokenizer object for encoding/decoding text
            text: The prompt text
            max_seq_length: Maximum sequence length to generate
            temperature: Temperature for sampling (optional)
            
        Returns:
            generated_text: The generated text
        """
        self.eval()  # Set the model to evaluation mode
        device = next(self.parameters()).device # Get device from model
        
        # Tokenize the prompt
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens = tokens[:-1]  # Remove EOS token to keep generating
        
        generated_tokens = tokens.copy()
        
        # Generate text until EOS token or max length
        with torch.no_grad(): # Disable gradient calculation for generation
            for _ in range(max_seq_length):
                # Ensure the sequence doesn't exceed the maximum length the model can handle
                if len(generated_tokens) >= self.max_seq_len:
                    # Use only the last max_seq_len tokens
                    context = generated_tokens[-(self.max_seq_len-1):] # Leave space for next token
                else:
                    context = generated_tokens
                
                # Convert to tensor
                input_tensor = torch.tensor([context], dtype=torch.long).to(device)
                
                # Get prediction
                # Create attention mask for current input length
                mask = self.generate_square_subsequent_mask(input_tensor.size(1), device)
                output = self.forward(input_tensor, mask)
                
                # Sample next token
                logits = output[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                # For highest probability (undergrad version)
                # next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                
                next_token_item = next_token.item()
                
                # Add the predicted token to the generated tokens
                generated_tokens.append(next_token_item)
                
                # If EOS token is generated, stop
                if next_token_item == tokenizer.eos_id:
                    break
        
        # Decode the generated tokens
        generated_text = tokenizer.decode(generated_tokens)
        
        return generated_text 