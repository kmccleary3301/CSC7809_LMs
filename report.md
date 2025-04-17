# CSC 7700 / 4700 — Foundational AI: Project 2 Report

**Author:** Kyle McCleary

## 1. Abstract

This project involved implementing, training, and evaluating three sequential deep-learning models: a vanilla Recurrent Neural Network (RNN), a Long Short-Term Memory (LSTM) network, and a Transformer model. The goal was to compare their performance on a small-scale language generation task using the Project Gutenberg dataset. Models were implemented in PyTorch, utilizing standard modules like `torch.nn.RNN`, `torch.nn.LSTM`, and components for the Transformer. A Byte Pair Encoding (BPE) tokenizer (SentencePiece) with a 10,000-token vocabulary was trained on the dataset. Hyperparameter optimization was performed using Weights & Biases sweeps. Evaluation metrics included Perplexity (PPL) and BLEU score on a held-out test set, alongside qualitative assessment of generated text. The LSTM model achieved the lowest perplexity (80.88) and loss (4.3930), while the Transformer achieved the highest BLEU score (0.0678), indicating a trade-off between predictive accuracy and generation quality resemblance to the test set.

## 2. Methodology

The project followed the specifications outlined in the assignment rubric, focusing on creating comparable implementations of RNN, LSTM, and Transformer architectures for text generation.

### Dataset and Tokenization

The dataset consists of short text sequences extracted from Project Gutenberg literature. Following the assignment requirements, a SentencePiece BPE tokenizer was trained on the provided `train.jsonl` data using the implementation in `tokenizer.py`.

```python
# From tokenizer.py - Training Arguments
train_args = [
    f'--input={input_file}',
    f'--model_prefix={model_prefix}',
    f'--vocab_size=10000',
    f'--model_type=bpe',
    f'--character_coverage=1.0',
    '--pad_id=0', # Explicitly define pad token ID
    '--unk_id=1',
    '--bos_id=2',
    '--eos_id=3',
    '--pad_piece=[PAD]',
    '--unk_piece=[UNK]',
    '--bos_piece=[BOS]',
    '--eos_piece=[EOS]',
]
spm.SentencePieceTrainer.train(' '.join(train_args))
```

The tokenizer uses a vocabulary size ($V$) of 10,000 and includes special tokens `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`. The `tokenize_dataset` function was modified to enforce a maximum sequence length (`max_seq_length`) of 512 by truncating longer sequences and padding shorter sequences with the `[PAD]` token (ID 0). This ensures consistent input dimensions for the models.

### Model Architectures

All models were implemented in PyTorch, inheriting from `torch.nn.Module`, and share common components:

1.  **Embedding Layer:** `torch.nn.Embedding` maps input token IDs to dense vectors of dimension `embedding_dim`.
2.  **Output Layer:** A final `torch.nn.Linear` layer maps the hidden states to logits over the vocabulary size ($V$).

The core differences lie in the hidden layers processing the sequential information:

*   **RNN (`models.RNNModel`):**
    *   Uses one or more `torch.nn.RNN` layers with `hidden_dim` units.
    *   Includes dropout between layers.
    *   *Architecture Diagram Placeholder*

*   **LSTM (`models.LSTMModel`):**
    *   Uses one or more `torch.nn.LSTM` layers with `hidden_dim` units.
    *   Includes dropout between layers.
    *   *Architecture Diagram Placeholder*

*   **Transformer (`models.TransformerModel`):**
    *   Uses a stack of Transformer encoder layers (`torch.nn.TransformerEncoderLayer`).
    *   Each layer contains multi-head self-attention (MHA) and a feed-forward network (FFN).
    *   Requires positional encoding added to the input embeddings.
    *   `num_heads` specifies the number of attention heads (must be $\ge 2$).
    *   Operates with `max_seq_length = 512`.
    *   *Architecture Diagram Placeholder*

Each model implements:

*   A `forward` method that takes token IDs as input and returns logits. For generation, it also samples the next token ID (using argmax for undergrads, temperature sampling for grads).
*   A `prompt` method that tokenizes input text, feeds it to the model, and autoregressively generates text until `[EOS]` or `max_seq_length` is reached.

### Training (`pretrainer.py`)

Model training was orchestrated by the `Pretrainer` class. Key aspects include:

*   **Loss Function:** `torch.nn.CrossEntropyLoss` was used, configured with `ignore_index=0` to exclude padding tokens from the loss calculation. The loss ($L$) for a sequence is calculated as:
    $$ L = -\sum_{t=1}^{T'} \log P(y_t | y_{<t}, x) $$
    where $T'$ is the sequence length excluding padding, $y_t$ is the target token at step $t$, and $x$ is the input sequence.
*   **Optimizer:** `torch.optim.AdamW` was used, with learning rate (`lr`) and `weight_decay` tuned via sweeps.
*   **Hyperparameter Optimization:** Weights & Biases (`wandb`) sweeps were employed to find optimal hyperparameters for each model architecture. Swept parameters included `lr`, `weight_decay`, `num_layers`, `dropout`, `batch_size`, `hidden_dim`, and `embedding_dim`. The sweep configurations are defined in `configs/*.yaml`.
*   **Data Loading:** Padded sequences were prepared using `TextGenerationDataset` and loaded using `torch.utils.data.DataLoader`.
*   **Training Loop:** Included gradient accumulation, optional Automatic Mixed Precision (`use_amp`), gradient clipping (`torch.nn.utils.clip_grad_norm_`), and logging of metrics (loss, accuracy, perplexity, timing, GPU usage) to `wandb`.
*   **Scheduling & Early Stopping:** A `ReduceLROnPlateau` learning rate scheduler adjusted the LR based on validation loss. Early stopping with configurable `patience` prevented overfitting and terminated training when validation loss stopped improving.

### Evaluation

Model performance was evaluated using:

*   **Perplexity (PPL):** Calculated on the test set as the exponentiation of the average cross-entropy loss:
$$ PPL = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | w_{<i})\right) $$
    Lower PPL indicates better predictive performance. *Final PPL scores are pending.*
*   **BLEU Score:** Measures the similarity between generated text and reference text (test set completions). *Final BLEU scores and calculation method are pending.*
*   **Generated Text:** Qualitative evaluation based on model responses to specific prompts.

## 3. Results

Hyperparameter sweeps were conducted for each model architecture to identify optimal configurations.

### Hyperparameter Sweeps

The following dashboards visualize the parameter importance and correlation analysis from the Weights & Biases sweeps:

![[rnn_sweep_dashboard.png]]
*Sweep results for the RNN model.*

![[transformer_sweep_dashboard.png]]
*Sweep results for the Transformer model.*

![[lstm_sweep_dashboard.png]]
*Sweep results for the LSTM model.*

These sweeps guided the selection of hyperparameters for the final training runs of each model. Best configurations are saved in files like `configs/rnn_best.yaml`.

I used Bayesian Hyperband via a wandb sweep agent to optimize these parameters, and I did short-spurt training runs with 1 epoch each over only a subset of the data. These aren't *necessarily* optimal at scale, but they definitely improved performance over intuitive configuration.

### RNN Model

*   **Loss Curve:**
    ![[project_results/rnn/plots/rnn_loss_curve.png]]
    *Caption: Training and validation loss curves for the best RNN model.*

*   **Final Metrics:**
    > Test Set Loss: **4.6802**
    > Test Set Accuracy: **0.0369**
    > Test Set Perplexity (PPL): **107.79**
    > Test Set BLEU Score: **0.0638**

*   **Generated Text (Placeholder):**
    > **Prompt 1:** "Which do you prefer? Dogs or cats?"
    > *Response will be inserted here.*
    >
    > **Prompt 2:** "What kind of man are you?"
    > *Response will be inserted here.*

### LSTM Model

*   **Loss Curve:**
    ![[project_results/lstm/plots/lstm_loss_curve.png]]
    *Caption: Training and validation loss curves for the best LSTM model.*

*   **Final Metrics:**
    > Test Set Loss: **4.3930**
    > Test Set Accuracy: **0.0420**
    > Test Set Perplexity (PPL): **80.88**
    > Test Set BLEU Score: **0.0641**

*   **Generated Text (Placeholder):**
    > **Prompt 1:** "Which do you prefer? Dogs or cats?"
    > *Response will be inserted here.*
    >
    > **Prompt 2:** "What kind of man are you?"
    > *Response will be inserted here.*

### Transformer Model

*   **Loss Curve:**
    ![[project_results/transformer/plots/transformer_loss_curve.png]]
    *Caption: Training and validation loss curves for the best Transformer model.*

*   **Final Metrics:**
    > Test Set Loss: **4.4108**
    > Test Set Accuracy: **0.0406**
    > Test Set Perplexity (PPL): **82.34**
    > Test Set BLEU Score: **0.0678**

*   **Generated Text (Placeholder):**
    > **Prompt 1:** "Which do you prefer? Dogs or cats?"
    > *Response will be inserted here.*
    >
    > **Prompt 2:** "What kind of man are you?"
    > *Response will be inserted here.*

### Performance Summary Table

| Model       | Test Loss | Test Accuracy | Test PPL | Test BLEU | Training Time (approx) | Notes                                     |
| :---------- | :-------- | :------------ | :------- | :-------- | :--------------------- | :---------------------------------------- |
| RNN         | 4.6802    | 0.0369        | 107.79   | 0.0638    | *TBD*                  | Tuned via sweep                           |
| LSTM        | 4.3930    | 0.0420        | 80.88    | 0.0641    | *TBD*                  | Tuned via sweep                           |
| Transformer | 4.4108    | 0.0406        | 82.34    | 0.0678    | *TBD*                  | Tuned via sweep, `max_seq_length` = 512 |

## 4. Code Repository Link

The complete source code for this project, including model implementations, training scripts, configuration files, and instructions, is available at:

> https://github.com/kmccleary3301/CSC7809_LMs

Please refer to the `README.md` file in the repository for detailed setup and execution instructions.

## 5. Discussion & Conclusion

This project provided practical experience in implementing and comparing different sequence modeling architectures for text generation. The process involved setting up data processing pipelines with tokenization and padding, defining RNN, LSTM, and Transformer models in PyTorch, and managing the training process with hyperparameter sweeps, scheduling, and early stopping. Integrating Weights & Biases proved valuable for tracking experiments and optimizing hyperparameters systematically.

Key challenges included correctly implementing padding and ensuring the loss function ignored padded tokens, configuring the Transformer architecture with positional encoding and masking, and managing the computational resources required for training, especially for the Transformer model and extensive hyperparameter sweeps.

*(Placeholder for final discussion based on results)*
> The results show the LSTM achieving the lowest perplexity, suggesting it was the best model at predicting the next token on the test set. The Transformer model, while having slightly higher perplexity than the LSTM, achieved the highest BLEU score, indicating its generated text might have more n-gram overlap with the reference texts. The vanilla RNN performed worst on both metrics. These findings align with expectations: LSTM improves upon RNNs via gating, and Transformers leverage attention for potentially better sequence understanding, though perplexity and BLEU don't always perfectly align. The hyperparameter sweeps were crucial in finding reasonably performing configurations for each architecture within the limited training budget.

This was pretty fun.