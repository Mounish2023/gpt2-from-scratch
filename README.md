---
title: Gpt2 From Scratch
emoji: 🐨
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.40.0
app_file: app.py
pinned: false
short_description: Reimplementation of gpt2 from scratch in PyTorch
---
# GPT-2 From Scratch 🐨

A complete reimplementation of GPT-2 from scratch in PyTorch, trained on the FineWeb-10B dataset using distributed training on 8×A100 GPUs.

## 🎯 Project Overview

This project implements a 124M parameter GPT-2 model trained from scratch, closely following the original GPT-2/GPT-3 architecture and hyperparameters. The model achieves comparable performance to the original GPT-2 on various benchmarks including HellaSwag evaluation. you can access the final model parameters at https://drive.google.com/file/d/1uBh7gq_VsoSoT43HYjNO74D-KfivL7jx/view?usp=drive_link

The code to do inference with the model is in app.py

### Key Features

- **Complete GPT-2 implementation** from scratch in PyTorch
- **124M parameters** - matching GPT-2 small configuration
- **Distributed training** using PyTorch DDP on 8×A100 SXM GPUs
- **Large-scale training** on 10B tokens from FineWeb dataset
- **Gradio web interface** for interactive text generation
- **HellaSwag evaluation** for model benchmarking
- **Efficient tokenization** using tiktoken (GPT-2 tokenizer)

## 🏗️ Architecture

The model follows the standard Transformer decoder architecture:

- **Layers**: 12 transformer blocks
- **Hidden size**: 768 dimensions
- **Attention heads**: 12 heads (64 dimensions each)
- **Context length**: 1024 tokens
- **Vocabulary size**: 50,304 tokens
- **Parameters**: ~124M total parameters

### Key Components

- `CausalSelfAttention`: Multi-head causal self-attention with Flash Attention
- `MLP`: Feed-forward network with GELU activation
- `Block`: Transformer block with pre-norm layer normalization
- `GPT`: Main model class with token and positional embeddings

## 📊 Training Details

### Dataset
- **FineWeb-Edu 10B**: High-quality educational web content (10 billion tokens)
- **Tokenization**: GPT-2 BPE tokenizer via tiktoken
- **Data sharding**: Distributed across multiple files for efficient loading

### Training Configuration
- **Batch size**: 524,288 tokens (2^19)
- **Sequence length**: 1024 tokens
- **Learning rate**: 6e-4 with cosine decay schedule
- **Warmup steps**: 715 steps
- **Total steps**: 19,073 steps (~10B tokens)
- **Weight decay**: 0.1
- **Optimizer**: AdamW with β₁=0.9, β₂=0.95

### Infrastructure
- **Hardware**: 8×A100 SXM 80GB GPUs (Lambda Cloud)
- **Distributed training**: PyTorch DistributedDataParallel (DDP)
- **Mixed precision**: bfloat16 for training efficiency
- **Gradient clipping**: 1.0 norm clipping

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd gpt2-from-scratch

# Install dependencies
pip install -r requirements.txt
```

### Running the Web Interface

```bash
python app.py
```

This launches a Gradio interface where you can interact with the trained model for text generation.

### Training from Scratch

1. **Prepare the dataset**:
```bash
python fineweb.py
```

2. **Single GPU training**:
```bash
python train.py
```

3. **Multi-GPU distributed training**:
```bash
torchrun --standalone --nproc_per_node=8 train.py
```

### Evaluation

Run HellaSwag evaluation:
```bash
python hellaswag.py -m gpt2 -d cuda
```

## 📁 Project Structure

```
gpt2-from-scratch/
├── app.py                 # Gradio web interface
├── model.py              # GPT model implementation
├── train.py              # Training script with DDP support
├── dataloader.py         # Efficient data loading
├── ddp.py                # Distributed training setup
├── fineweb.py            # Dataset preparation
├── hellaswag.py          # HellaSwag evaluation
├── play.ipynb            # Interactive notebook
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🎮 Usage Examples

### Text Generation

```python
import torch
from model import GPT, GPTConfig
import tiktoken

# Load trained model
checkpoint = torch.load("model_19072.pt", map_location='cpu')
model = GPT(checkpoint['config'])
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()

# Generate text
enc = tiktoken.get_encoding("gpt2")
prompt = "The future of artificial intelligence is"
# ... (generation code)
```

### Model Inference

The trained model supports various generation parameters:
- **Temperature**: Controls randomness (0.1-2.0)
- **Top-k sampling**: Limits vocabulary to top-k tokens
- **Max tokens**: Controls generation length

## 📈 Performance

### Training Metrics
- **Final validation loss**: ~3.1 (comparable to GPT-2)
- **Training time**: ~24 hours on 8×A100 GPUs
- **Tokens/second**: ~50,000 tokens/sec during training

### Evaluation Results
- **HellaSwag accuracy**: ~29% (comparable to original GPT-2 124M)
- **Perplexity**: Competitive with original GPT-2 on validation set

## 🛠️ Technical Implementation

### Key Optimizations
- **Weight sharing**: Embedding and output projection layers share weights
- **Flash Attention**: Uses PyTorch's `scaled_dot_product_attention`
- **Gradient accumulation**: Enables large effective batch sizes
- **Mixed precision**: bfloat16 training for memory efficiency
- **Fused AdamW**: Hardware-optimized optimizer when available

### Distributed Training
- **Data parallelism**: Each GPU processes different data batches
- **Gradient synchronization**: Automatic gradient averaging across GPUs
- **Load balancing**: Even data distribution across processes

## 🔧 Configuration

The model configuration is highly customizable through the `GPTConfig` dataclass:

```python
@dataclass
class GPTConfig:
    n_embd: int = 768        # Embedding dimension
    n_layer: int = 12        # Number of layers
    n_head: int = 12         # Number of attention heads
    block_size: int = 1024   # Context length
    vocab_size: int = 50304  # Vocabulary size
```

## 📚 Dependencies

- **PyTorch**: Deep learning framework
- **tiktoken**: GPT-2 tokenizer
- **transformers**: For loading pretrained weights (optional)
- **datasets**: For FineWeb dataset loading
- **gradio**: Web interface
- **tqdm**: Progress bars
- **numpy**: Numerical computations

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional evaluation benchmarks
- Model size variations (medium, large, xl)
- Different training datasets
- Advanced sampling techniques
- Model compression techniques

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **OpenAI** for the original GPT-2 architecture and research
- **HuggingFace** for the FineWeb dataset and transformers library
- **Andrej Karpathy** for educational resources on transformer training
- **PyTorch team** for the excellent deep learning framework
- **Lambda Labs** for providing GPU infrastructure

## 📞 Contact

For questions or suggestions, please open an issue or reach out via the project repository.

---

*This implementation demonstrates the complete pipeline of training a large language model from scratch, including data preparation, distributed training, and evaluation.*