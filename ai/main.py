"""
Simple GPT-2 inference using MAX on CPU.
A minimal example to get started with MAX and transformers.
"""

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor, TensorType, defaults
from max.graph import DeviceRef
from max.nn.module_v3 import Embedding, Linear, Module, Sequential


# Step 1: Config - just a simple dataclass
class GPT2Config:
    vocab_size = 50257
    n_positions = 1024
    n_embd = 768
    n_layer = 12
    n_head = 12
    layer_norm_epsilon = 1e-5


# Step 2: Layer Normalization
class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.weight = Tensor.ones([dim])
        self.bias = Tensor.zeros([dim])

    def __call__(self, x):
        return F.layer_norm(x, gamma=self.weight, beta=self.bias, epsilon=self.eps)


# Step 3: Causal Mask - prevents looking at future tokens
@F.functional
def causal_mask(seq_len, dtype, device):
    from max.graph import Dim

    n = Dim(seq_len)
    mask = Tensor.constant(float("-inf"), dtype=dtype, device=device)
    mask = F.broadcast_to(mask, shape=(seq_len, n))
    return F.band_part(mask, num_lower=None, num_upper=0, exclude=True)


# Step 4: Multi-head Attention
class GPT2Attention(Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head

        self.c_attn = Linear(self.n_embd, 3 * self.n_embd, bias=True)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=True)

    def __call__(self, x):
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.c_attn(x)
        q, k, v = F.split(qkv, [self.n_embd, self.n_embd, self.n_embd], axis=2)

        # Reshape for multi-head attention
        q = q.reshape([B, T, self.n_head, self.head_dim]).transpose(-3, -2)
        k = k.reshape([B, T, self.n_head, self.head_dim]).transpose(-3, -2)
        v = v.reshape([B, T, self.n_head, self.head_dim]).transpose(-3, -2)

        # Attention scores
        scale = self.head_dim**-0.5
        attn = (q @ k.transpose(-1, -2)) * scale

        # Apply causal mask
        mask = causal_mask(T, 0, dtype=attn.dtype, device=attn.device)
        attn = attn + mask

        # Softmax and apply to values
        attn = F.softmax(attn)
        out = attn @ v

        # Merge heads back
        out = out.transpose(-3, -2).reshape([B, T, C])
        return self.c_proj(out)


# Step 5: MLP (Feed-Forward Network)
class GPT2MLP(Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=True)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=True)

    def __call__(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        return self.c_proj(x)


# Step 6: Transformer Block
class GPT2Block(Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# Step 7: Full GPT-2 Model
class GPT2(Module):
    def __init__(self, config):
        super().__init__()
        self.wte = Embedding(config.vocab_size, dim=config.n_embd)  # Token embeddings
        self.wpe = Embedding(config.n_positions, dim=config.n_embd)  # Position embeddings
        self.h = Sequential(*(GPT2Block(config) for _ in range(config.n_layer)))
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, input_ids):
        B, T = input_ids.shape

        # Get embeddings
        tok_emb = self.wte(input_ids)
        pos = Tensor.arange(T, dtype=input_ids.dtype, device=input_ids.device)
        pos_emb = self.wpe(pos)

        x = tok_emb + pos_emb
        x = self.h(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def generate(model, tokenizer, device, prompt, max_tokens=30, temperature=0.8):
    """Generate text token by token."""
    tokens = tokenizer.encode(prompt)
    input_ids = Tensor.constant([tokens], dtype=DType.int64, device=device)

    print(f"Prompt: {prompt}")
    print("-" * 40)

    for _ in range(max_tokens):
        logits = model(input_ids)
        next_logits = logits[0, -1, :]

        # Temperature sampling
        if temperature > 0:
            next_logits = next_logits / Tensor.constant(
                temperature, dtype=next_logits.dtype, device=device
            )
            probs = F.softmax(next_logits)
            probs_np = np.from_dlpack(probs.to(CPU()))
            next_id = np.random.choice(len(probs_np), p=probs_np)
        else:
            next_id = int(np.from_dlpack(F.argmax(next_logits).to(CPU())))

        # Append new token
        next_tensor = Tensor.constant([[next_id]], dtype=DType.int64, device=device)
        input_ids = F.concat([input_ids, next_tensor], axis=1)

        # Stop at end of text
        if next_id == tokenizer.eos_token_id:
            break

    # Decode result
    result_ids = np.from_dlpack(input_ids.to(CPU())).flatten().tolist()
    return tokenizer.decode(result_ids)


def main():
    print("Loading GPT-2 from HuggingFace...")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("Creating MAX model...")
    _, device = defaults()
    print(f"Device: {device}")

    config = GPT2Config()
    model = GPT2(config)

    print("Loading weights...")
    model.load_state_dict(hf_model.state_dict())
    model.to(device)

    # Transpose Conv1D weights to Linear format
    for name, child in model.descendents:
        if isinstance(child, Linear):
            if any(n in name for n in ["c_attn", "c_proj", "c_fc"]):
                child.weight = child.weight.T

    print("Compiling model...")
    token_type = TensorType(
        DType.int64, ("batch", "seq"), device=DeviceRef.from_device(device)
    )
    compiled = model.compile(token_type)

    print("\n" + "=" * 40)
    print("Generating text...")
    print("=" * 40 + "\n")

    result = generate(compiled, tokenizer, device, "The meaning of life is", max_tokens=30)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
