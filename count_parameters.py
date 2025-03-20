import torch
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 6
    model_dim: int = 768

# Copy just the model architecture parts we need
class CastedLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

class Block(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config.model_dim, config.num_heads)
        self.mlp = MLP(config.model_dim)

class MLP(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.c_proj = CastedLinear(dim, dim)

class GPT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.num_layers
        self.embed = torch.nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = torch.nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.lm_head = CastedLinear(config.model_dim, config.vocab_size)

def count_parameters(model):
    # Get total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}\n")
    
    # Print details for each named parameter
    print("Parameter details:")
    print("-" * 75)
    print(f"{'Name':<30} {'Shape':<20} {'Parameters':>12} {'dtype':>10}")
    print("-" * 75)
    
    for name, param in model.named_parameters():
        shape_str = str(tuple(param.shape))
        num_params = param.numel()
        dtype_str = str(param.dtype).split('.')[-1]
        print(f"{name:<30} {shape_str:<20} {num_params:>12,} {dtype_str:>10}")

def main():
    config = GPTConfig()
    model = GPT(config)
    count_parameters(model)

if __name__ == "__main__":
    main() 