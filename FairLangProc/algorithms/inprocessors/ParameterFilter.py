import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import math

class MovementPrunedLinear(nn.Module):
    """
    Linear layer with movement pruning using learnable binary gates.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        bias: Whether to use bias
        block_size: Size of pruning blocks (must divide both dimensions)
        tau: Gating threshold
        init_sparsity: Initial sparsity level (0-1)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True, 
                 block_size: int = 32, tau: float = 0.05, init_sparsity: float = 0.1):
        super().__init__()
        
        # Validate block size
        if in_features % block_size != 0 or out_features % block_size != 0:
            raise ValueError(f"Block size {block_size} must divide both dimensions")
            
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.tau = tau
        
        # Frozen linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        for param in self.linear.parameters():
            param.requires_grad = False
            
        # Score matrix for block-wise pruning
        self.score_shape = (out_features // block_size, in_features // block_size)
        self.scores = nn.Parameter(self._init_scores(init_sparsity))
        
        # Cache for mask computation
        self._cached_mask = None
        self._cached_scores = None
        
    def _init_scores(self, sparsity: float) -> torch.Tensor:
        """Initialize scores to achieve target sparsity"""
        scores = torch.randn(self.score_shape) * 0.1
        # Adjust initialization to target sparsity
        target_logit = math.log(sparsity / (1 - sparsity))
        return scores + target_logit
        
    def _compute_mask(self) -> torch.Tensor:
        """Compute full-size mask from block scores"""
        if (self._cached_mask is not None and 
            torch.equal(self._cached_scores, self.scores.data)):
            return self._cached_mask
            
        # Compute block mask
        block_probs = torch.sigmoid(self.scores)
        block_mask = (block_probs > self.tau).float()
        
        # Upsample to full size
        full_mask = block_mask.repeat_interleave(self.block_size, dim=0)\
                              .repeat_interleave(self.block_size, dim=1)
        
        # Cache results
        self._cached_mask = full_mask
        self._cached_scores = self.scores.data.clone()
        
        return full_mask
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self._compute_mask()
        pruned_weight = self.linear.weight * mask
        return F.linear(x, pruned_weight, self.linear.bias)
        
    def get_sparsity(self) -> float:
        """Return actual sparsity level"""
        mask = self._compute_mask()
        return 1.0 - mask.mean().item()
        
    def set_tau(self, tau: float):
        """Update threshold and clear cache"""
        self.tau = tau
        self._cached_mask = None


class PrunedMultiHeadAttention(nn.Module):
    """Multi-head attention with movement pruning on projections"""
    
    def __init__(self, hidden_size: int, num_heads: int, block_size: int = 32, 
                 tau: float = 0.05, prune_qkv: bool = True, prune_out: bool = True):
        super().__init__()
        
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
            
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projection layers with optional pruning
        LinearLayer = MovementPrunedLinear if prune_qkv else nn.Linear
        self.q_proj = LinearLayer(hidden_size, hidden_size, bias=False, 
                                 block_size=block_size, tau=tau)
        self.k_proj = LinearLayer(hidden_size, hidden_size, bias=False,
                                 block_size=block_size, tau=tau)  
        self.v_proj = LinearLayer(hidden_size, hidden_size, bias=False,
                                 block_size=block_size, tau=tau)
        
        OutLayer = MovementPrunedLinear if prune_out else nn.Linear
        self.out_proj = OutLayer(hidden_size, hidden_size, bias=True,
                                block_size=block_size, tau=tau)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.out_proj(attn_output)


def apply_movement_pruning(model: nn.Module, target_modules: list = None, 
                          block_size: int = 32, tau: float = 0.05) -> nn.Module:
    """
    Apply movement pruning to specified modules in a model.
    
    Args:
        model: Target model
        target_modules: List of module name patterns to replace
        block_size: Pruning block size
        tau: Gating threshold
    """
    
    if target_modules is None:
        target_modules = ['query', 'key', 'value', 'dense']
        
    replacements = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if module name matches target patterns
            if any(target in name.lower() for target in target_modules):
                # Create pruned replacement
                pruned_module = MovementPrunedLinear(
                    module.in_features, 
                    module.out_features,
                    bias=module.bias is not None,
                    block_size=block_size,
                    tau=tau
                )
                
                # Copy weights
                pruned_module.linear.weight.data.copy_(module.weight.data)
                if module.bias is not None:
                    pruned_module.linear.bias.data.copy_(module.bias.data)
                    
                replacements[name] = pruned_module
    
    # Apply replacements
    for name, new_module in replacements.items():
        parent_name = '.'.join(name.split('.')[:-1])
        attr_name = name.split('.')[-1]
        
        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            
        setattr(parent, attr_name, new_module)
    
    return model


# Training utilities
class SparsityScheduler:
    """Schedule tau parameter during training"""
    
    def __init__(self, initial_tau: float = 0.05, final_tau: float = 0.5, 
                 warmup_steps: int = 1000, total_steps: int = 10000):
        self.initial_tau = initial_tau
        self.final_tau = final_tau  
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
    def get_tau(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.initial_tau
        
        progress = min(1.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        return self.initial_tau + (self.final_tau - self.initial_tau) * progress


def get_pruning_loss(model: nn.Module, lambda_sparsity: float = 1e-4) -> torch.Tensor:
    """Compute regularization loss for sparsity"""
    sparsity_loss = 0.0
    
    for module in model.modules():
        if isinstance(module, MovementPrunedLinear):
            # L1 regularization on sigmoid scores
            probs = torch.sigmoid(module.scores)
            sparsity_loss += probs.sum()
            
    return lambda_sparsity * sparsity_loss