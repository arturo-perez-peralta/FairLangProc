import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

class EmbeddingBasedRegularizer(nn.Module, ABC):
    """
    Regularizer that enforces similarity between embeddings of counterfactual word pairs.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        word_pairs: List of (word1, word2) tuples for counterfactual pairs
        reg_strength: Regularization strength coefficient
        pooling_strategy: How to pool token embeddings ('first', 'mean', 'max', 'cls')
        normalize_embeddings: Whether to L2 normalize embeddings before distance computation
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        word_pairs: List[Tuple[str, str]],
        reg_strength: float = 0.01,
        pooling_strategy: str = 'first',
        normalize_embeddings: bool = False
    ):
        super().__init__()
        
        if not word_pairs:
            raise ValueError("word_pairs cannot be empty")
        if reg_strength < 0:
            raise ValueError("reg_strength must be non-negative")
            
        self.model = model
        self.tokenizer = tokenizer
        self.reg_strength = reg_strength
        self.pooling_strategy = pooling_strategy
        self.normalize_embeddings = normalize_embeddings
        
        # Pre-tokenize word pairs
        self._prepare_word_embeddings(word_pairs)
        
    def _prepare_word_embeddings(self, word_pairs: List[Tuple[str, str]]):
        """Pre-tokenize and cache word pairs"""
        words_1, words_2 = zip(*word_pairs)
        
        # Tokenize with consistent padding
        self.tokens_1 = self.tokenizer(
            list(words_1), 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            add_special_tokens=True
        )
        
        self.tokens_2 = self.tokenizer(
            list(words_2), 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            add_special_tokens=True
        )
        
    def _pool_embeddings(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool token embeddings according to strategy"""
        if self.pooling_strategy == 'first' or self.pooling_strategy == 'cls':
            return embeddings[:, 0, :]
        elif self.pooling_strategy == 'mean':
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling_strategy == 'max':
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            embeddings = embeddings * mask_expanded + (1 - mask_expanded) * (-1e9)
            return torch.max(embeddings, 1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def _compute_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss from word pair embeddings"""
        # Move tokens to model device
        device = next(self.model.parameters()).device
        tokens_1 = {k: v.to(device) for k, v in self.tokens_1.items()}
        tokens_2 = {k: v.to(device) for k, v in self.tokens_2.items()}
        
        # Get embeddings
        embeddings_1 = self._get_embedding(tokens_1)
        embeddings_2 = self._get_embedding(tokens_2)
        
        # Pool embeddings
        pooled_1 = self._pool_embeddings(embeddings_1, tokens_1['attention_mask'])
        pooled_2 = self._pool_embeddings(embeddings_2, tokens_2['attention_mask'])
        
        # Normalize if requested
        if self.normalize_embeddings:
            pooled_1 = F.normalize(pooled_1, p=2, dim=-1)
            pooled_2 = F.normalize(pooled_2, p=2, dim=-1)
        
        # Compute pairwise distances
        diff = pooled_1 - pooled_2
        distances = torch.norm(diff, p=2, dim=-1)
        
        return distances.mean()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with regularization"""
        
        # Standard model forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            **kwargs
        )
        
        # Compute regularization loss
        reg_loss = self._compute_regularization_loss()
        reg_loss *= self.reg_strength
        
        # Combine losses
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            total_loss = outputs.loss + reg_loss
        else:
            total_loss = reg_loss
            
        return {
            'loss': total_loss,
            'outputs': outputs,
            'reg_loss': reg_loss
        }
    
    @abstractmethod
    def _get_embedding(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract embeddings from tokenized inputs"""
        pass


class BERTEmbeddingRegularizer(EmbeddingBasedRegularizer):
    """BERT-specific embedding regularizer"""
    
    def _get_embedding(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            return outputs.last_hidden_state


class RoBERTaEmbeddingRegularizer(EmbeddingBasedRegularizer):
    """RoBERTa-specific embedding regularizer"""
    
    def _get_embedding(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            return outputs.last_hidden_state


def compute_attention_entropy(
    attentions: Tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    aggregate_layers: bool = True,
    aggregate_heads: bool = True,
    return_per_sample: bool = False
) -> torch.Tensor:
    """
    Compute attention entropy across transformer layers.
    
    Args:
        attentions: Tuple of attention tensors (layers, batch, heads, seq, seq)
        attention_mask: Mask for padded tokens (batch, seq)
        aggregate_layers: Whether to average across layers
        aggregate_heads: Whether to average across attention heads
        return_per_sample: Whether to return per-sample entropies
    
    Returns:
        Entropy tensor
    """
    
    if not attentions:
        raise ValueError("attentions tuple is empty")
        
    # Stack attention layers: (layers, batch, heads, seq, seq)
    stacked_attn = torch.stack(attentions)
    
    if aggregate_heads:
        # Average over attention heads: (layers, batch, seq, seq)
        stacked_attn = stacked_attn.mean(dim=2)
    
    batch_size = stacked_attn.size(1)
    sample_entropies = []
    
    for b in range(batch_size):
        # Get mask for current sample
        mask = attention_mask[b].bool()
        valid_length = mask.sum().item()
        
        if valid_length == 0:
            warnings.warn(f"Sample {b} has no valid tokens")
            sample_entropies.append(torch.tensor(0.0, device=stacked_attn.device))
            continue
            
        # Extract valid attention weights
        if aggregate_heads:
            sample_attn = stacked_attn[:, b, :valid_length, :valid_length]
        else:
            sample_attn = stacked_attn[:, b, :, :valid_length, :valid_length]
        
        # Compute entropy: -sum(p * log(p))
        attn_probs = F.softmax(sample_attn, dim=-1)
        log_probs = F.log_softmax(sample_attn, dim=-1)
        entropy = -(attn_probs * log_probs).sum(dim=-1)
        
        # Aggregate across positions and potentially layers
        if aggregate_layers:
            entropy = entropy.mean(dim=0)  # Average over layers
        entropy = entropy.mean(dim=-1)  # Average over sequence positions
        
        if aggregate_layers:
            sample_entropies.append(entropy.mean())
        else:
            sample_entropies.append(entropy.sum())  # Sum over layers
    
    entropies = torch.stack(sample_entropies)
    
    if return_per_sample:
        return entropies
    else:
        return entropies.mean()


class EntropyAttentionRegularizer(nn.Module):
    """
    Attention entropy regularizer for transformer models.
    
    Args:
        model: Transformer model
        reg_strength: Regularization strength
        aggregate_layers: Whether to average entropy across layers
        aggregate_heads: Whether to average entropy across attention heads
        entropy_target: Target entropy value (None for minimization)
    """
    
    def __init__(
        self,
        model: nn.Module,
        reg_strength: float = 0.01,
        aggregate_layers: bool = True,
        aggregate_heads: bool = True,
        entropy_target: Optional[float] = None
    ):
        super().__init__()
        
        if reg_strength < 0:
            raise ValueError("reg_strength must be non-negative")
            
        self.model = model
        self.reg_strength = reg_strength
        self.aggregate_layers = aggregate_layers
        self.aggregate_heads = aggregate_heads
        self.entropy_target = entropy_target
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with entropy regularization"""
        
        # Ensure attention outputs are returned
        kwargs['output_attentions'] = True
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            **kwargs
        )
        
        # Compute attention entropy
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        entropy = compute_attention_entropy(
            outputs.attentions,
            attention_mask,
            self.aggregate_layers,
            self.aggregate_heads
        )
        
        # Compute regularization loss
        if self.entropy_target is not None:
            # Target specific entropy value
            reg_loss = F.mse_loss(entropy, torch.tensor(self.entropy_target, device=entropy.device))
        else:
            # Minimize entropy (negative entropy maximization)
            reg_loss = -entropy
            
        reg_loss *= self.reg_strength
        
        # Combine losses
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            total_loss = outputs.loss + reg_loss
        else:
            total_loss = reg_loss
            
        return {
            'loss': total_loss,
            'outputs': outputs,
            'reg_loss': reg_loss,
            'attention_entropy': entropy
        }


class CombinedRegularizer(nn.Module):
    """
    Combines multiple regularizers with individual weights.
    
    Args:
        model: Base model
        regularizers: Dict mapping regularizer names to (regularizer, weight) tuples
    """
    
    def __init__(self, model: nn.Module, regularizers: Dict[str, Tuple[nn.Module, float]]):
        super().__init__()
        
        if not regularizers:
            raise ValueError("regularizers dict cannot be empty")
            
        self.model = model
        self.regularizers = nn.ModuleDict()
        self.weights = {}
        
        for name, (regularizer, weight) in regularizers.items():
            if weight < 0:
                raise ValueError(f"Weight for {name} must be non-negative")
            self.regularizers[name] = regularizer
            self.weights[name] = weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with combined regularization"""
        
        # Base model forward
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        total_reg_loss = 0.0
        reg_losses = {}
        
        # Apply each regularizer
        for name, regularizer in self.regularizers.items():
            reg_output = regularizer(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            reg_loss = reg_output.get('reg_loss', reg_output.get('loss', 0.0))
            
            weighted_loss = reg_loss * self.weights[name]
            total_reg_loss += weighted_loss
            reg_losses[f'{name}_loss'] = reg_loss
        
        # Combine with base loss
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            total_loss = outputs.loss + total_reg_loss
        else:
            total_loss = total_reg_loss
            
        result = {
            'loss': total_loss,
            'outputs': outputs,
            'total_reg_loss': total_reg_loss
        }
        result.update(reg_losses)
        
        return result


# Utility functions for common regularizer setups
def create_gender_bias_regularizer(
    model: nn.Module, 
    tokenizer: Any,
    reg_strength: float = 0.01,
    model_type: str = 'bert'
) -> EmbeddingBasedRegularizer:
    """Create regularizer for common gender bias word pairs"""
    
    gender_pairs = [
        ('he', 'she'), ('him', 'her'), ('his', 'hers'),
        ('man', 'woman'), ('male', 'female'), ('boy', 'girl'),
        ('father', 'mother'), ('son', 'daughter'), ('brother', 'sister'),
        ('husband', 'wife'), ('king', 'queen'), ('prince', 'princess')
    ]
    
    if model_type.lower() == 'bert':
        return BERTEmbeddingRegularizer(model, tokenizer, gender_pairs, reg_strength)
    elif model_type.lower() == 'roberta':
        return RoBERTaEmbeddingRegularizer(model, tokenizer, gender_pairs, reg_strength)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def create_attention_entropy_regularizer(
    model: nn.Module,
    reg_strength: float = 0.01,
    target_entropy: Optional[float] = None
) -> EntropyAttentionRegularizer:
    """Create attention entropy regularizer"""
    
    return EntropyAttentionRegularizer(
        model=model,
        reg_strength=reg_strength,
        entropy_target=target_entropy
    )