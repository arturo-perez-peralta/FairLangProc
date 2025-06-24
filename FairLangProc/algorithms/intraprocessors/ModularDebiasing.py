# Standard imports
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, Tuple, Any
import logging
import warnings

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Transformers
from transformers import AutoModel, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiffPrunedDebiasing(nn.Module, ABC):
    """
    Implements differential pruning for bias mitigation in pretrained models.
    
    This class uses concrete relaxation for differentiable pruning and applies
    sparse updates to mitigate bias while maintaining model performance.
    
    Args:
        base_model (nn.Module): Pretrained model (e.g., BERT, GPT-2)
        input_ids_A (Dict[str, torch.Tensor]): Batch dict for demographic group A
        input_ids_B (Dict[str, torch.Tensor]): Batch dict for demographic group B
        lambda_sparse (float): Weight for sparsity loss
        lambda_bias (float): Weight for bias mitigation loss
        bias_kernel (Optional[nn.Module]): Kernel for bias loss computation
        zeta (float): Temperature parameter for concrete relaxation
        gamma (float): Lower bound for concrete relaxation
        beta (float): Temperature scaling for concrete relaxation
        device (str): Device to run computations on
        eps (float): Small epsilon for numerical stability
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        input_ids_A: Dict[str, torch.Tensor],
        input_ids_B: Dict[str, torch.Tensor],
        lambda_sparse: float = 1.0,
        lambda_bias: float = 1.0,
        bias_kernel: Optional[nn.Module] = None,
        zeta: float = 1.1,
        gamma: float = -0.1,
        beta: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        eps: float = 1e-8
    ):
        super().__init__()
        
        # Validation
        self._validate_inputs(input_ids_A, input_ids_B, lambda_sparse, lambda_bias)
        
        # Core components
        self.base_model = base_model
        self.lambda_sparse = lambda_sparse
        self.lambda_bias = lambda_bias
        self.zeta = zeta
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.device = device
        
        # Bias kernel (default to identity if not provided)
        self.bias_kernel = bias_kernel or nn.Identity()
        
        # Input data for bias computation
        self.inputs_A = self._move_to_device(input_ids_A)
        self.inputs_B = self._move_to_device(input_ids_B)
        
        # Freeze base model parameters
        self._freeze_base_model()
        
        # Initialize sparse parameters
        self._init_sparse_parameters()
        
        # Get encoder reference
        self._get_encoder()
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"Initialized DiffPrunedDebiasing with {self._count_sparse_params()} sparse parameters")

    def _validate_inputs(self, inputs_A: Dict, inputs_B: Dict, lambda_sparse: float, lambda_bias: float):
        """Validate input parameters"""
        if not isinstance(inputs_A, dict) or not isinstance(inputs_B, dict):
            raise ValueError("input_ids_A and input_ids_B must be dictionaries")
        
        if lambda_sparse < 0 or lambda_bias < 0:
            raise ValueError("Lambda values must be non-negative")
        
        # Check if both groups have the same keys
        if set(inputs_A.keys()) != set(inputs_B.keys()):
            raise ValueError("Input dictionaries must have the same keys")

    def _move_to_device(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move input tensors to device"""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in inputs.items()}

    def _freeze_base_model(self):
        """Freeze all base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        logger.info("Frozen base model parameters")

    def _init_sparse_parameters(self):
        """Initialize mask (m) and magnitude (w) parameters for each layer"""
        self.sparse_params = nn.ParameterDict()
        self.name_mapping = {}
        
        param_count = 0
        for name, param in self.base_model.named_parameters():
            clean_name = name.replace('.', '_').replace('-', '_')
            self.name_mapping[clean_name] = name
            
            # Skip bias parameters and very small parameters
            if 'bias' not in name and param.numel() > 1:
                # Initialize log-alpha for concrete relaxation
                self.sparse_params[f'{clean_name}_log_alpha'] = nn.Parameter(
                    torch.randn(param.shape, device=self.device) * 0.01,
                    requires_grad=True
                )
                
                # Initialize magnitude parameters (small random initialization)
                self.sparse_params[f'{clean_name}_w'] = nn.Parameter(
                    torch.randn(param.shape, device=self.device) * 0.001,
                    requires_grad=True
                )
                param_count += param.numel()
        
        logger.info(f"Initialized sparse parameters for {param_count} base parameters")

    def _count_sparse_params(self) -> int:
        """Count total sparse parameters"""
        return sum(p.numel() for p in self.sparse_params.parameters())

    def get_concrete_mask(self, log_alpha: torch.Tensor) -> torch.Tensor:
        """
        Concrete relaxation of binary mask for differentiable pruning
        
        Args:
            log_alpha: Log-alpha parameters for concrete distribution
            
        Returns:
            Differentiable binary mask approximation
        """
        # Sample from uniform distribution
        u = torch.rand_like(log_alpha, device=log_alpha.device)
        u = torch.clamp(u, self.eps, 1 - self.eps)  # Avoid log(0)
        
        # Concrete relaxation
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / self.beta)
        
        # Scale to [gamma, zeta] and clamp to [0, 1]
        mask = s * (self.zeta - self.gamma) + self.gamma
        return torch.clamp(mask, 0, 1)

    def apply_sparse_updates(self) -> Dict[str, torch.Tensor]:
        """
        Apply current sparse updates (m*w) to base parameters
        
        Returns:
            Dictionary of updated parameters
        """
        updated_params = {}
        
        # Clone all base parameters
        for name, param in self.base_model.named_parameters():
            updated_params[name] = param.data.clone()
        
        # Apply sparse updates
        for clean_name in self.name_mapping:
            if f'{clean_name}_log_alpha' in self.sparse_params:
                original_name = self.name_mapping[clean_name]
                log_alpha = self.sparse_params[f'{clean_name}_log_alpha']
                w = self.sparse_params[f'{clean_name}_w']
                
                # Get differentiable mask
                m = self.get_concrete_mask(log_alpha)
                
                # Apply sparse update: θ' = θ + m ⊙ w
                updated_params[original_name] = updated_params[original_name] + (m * w)
        
        return updated_params

    def forward_with_updated_params(self, inputs: Dict[str, torch.Tensor], 
                                  encoder_only: bool = False) -> torch.Tensor:
        """
        Forward pass using updated parameters (base + sparse updates)
        
        Args:
            inputs: Input dictionary (input_ids, attention_mask, etc.)
            encoder_only: If True, return only encoder outputs
            
        Returns:
            Model outputs or embeddings
        """
        updated_params = self.apply_sparse_updates()
        
        # Context manager for temporary parameter updates
        original_params = {}
        try:
            # Save and update parameters
            for name, param in self.base_model.named_parameters():
                original_params[name] = param.data.clone()
                param.data.copy_(updated_params[name])
            
            # Forward pass
            if encoder_only:
                outputs = self.encoder(**inputs)
                return self._get_embedding(outputs.last_hidden_state)
            else:
                return self.base_model(**inputs)
                
        finally:
            # Restore original parameters
            for name, param in self.base_model.named_parameters():
                param.data.copy_(original_params[name])

    def compute_sparse_loss(self) -> torch.Tensor:
        """
        Compute L0 regularization loss using concrete relaxation
        
        Returns:
            Sparse regularization loss
        """
        total_sparse_loss = 0.0
        
        for name in self.sparse_params:
            if name.endswith('_log_alpha'):
                log_alpha = self.sparse_params[name]
                
                # Compute probability of being non-zero
                # P(z > 0) = sigmoid(log_alpha - β * log(-γ/ζ))
                threshold = self.beta * torch.log(torch.tensor(-self.gamma / self.zeta, device=log_alpha.device))
                prob_nonzero = torch.sigmoid(log_alpha - threshold)
                
                # L0 regularization encourages sparsity
                total_sparse_loss += prob_nonzero.sum()
        
        return total_sparse_loss / len([n for n in self.sparse_params if n.endswith('_log_alpha')])

    def compute_bias_loss(self) -> torch.Tensor:
        """
        Compute bias mitigation loss as distance between group representations
        
        Returns:
            Bias mitigation loss
        """
        try:
            # Get group representations
            repr_a = self.forward_with_updated_params(self.inputs_A, encoder_only=True)
            repr_b = self.forward_with_updated_params(self.inputs_B, encoder_only=True)
            
            # Apply bias kernel if provided
            repr_a = self.bias_kernel(repr_a)
            repr_b = self.bias_kernel(repr_b)
            
            # Compute mean representations
            mean_a = repr_a.mean(dim=0)
            mean_b = repr_b.mean(dim=0)
            
            # Bias loss: minimize distance between group means
            bias_loss = F.mse_loss(mean_a, mean_b)
            
            return bias_loss
            
        except Exception as e:
            logger.warning(f"Error computing bias loss: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def forward(self, input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None, 
                labels: torch.Tensor = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass with combined losses
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels (if available)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing losses and outputs
        """
        # Prepare inputs
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, **kwargs}
        if labels is not None:
            inputs['labels'] = labels
        
        # Forward pass with updated parameters
        outputs = self.forward_with_updated_params(inputs, encoder_only=False)
        
        # Get base loss
        base_loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else torch.tensor(0.0, device=self.device)
        
        # Compute regularization losses
        sparse_loss = self.compute_sparse_loss()
        bias_loss = self.compute_bias_loss()
        
        # Combined loss
        total_loss = base_loss + self.lambda_sparse * sparse_loss + self.lambda_bias * bias_loss
        
        return {
            'loss': total_loss,
            'base_loss': base_loss,
            'sparse_loss': sparse_loss,
            'bias_loss': bias_loss,
            'logits': getattr(outputs, 'logits', None)
        }

    def get_sparsity_stats(self) -> Dict[str, float]:
        """
        Compute detailed sparsity statistics
        
        Returns:
            Dictionary with sparsity metrics
        """
        total_params = 0
        pruned_params = 0
        layer_stats = {}
        
        for name in self.sparse_params:
            if name.endswith('_log_alpha'):
                log_alpha = self.sparse_params[name]
                
                # Compute pruning probability
                threshold = self.beta * torch.log(torch.tensor(-self.gamma / self.zeta, device=log_alpha.device))
                prob_pruned = torch.sigmoid(threshold - log_alpha)  # P(z = 0)
                
                # Count parameters
                layer_pruned = (prob_pruned > 0.5).sum().item()
                layer_total = prob_pruned.numel()
                
                pruned_params += layer_pruned
                total_params += layer_total
                
                # Store layer-wise stats
                layer_name = name.replace('_log_alpha', '')
                layer_stats[layer_name] = {
                    'sparsity': layer_pruned / layer_total,
                    'pruned_params': layer_pruned,
                    'total_params': layer_total
                }
        
        overall_sparsity = pruned_params / total_params if total_params > 0 else 0.0
        
        return {
            'overall_sparsity': overall_sparsity,
            'pruned_params': pruned_params,
            'total_params': total_params,
            'layer_stats': layer_stats
        }

    def apply_hard_pruning(self, threshold: float = 0.5):
        """
        Apply hard pruning by setting low-probability parameters to zero
        
        Args:
            threshold: Probability threshold for pruning
        """
        with torch.no_grad():
            for name in self.sparse_params:
                if name.endswith('_log_alpha'):
                    log_alpha = self.sparse_params[name]
                    w_name = name.replace('_log_alpha', '_w')
                    
                    # Compute pruning probability
                    threshold_val = self.beta * torch.log(torch.tensor(-self.gamma / self.zeta, device=log_alpha.device))
                    prob_pruned = torch.sigmoid(threshold_val - log_alpha)
                    
                    # Zero out low-probability parameters
                    mask = prob_pruned < threshold
                    self.sparse_params[w_name][mask] = 0.0
        
        logger.info(f"Applied hard pruning with threshold {threshold}")

    def save_sparse_params(self, path: str):
        """Save sparse parameters to file"""
        torch.save({
            'sparse_params': self.sparse_params.state_dict(),
            'name_mapping': self.name_mapping,
            'hyperparams': {
                'lambda_sparse': self.lambda_sparse,
                'lambda_bias': self.lambda_bias,
                'zeta': self.zeta,
                'gamma': self.gamma,
                'beta': self.beta
            }
        }, path)
        logger.info(f"Saved sparse parameters to {path}")

    def load_sparse_params(self, path: str):
        """Load sparse parameters from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.sparse_params.load_state_dict(checkpoint['sparse_params'])
        self.name_mapping = checkpoint['name_mapping']
        logger.info(f"Loaded sparse parameters from {path}")

    def to(self, device):
        """Override to() to handle device transfer consistently"""
        self.device = device if isinstance(device, str) else str(device)
        super().to(device)
        
        # Move sparse parameters
        for key in self.sparse_params:
            self.sparse_params[key] = self.sparse_params[key].to(device)
        
        # Move input tensors
        self.inputs_A = self._move_to_device(self.inputs_A)
        self.inputs_B = self._move_to_device(self.inputs_B)
        
        # Move bias kernel
        if self.bias_kernel is not None:
            self.bias_kernel = self.bias_kernel.to(device)
        
        return self

    @abstractmethod
    def _get_embedding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from hidden states (to be implemented by subclasses)"""
        pass

    @abstractmethod
    def _get_encoder(self):
        """Get encoder reference from base model (to be implemented by subclasses)"""
        pass


class DiffPrunedBERT(DiffPrunedDebiasing):
    """BERT-specific implementation of differential pruning"""
    
    def _get_embedding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract [CLS] token embedding for BERT"""
        return hidden_states[:, 0, :]  # [CLS] token is at position 0

    def _get_encoder(self):
        """Get BERT encoder"""
        if hasattr(self.base_model, 'bert'):
            self.encoder = self.base_model.bert
        elif hasattr(self.base_model, 'encoder'):
            self.encoder = self.base_model.encoder
        else:
            # Fallback: use the base model itself
            self.encoder = self.base_model
            logger.warning("Could not find BERT encoder, using base model")


class DiffPrunedGPT(DiffPrunedDebiasing):
    """GPT-specific implementation of differential pruning"""
    
    def _get_embedding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract last token embedding for GPT"""
        return hidden_states[:, -1, :]  # Last token

    def _get_encoder(self):
        """Get GPT transformer"""
        if hasattr(self.base_model, 'transformer'):
            self.encoder = self.base_model.transformer
        elif hasattr(self.base_model, 'gpt_neox'):
            self.encoder = self.base_model.gpt_neox
        else:
            self.encoder = self.base_model
            logger.warning("Could not find GPT transformer, using base model")


class DiffPrunedRoBERTa(DiffPrunedDebiasing):
    """RoBERTa-specific implementation of differential pruning"""
    
    def _get_embedding(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract [CLS] token embedding for RoBERTa"""
        return hidden_states[:, 0, :]  # [CLS] token is at position 0

    def _get_encoder(self):
        """Get RoBERTa encoder"""
        if hasattr(self.base_model, 'roberta'):
            self.encoder = self.base_model.roberta
        else:
            self.encoder = self.base_model
            logger.warning("Could not find RoBERTa encoder, using base model")


# Utility functions
def create_group_inputs(tokenizer, texts_a: list, texts_b: list, max_length: int = 128) -> Tuple[Dict, Dict]:
    """
    Create tokenized inputs for two demographic groups
    
    Args:
        tokenizer: Hugging Face tokenizer
        texts_a: List of texts for group A
        texts_b: List of texts for group B
        max_length: Maximum sequence length
        
    Returns:
        Tuple of input dictionaries for groups A and B
    """
    inputs_a = tokenizer(
        texts_a, 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    )
    
    inputs_b = tokenizer(
        texts_b, 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    )
    
    return inputs_a, inputs_b


def get_model_class(model_name: str) -> DiffPrunedDebiasing:
    """
    Get appropriate debiasing class based on model name
    
    Args:
        model_name: Name of the base model
        
    Returns:
        Appropriate debiasing class
    """
    model_name_lower = model_name.lower()
    
    if 'bert' in model_name_lower and 'roberta' not in model_name_lower:
        return DiffPrunedBERT
    elif 'roberta' in model_name_lower:
        return DiffPrunedRoBERTa
    elif any(name in model_name_lower for name in ['gpt', 'gpt2', 'gpt-2']):
        return DiffPrunedGPT
    else:
        logger.warning(f"Unknown model type for {model_name}, defaulting to BERT")
        return DiffPrunedBERT


# Example usage and training utilities
class DebiasTrainer:
    """Trainer class for differential pruning debiasing"""
    
    def __init__(self, model: DiffPrunedDebiasing, optimizer: torch.optim.Optimizer = None):
        self.model = model
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.history = {'loss': [], 'sparse_loss': [], 'bias_loss': []}
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(**batch)
        outputs['loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Record metrics
        metrics = {
            'loss': outputs['loss'].item(),
            'base_loss': outputs['base_loss'].item(),
            'sparse_loss': outputs['sparse_loss'].item(),
            'bias_loss': outputs['bias_loss'].item()
        }
        
        for key in ['loss', 'sparse_loss', 'bias_loss']:
            self.history[key].append(metrics[key])
        
        return metrics
    
    def train(self, dataloader: DataLoader, num_epochs: int = 5, 
              log_interval: int = 100, eval_fn: callable = None):
        """Training loop"""
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for step, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                
                if step % log_interval == 0:
                    sparsity_stats = self.model.get_sparsity_stats()
                    logger.info(f"Epoch {epoch}, Step {step}: "
                              f"Loss={metrics['loss']:.4f}, "
                              f"Sparsity={sparsity_stats['overall_sparsity']:.3f}")
            
            # Evaluation
            if eval_fn is not None:
                eval_metrics = eval_fn(self.model)
                logger.info(f"Epoch {epoch} evaluation: {eval_metrics}")
            
            logger.info(f"Epoch {epoch} completed. Average loss: {epoch_loss/len(dataloader):.4f}")