# Standard imports
from abc import ABC, abstractmethod
from typing import Callable, Optional

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hugging Face
from transformers import AutoModel, AutoTokenizer



class DiffPrunedDebiasing(nn.Module, ABC):
    """
    Implements differ pruning for bias mitigation in pretrained models.
    
    Args:
        base_model (nn.Module):     Pretrained model (e.g., BERT, GPT-2)
        input_ids_A (torch.Tensor): Tensor with ids of text with demographic information of group A
        input_ids_B (torch.Tensor): Tensor with ids of text with demographic information of group B
        lambda_sparse (float):      Weight for sparsity loss
        lambda_bias (float):        Weight for bias mitigation loss
        bias_kernel (Callable):     Kernel for the embeddings of the bias loss. If None, defaults to the identity
        zeta (float):               Temperature for concrete relaxation
        gamma (float):              Parameter for concrete relaxation
        beta (float):               Parameter for concrete relaxation
    """
    def __init__(
        self,
        base_model: nn.Module,
        input_ids_A: torch.Tensor,
        input_ids_B: torch.Tensor,
        lambda_sparse: float = 1.0,
        lambda_bias: float = 1.0,
        bias_kernel: Callable = None,
        zeta=1.1,
        gamma=-0.1,
        beta=1.0
    ):
        super().__init__()
        self.base_model = base_model
        self.lambda_sparse = lambda_sparse
        self.lambda_bias = lambda_bias
        self.zeta = zeta
        self.gamma = gamma
        self.beta = beta
        self.kernel = bias_kernel

        self.inputs_A = input_ids_A
        self.inputs_B = input_ids_B
        
        # Freeze the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Initialize sparse parameters (m*w)
        self._init_sparse_parameters()
        self._get_encoder()


    def _init_sparse_parameters(self):
        """Initialize mask (m) and magnitude (w) parameters for each layer"""
        self.sparse_params = nn.ParameterDict()
        self.name_mapping = {}
        
        for name, param in self.base_model.named_parameters():
            clean_name = name.replace('.', '_')
            self.name_mapping[clean_name] = name
            if 'bias' not in name:  # Typically we don't prune biases
                # Initialize mask parameters (logα)
                self.sparse_params[f'{clean_name}_log_alpha'] = nn.Parameter(
                    torch.randn(param.shape) * 0.01,
                    requires_grad=True
                )
                # Initialize magnitude parameters
                self.sparse_params[f'{clean_name}_w'] = nn.Parameter(
                    torch.zeros(param.shape),
                    requires_grad=True
                )


    def get_concrete_mask(self, log_alpha):
        """Differentiable mask using concrete relaxation"""
        u = torch.rand_like(log_alpha)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / self.beta)
        return s * (self.zeta - self.gamma) + self.gamma
    
    def apply_sparse_updates(self):
        """
        Applies current sparse updates (m*w) to base parameters
        Returns: Dictionary of UPDATED parameters
        """
        updated_params = {} 
        # First clone all base parameters
        for name, param in self.base_model.named_parameters():
            updated_params[name] = param.data.clone().requires_grad_(True)
        
        # Apply sparse updates
        for clean_name in self.name_mapping:
            if f'{clean_name}_log_alpha' in self.sparse_params:
                original_name = self.name_mapping[clean_name]
                log_alpha = self.sparse_params[f'{clean_name}_log_alpha']
                w = self.sparse_params[f'{clean_name}_w']
                
                m = torch.clamp(self.get_concrete_mask(log_alpha), 0, 1)
                updated_params[original_name] = updated_params[original_name] + (m * w)
                
        return updated_params
    



    def forward_with_updated_params(self, encoder, **kwargs):
        """
        Forward pass using UPDATED parameters (base + sparse updates)
        Returns: Same as base model forward
        """
        updated_params = self.apply_sparse_updates()
        
        # Save original parameters
        original_params = {n: p.data.clone() for n, p in self.base_model.named_parameters()}
        
        try:
            # Apply updates
            for name, param in self.base_model.named_parameters():
                param.data.copy_(updated_params[name])
            
            # Forward pass
            if encoder:
                self._get_encoder()
                outputs = self.encoder(**kwargs)
                return self._get_embedding(outputs.last_hidden_state)
            return self.base_model(**kwargs)
        finally:
            # Restore parameters
            for name, param in self.base_model.named_parameters():
                param.data.copy_(original_params[name])


    
    def get_base_features(self, **kwargs):
        """
        Returns base model outputs (without head) using UPDATED parameters
        Output: Last hidden states (batch_size, seq_len, hidden_dim)
        """
        updated_params = self.apply_sparse_updates()
        
        # Save original parameters
        original_params = {n: p.data.clone() for n, p in self.base_model.named_parameters()}
        
        # Temporarily apply updates
        for name, param in self.base_model.named_parameters():
            param.data.copy_(updated_params[name])
        
        # Forward through base model only
        with torch.no_grad():
            outputs = self.encoder(**kwargs)
            hidden_states = outputs.last_hidden_state
        
        # Restore original parameters
        for name, param in self.base_model.named_parameters():
            param.data.copy_(original_params[name])
            
        return hidden_states

    
    def get_concrete_mask(self, log_alpha):
        """Concrete relaxation of L0 norm (for differentiable pruning)"""
        u = torch.rand_like(log_alpha)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / self.beta)
        return s * (self.zeta - self.gamma) + self.gamma
    
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # 1. Compute original model outputs
        outputs = self.forward_with_updated_params(
            encoder = False,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        # 2. Apply sparse parameter updates
        total_sparse_loss = self.compute_sparse_loss()
        
        # 3. Compute bias mitigation loss (if group masks provided)
        bias_loss = self.compute_bias_loss()
        
        # Combine losses
        loss = outputs.loss + \
               self.lambda_sparse * total_sparse_loss + \
               self.lambda_bias * bias_loss
        
        return {
            'loss': loss,
            'original_loss': outputs.loss,
            'sparse_loss': total_sparse_loss,
            'bias_loss': bias_loss,
            'logits': outputs.logits
        }

    def compute_sparse_loss(self):
        """
        Computation of sparse loss (\mathcal{L}^0) through the relaxed concrete distribution
        """
        
        total_sparse_loss = 0.0

        for name, param in self.base_model.named_parameters():
            if f'{name}_log_alpha' in self.sparse_params:
                log_alpha = self.sparse_params[f'{name}_log_alpha']
                w = self.sparse_params[f'{name}_w']
                
                # Get concrete mask
                m = self.get_concrete_mask(log_alpha)
                clamped_m = torch.clamp(m, min=0, max=1)
                
                # Apply sparse update
                param.data = param.data + clamped_m * w
                
                # Compute L0 regularization term
                sparse_loss = torch.sigmoid(log_alpha - self.beta * torch.log(-self.gamma/self.zeta))
                total_sparse_loss += sparse_loss.mean()

        return total_sparse_loss

    
    def compute_bias_loss(self):
        """
        Compute debias loss as the difference of the kernel of the counterfactual pairs
        """
        bias_loss = 0.0

        # Get hidden states from last layer
        group_a = self.forward_with_updated_params(encoder=True, **self.inputs_A)
        group_b = self.forward_with_updated_params(encoder=True, **self.inputs_B)
        
        group_a_mean = group_a.mean(dim=0)
        group_b_mean = group_b.mean(dim=0)
        
        return F.mse_loss(
            group_a_mean.requires_grad_(True), 
            group_b_mean.requires_grad_(True)
            )

    @abstractmethod
    def _get_embedding(self):
        pass

    @abstractmethod
    def _get_encoder(self):
        pass

    
    def get_sparsity(self):
        """Compute fraction of parameters that are pruned (m ≈ 0)"""
        total_params = 0
        zero_params = 0
        
        for name in self.sparse_params:
            if name.endswith('_log_alpha'):
                log_alpha = self.sparse_params[name]
                prob = torch.sigmoid(log_alpha - self.beta * torch.log(-self.gamma/self.zeta))
                zero_params += (prob < 0.5).sum().item()
                total_params += prob.numel()
                
        return zero_params / total_params

    def to(self, device):
        """Override to() to handle device transfer consistently"""
        super().to(device)
        # Move sparse parameters
        for key in self.sparse_params:
            self.sparse_params[key] = self.sparse_params[key].to(device)
        # Move input tensors
        self.inputs_A = {k: v.to(device) for k, v in self.inputs_A.items()}
        self.inputs_B = {k: v.to(device) for k, v in self.inputs_B.items()}
        return self





class DiffPrunningBERT(DiffPrunedDebiasing):

    def _get_embedding(self, outputs):
        return outputs[:, 0, :]

    def _get_encoder(self):
        self.encoder = self.base_model.bert