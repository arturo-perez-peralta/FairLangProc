# Standard imports
from abc import ABC, abstractmethod
from typing import Callable
from contextlib import contextmanager

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom imports
from FairLangProc.algorithms.output import CustomOutput



class DiffPrunedDebiasing(nn.Module, ABC):
    """
    Implements differ pruning for bias mitigation in pretrained models.
    
    Args:
        model (nn.Module):     Pretrained model (e.g., BERT, GPT-2)
        input_ids_A (torch.Tensor): Tensor with ids of text with demographic information of group A
        input_ids_B (torch.Tensor): Tensor with ids of text with demographic information of group B
        lambda_sparse (float):      Weight for sparsity loss
        lambda_bias (float):        Weight for bias mitigation loss
        bias_kernel (Callable):     Kernel for the embeddings of the bias loss. If None, defaults to the identity
        upper (float):              Parameter for concrete relaxation
        lower (float):              Parameter for concrete relaxation
        temp (float):               Temperature for concrete relaxation loss
    """

    def __init__(
        self,
        model: nn.Module,
        input_ids_A: torch.Tensor,
        input_ids_B: torch.Tensor,
        lambda_sparse: float = 1.0,
        lambda_bias: float = 1.0,
        bias_kernel: Callable = None,
        upper: float = 1.1,
        lower: float = -0.1
    ):
        super().__init__()
        self.model = model
        self.lambda_sparse = lambda_sparse
        self.lambda_bias = lambda_bias
        self.upper = upper
        self.lower = lower
        self.kernel = bias_kernel

        self.inputs_A = input_ids_A
        self.inputs_B = input_ids_B
        
        # Freeze the base model
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Initialize sparse parameters (m*w)
        self._init_sparse_parameters()



    def _init_sparse_parameters(self):
        """Initialize mask (m) and magnitude (w) parameters for each layer"""
        self.sparse_params = nn.ParameterDict()
        self.name_mapping = {}
        
        for name, param in self.model.named_parameters():
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


    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass"""
        
        # Apply sparse update
        with self._sparse_parameter_scope():
            # Get the outputs
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Compute final outputs with debiased representations
            logits = outputs.logits
            
            # Compute loses
            if labels is not None:
                task_loss = F.cross_entropy(logits, labels)
                sparse_loss = self.compute_sparse_loss()
                bias_loss = self.compute_bias_loss()
                
                total_loss = (task_loss + 
                            self.lambda_sparse * sparse_loss +
                            self.lambda_bias * bias_loss)
            else:
                total_loss = None
        
        return CustomOutput(
            loss = total_loss,
            logits = logits
        )
    


    @contextmanager
    def _sparse_parameter_scope(self):
        """Context manager for applying sparse updates"""
        original_params = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        try:
            # Apply sparse updates
            for name, param in self.model.named_parameters():
                if f'{name}_log_alpha' in self.sparse_params:
                    log_alpha = self.sparse_params[f'{name}_log_alpha']
                    w = self.sparse_params[f'{name}_w']
                    mask = torch.clamp(self.get_concrete_mask(log_alpha), 0, 1)
                    param.data.add_(mask * w)
            yield
        finally:
            # Restore original parameters
            for name, param in self.model.named_parameters():
                param.data.copy_(original_params[name])


    def get_concrete_mask(self, log_alpha):
        """Differentiable mask using concrete relaxation"""
        u = torch.rand_like(log_alpha)
        s = torch.sigmoid(torch.log(u) - torch.log(1 - u) + log_alpha)
        return s * (self.upper - self.lower) + self.lower
    
    
    def expected_l0_penalty(self, log_alpha):
        """Computation of L0 penalty"""
        return torch.sigmoid(log_alpha - torch.log(-self.lower/self.upper))


    def compute_sparse_loss(self):
        """Computation of sparse loss (mathcal{L}^0) through the relaxed concrete distribution"""
        total_sparse_loss = 0.0
        # Compute L0 regularization term
        for name, param in self.model.named_parameters():
            if f'{name}_log_alpha' in self.sparse_params:
                log_alpha = self.sparse_params[f'{name}_log_alpha']
                sparse_loss = self.expected_l0_penalty(log_alpha)
                total_sparse_loss += sparse_loss.item()

        return total_sparse_loss

    
    def compute_bias_loss(self):
        """
        Compute debias loss as the difference of the kernel of the counterfactual pairs
        """

        # Get hidden states from last layer
        group_a = self._get_embedding(**self.inputs_A)
        group_b = self._get_embedding(**self.inputs_B)
        
        group_a_mean = group_a.mean(dim=0)
        group_b_mean = group_b.mean(dim=0)
        
        return F.mse_loss(
            group_a_mean.requires_grad_(True), 
            group_b_mean.requires_grad_(True)
            )

    @abstractmethod
    def _get_embedding(self):
        pass

    
    def get_sparsity(self):
        """Compute fraction of parameters that are pruned (m ≈ 0)"""
        total_params = 0
        zero_params = 0
        
        for name in self.sparse_params:
            if name.endswith('_log_alpha'):
                log_alpha = self.sparse_params[name]
                prob = torch.sigmoid(log_alpha - self.beta * torch.log(-self.lower/self.upper))
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

    def _get_embedding(self, input_ids, attention_mask = None, token_type_ids = None):
        return self.model.bert(
            input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids
            ).last_hidden_state[:,0,:]





