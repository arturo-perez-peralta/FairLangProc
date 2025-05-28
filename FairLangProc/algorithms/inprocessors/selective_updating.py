import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer

#=====================================================
# Helper functions to freeze/unfreeze parameters
#=====================================================

def freeze_all_parameters(model: nn.Module):
    """
    Freeze all parameters of the model.
    
    Args:
        model (nn.Module): model whose parameters will be frozen
    """
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_by_name(model: nn.Module, parameters: list[str]):
    """
    Unfreeze parameters in the model whose names contain any of the specified substrings.
    
    Args:
        model (nn.Module):      The model whose parameters will be adjusted.
        substrings (list[str]): List of substrings to search for in parameter names.
    """
    for name, param in model.named_parameters():
        if any(par in name for par in parameters):
            param.requires_grad = True


def selective_unfreezing(model: nn.Module, substrings: list[str]):
    """
    Freeze all model's parameters and selectively unfreeze those specified in parameters
    
    Args:
        model (nn.Module):      The model whose parameters will be adjusted.
        substrings (list[str]): List of substrings to search for in parameter names.
    """
    freeze_all_parameters(model)
    unfreeze_by_name(model, substrings)