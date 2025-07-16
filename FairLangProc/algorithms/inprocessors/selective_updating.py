"""FairLangProc.algorithms.inprocessors.selective_updating.py

Submodule inside of the FairLangProc.algorithms.inprocessors module which stores all
processors related with the selective update of model parameters.

The supported method is selective unfreezing by name.
"""

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer

#=====================================================
# Helper functions to freeze/unfreeze parameters
#=====================================================

def freeze_all_parameters(model: nn.Module) -> None:
    r"""Freeze all parameters of the model.
    
    Parameters
    ----------
    model : nn.Module
        Model whose parameters will be frozen.
    """
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_by_name(model: nn.Module, parameters: list[str]) -> None:
    r"""Unfreeze parameters in the model whose names contain any of the specified substrings.
    
    Parameters
    ----------
    model : nn.Module      
        The model whose parameters will be adjusted.
    substrings : list[str] 
        List of substrings to search for in parameter names.
    """
    for name, param in model.named_parameters():
        if any(par in name for par in parameters):
            param.requires_grad = True


def selective_unfreezing(model: nn.Module, substrings: list[str]) -> None:
    """Freeze all model's parameters and selectively unfreeze those specified in parameters.
    
    Parameters
    ----------
    model : nn.Module      
        The model whose parameters will be adjusted.
    substrings : list[str] 
        List of substrings to search for in parameter names.

    Example
    -------
    >>> from FairLangProc.algorithms.inprocessors import selective_unfreezing
    >>> 
    >>> FrozenBert = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    >>> selective_unfreezing(FrozenBert, ["attention.self", "attention.output"])
    >>> 
    >>> trainer = Trainer(
            model=FrozenBert,
            args=training_args,
            train_dataset=train_CDA,
            eval_dataset=val_dataset,
            optimizers=(
                AdamW(FrozenBert.parameters(), lr=1e-5, weight_decay=0.1),
                None
                )
    >>> )
    >>> trainer.train()
    >>> results = trainer.evaluate()
    >>> print(results)
    """
    freeze_all_parameters(model)
    unfreeze_by_name(model, substrings)