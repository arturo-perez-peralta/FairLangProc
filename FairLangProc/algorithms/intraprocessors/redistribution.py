"""Submodule inside of the FairLangProc.algorithms.intraprocessors module which stores all
processors related with the redistribution of model parameters.

The supported method is EAT.
"""

import torch
import torch.nn as nn


def add_EAT_hook(model: nn.Module, beta: float = 1.1):
    """Insert hook to modify attention scores.

    Parameters
    ----------
    model : nn.Module
        Model whose attention scores we want to modify.
    beta : float
        Temperature parameter.

    Example
    -------
    >>> from FairLangProc.algorithms.intraprocessors import add_EAT_hook

    >>> EATBert = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    >>> beta = 1.5
    >>> add_EAT_hook(model=EATBert, beta=beta)

    >>> trainer = Trainer(
    ...     model=EATBert,
    ...     args=training_args,
    ...     train_dataset=train_dataset,
    ...     eval_dataset=val_dataset,
    ...     optimizers=(
    ...         AdamW(EATBert.parameters(), lr=1e-5, weight_decay=0.1),
    ...         None
    ...         )
    ... )
    >>> results = trainer.evaluate()
    >>> print(results)
    """
    def attention_hook(module, input, output):
        # output: tuple (attention_scores, ...)
        attention_scores = output[0]
        return (attention_scores * beta,) + output[1:]  # Scale attention scores

    # Register hooks
    for layer in model.base_model.encoder.layer:
        layer.attention.self.register_forward_hook(attention_hook)