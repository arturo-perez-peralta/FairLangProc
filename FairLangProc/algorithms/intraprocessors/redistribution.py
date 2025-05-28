import torch
import torch.nn as nn


def add_EAT_hook(model: nn.Module, beta: float = 1.1):
    """
    Insert hook to modify attention scores.

    Args:
        model (nn.Module):  model whose attention scores we want to modify
        beta (float):       temperature parameter
    """
    def attention_hook(module, input, output):
        # output: tuple (attention_scores, ...)
        attention_scores = output[0]
        return (attention_scores * beta,) + output[1:]  # Scale attention scores

    # Register hooks
    for layer in model.base_model.encoder.layer:
        layer.attention.self.register_forward_hook(attention_hook)