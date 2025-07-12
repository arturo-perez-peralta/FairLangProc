# Standard imports
from typing import Optional
from abc import ABC, abstractmethod

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hugging Face
from transformers import Trainer


class BLINDTrainer(Trainer, ABC):
#     """
#     Abstract class for implementing BLIND debiasing. Requires implementation of  `_get_loss` and `_loss` methods
# 
#     Args:
#         model (nn.Module):      Language model to be debiased
#         config (str):           Configuration (optional, only used if using AutoModel)
#         gamma (float):          Hyper-parameter that regulates the strenght of BLIND weights
#         alpha (float):          Hyper-parameter that regulates the strenght of the loss
#         temperature (float):    Hyper-parameter that regulates the softmax of the BLIND logodds
#         hidden_dim (int):       Hyper-parameter, hidden dimension of the language model
#     """
    def __init__(
            self,
            blind_optimizer,
            blind_model: nn.Module = None,
            hidden_dim: int = 768,
            temperature: float = 1.0,
            gamma: float = 2.0,
            alpha: float = 1.0,
            *args,
            **kwargs
            ):
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        if blind_model is None:
            blind_model = nn.Linear(hidden_dim, 2)
        self.blind_model = blind_model.to(self.args.device)
        self.blind_optimizer = blind_optimizer(self.blind_model.parameters())
        self.temperature = temperature
        self.gamma = gamma
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: Optional[torch.Tensor] = None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            success = (preds == labels).long()

        embedding = self._get_embedding(inputs)

        if self.model.training:
            self.blind_optimizer.zero_grad()
            logits_blind = self.blind_model(embedding.detach())
            loss_blind = F.cross_entropy(logits_blind, success).mean()
            loss_blind.backward()
            self.blind_optimizer.step()

        # BLIND inference
        with torch.no_grad():
            logits_blind = self.blind_model(embedding).detach()

        # Main loss
        loss_main = self.loss_func(logits, labels, logits_blind, success)
        self.log({"loss": loss_main.detach().cpu().item()})

        if return_outputs:
            return loss_main, outputs
        else:
            return loss_main

    def loss_func(self, logits, labels, logits_blind, labels_blind):
        """ BLIND loss"""

        prob_dist = F.softmax(logits, dim=1)
        prob_dist_BLIND = F.softmax(logits_blind / self.temperature, dim=1)

        pt = prob_dist.gather(1, labels.unsqueeze(1)).squeeze(1)
        pt_BLIND = prob_dist_BLIND.gather(1, labels_blind.unsqueeze(1)).squeeze(1)

        coef = torch.pow(1 - pt_BLIND, self.gamma)
        loss = -self.alpha * coef * torch.log(pt)

        return loss.mean()

    @abstractmethod
    def _get_embedding(self):
        pass