# Standard imports
import sys
import inspect
from typing import Optional
from abc import ABC, abstractmethod

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hugging Face
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# Custom imports
from FairLangProc.algorithms.output import CustomOutput




class BLINDModel(nn.Module, ABC):
    """
    Abstract class for implementing BLIND debiasing. Requires implementation of  `_get_loss` and `_loss` methods

    Args:
        model (nn.Module):      Language model to be debiased
        config (str):           Configuration (optional, only used if using AutoModel)
        gamma (float):          Hyper-parameter that regulates the strenght of BLIND weights
        temperature (float):    Hyper-parameter that regulates the softmax of the BLIND logodds
        hidden_dim (int):       Hyper-parameter, hidden dimension of the language model
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[str] = None,
        gamma: float = 2.0,
        temperature: float = 1.0,
        hidden_dim: int = 768,
        **kwargs_loss
    ):

        super().__init__()

        if isinstance(model, nn.Module):
            self.model = model
        elif isinstance(model, str):
            self.model_name = model
            self.model = self._load_model(self.model_name, config = config)
        else:
            raise TypeError

        self.has_head = hasattr(self.model, 'classifier') or hasattr(self.model, 'head')

        if not self.has_head:
            raise AttributeError("Given model has no head.")

        self.gamma = gamma
        self.temperature = temperature
        self.hidden_dim = hidden_dim

        self.BLIND = nn.Linear(hidden_dim, 2)

        self._get_loss(**kwargs_loss)


    @abstractmethod
    def _get_loss(self, **inputs):
        pass

    @abstractmethod
    def _loss(self, **inputs):
        pass

    @abstractmethod
    def _get_embedding(self, **inputs):
        pass


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels = None):
        """
        forward pass
        """

        # Extract embedding
        embedding = self._get_embedding(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        
        # Compute the head's logits

        if hasattr(self.model, "classifier"):
            logits = self.model.classifier(embedding)
        elif hasattr(self.model, "head"):
            logits = self.model.head(embedding)
            
        
        loss_main = None
        BLIND_loss = None
        if labels is not None:
            # Compute per-example cross entropy loss (without reduction).
            loss_main = self._loss(logits, labels)

            # Compute auxiliary predicted weight from the embedding.
            logits_BLIND = self.BLIND(embedding).squeeze(1)  # shape: (batch,)
            
            # Compute BLIND loss
            prob_dist = F.softmax(logits_BLIND / self.temperature, dim=1)
            pt = prob_dist.gather(1, labels.unsqueeze(1))
            BLIND_loss = torch.pow(1 - pt, self.gamma)

        
        if loss_main is not None and BLIND_loss is not None:
            # Not sure if I should put a minus sign here huh
            loss = loss_main * BLIND_loss
            loss = loss.mean()
        else:
            loss = None
        
        if loss is None:
            return CustomOutput(
                logits = logits,
                last_hidden_state = embedding
                )
        else:
            return CustomOutput(
                loss = loss,
                logits = logits,
                last_hidden_state = embedding
                )





class BLINDModelForClassification(BLINDModel):
    """
    Implementation for classification (the loss function is the cross entropy function)
    """

    def _load_model(self, model, config):
        return AutoModelForSequenceClassification(model, config = config)

    def _get_loss(self, n_labels):
        self.n_labels = n_labels
        if n_labels == 1:
            self.loss_fct = nn.MSELoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def _loss(self, logits, labels):
        loss = self.loss_fct(logits, labels)
        return loss


