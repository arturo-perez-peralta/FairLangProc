# Standard imports
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hugging Face
from transformers import Trainer, AutoModelForSequenceClassification
from transformers.utils import ModelOutput

# Custom imports
from FairLangProc.algorithms.output import CustomOutput



@dataclass
class BlindOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    BLIND_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None



class BLINDModel(nn.Module, ABC):
    """
    Abstract class for implementing BLIND debiasing. Requires implementation of  `_get_loss` and `_loss` methods

    Args:
        model (nn.Module):      Language model to be debiased
        config (str):           Configuration (optional, only used if using AutoModel)
        gamma (float):          Hyper-parameter that regulates the strenght of BLIND weights
        alpha (float):          Hyper-parameter that regulates the strenght of the loss
        temperature (float):    Hyper-parameter that regulates the softmax of the BLIND logodds
        hidden_dim (int):       Hyper-parameter, hidden dimension of the language model
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[str] = None,
        gamma: float = 2.0,
        alpha: float = 1.0,
        temperature: float = 1.0,
        hidden_dim: int = 768,
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
        self.alpha = alpha
        self.temperature = temperature
        self.hidden_dim = hidden_dim

        self.BLIND = nn.Linear(hidden_dim, 2)

        self._get_loss()


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
            logits_BLIND = self.BLIND(embedding).squeeze()  # shape: (batch, 2)
            
            
            # Compute BLIND loss
            pred = logits.argmax(dim=1)
            success = torch.where(pred == labels, 1, 0)
            BLIND_loss = self._loss(logits_BLIND, success)
            
            prob_dist = F.softmax(logits, dim=1)
            prob_dist_BLIND = F.softmax(logits_BLIND / self.temperature, dim=1)

            pt = prob_dist.gather(1, labels.unsqueeze(1))
            pt_BLIND = prob_dist_BLIND.gather(1, success.unsqueeze(1))
            coef = torch.pow(1 - pt_BLIND, self.gamma)
            total_loss = -self.alpha * coef * torch.log(pt)


        
        if total_loss is not None:
            loss = BLIND_loss.mean()
            return BlindOutput(
                loss = loss,
                BLIND_loss = BLIND_loss,
                logits = logits,
                last_hidden_state = embedding
                )
        elif loss_main is not None and total_loss is None:
            loss = loss_main.mean()
            return BlindOutput(
                loss = loss,
                logits = logits,
                last_hidden_state = embedding
                )
        else:
            loss = None
            return BlindOutput(
                logits = logits,
                last_hidden_state = embedding
                )





class BLINDTrainer(Trainer):
    def training_step(self, model, inputs, *args, **kwargs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        outputs = model(**inputs)
        loss = outputs.loss
        loss_blind = outputs.BLIND_loss

        if loss_blind is not None:
            self.accelerator.backward(loss_blind, retain_graph = True)

        if loss is not None:
            self.accelerator.backward(loss)

        return loss.detach()




class BLINDModelForClassification(BLINDModel):
    """
    Implementation for classification (the loss function is the cross entropy function)
    """

    def _load_model(self, model, config):
        return AutoModelForSequenceClassification(model, config = config)

    def _get_loss(self):
        self.loss_fct = nn.CrossEntropyLoss()

    def _loss(self, logits, labels):
        loss = self.loss_fct(logits, labels)
        return loss


