"""Submodule inside of the FairLangProc.algorithms.preprocessors module which stores all
processors related with reweighting training instances.

The supported method is BLIND debiasing.
"""

# Standard imports
from typing import Optional, Type
from abc import ABC, abstractmethod

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer, AdamW

# Hugging Face
from transformers import Trainer


class BLINDTrainer(Trainer, ABC):
    r"""Abstract class for implementing BLIND debiasing through a custom trainer. Requires implementation of  `_get_embedding` method.

    Extends the trainer class from hugging face thus inhereting all relevant methods and attributes.

    Example
    -------
    >>> from FairLangProc.algorithms.preprocessors import BLINDTrainer
    >>> 
    >>> BLINDModel = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    >>> BLINDClassifier = nn.Sequential(
              nn.Linear(HIDDEN_DIM_BERT, HIDDEN_DIM_BERT),
              nn.ReLU(),
              nn.Linear(HIDDEN_DIM_BERT, 2)
        )
    >>> class BLINDBERTTrainer(BLINDTrainer):
            def _get_embedding(self, inputs):
                return self.model.bert(
                    input_ids = inputs.get("input_ids"),
                    attention_mask = inputs.get("attention_mask"),
                    token_type_ids = inputs.get("token_type_ids")
                    ).last_hidden_state[:,0,:]
    >>>     
    >>> trainer = BLINDBERTTrainer(
            blind_model = BLINDClassifier,
            blind_optimizer = lambda x: AdamW(x, lr=1e-5, weight_decay=0.1),
            temperature = 1.0,
            gamma = 2.0,
            alpha = 1.0,
            model = BLINDModel,
            args = training_args,
            train_dataset = train_dataset,
            eval_dataset = val_dataset,
            optimizers=(
                AdamW(BLINDModel.parameters(), lr=1e-5, weight_decay=0.1),
                None
                )
        )
    >>> trainer.train()
    >>> results = trainer.evaluate()
    """

    def __init__(
            self,
            blind_optimizer: Optimizer = lambda x: AdamW(x, lr=1e-5, weight_decay=0.1),
            blind_model: nn.Module = None,
            hidden_dim: int = 768,
            temperature: float = 1.0,
            gamma: float = 2.0,
            alpha: float = 1.0,
            *args,
            **kwargs
            ) -> None:
        r"""Constructor of the BLINDTrainer class.

        Parameters
        ----------
        blind_optimizer : Optimizer
            Optimizer for the BLIND classifier.
        blind_model : nn.Module        
            Classifier used to measure the model's chance of succes for a given training instance.
        hidden_dim : int               
            Hyper-parameter, hidden dimension of the language model. I no blind model is given, the hidden dimension is used to create a simple classifier based on a linear layer.
        temperature : float            
            Hyper-parameter that regulates the softmax of the BLIND logodds.
        gamma : float   
            Hyper-parameter that regulates the strenght of BLIND weights.
        alpha : float      
            Hyper-parameter that regulates the strenght of the loss.
        *args, **kwargs                
            Usual arguments for a hugging face trainer.
        """
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
        """Compute loss step"""
        # Model outputs
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits 

        # Obtain success labels
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            success = (preds == labels).long()

        embedding = self._get_embedding(inputs)

        # Train step of BLIND model
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
        """BLIND loss"""
        prob_dist = F.softmax(logits, dim=1)
        prob_dist_BLIND = F.softmax(logits_blind / self.temperature, dim=1)

        pt = prob_dist.gather(1, labels.unsqueeze(1)).squeeze(1)
        pt_BLIND = prob_dist_BLIND.gather(1, labels_blind.unsqueeze(1)).squeeze(1)

        coef = torch.pow(1 - pt_BLIND, self.gamma)
        loss = -self.alpha * coef * torch.log(pt)

        return loss.mean()

    @abstractmethod
    def _get_embedding(self):
        """Abstract methods which computes the embedding of a given model's outputs."""
        pass