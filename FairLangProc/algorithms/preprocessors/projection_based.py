
# Standard libraries
import sys
from typing import TypeVar, Optional, Union
from abc import abstractmethod, ABC

# External dependencies
import numpy as np
from sklearn.decomposition import PCA

# Pytorch
import torch
import torch.nn as nn

# Hugging Face
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

lm_tokenizer = TypeVar("lm_tokenizer", bound = "PreTrainedTokenizer")


class SentDebiasModel(nn.Module, ABC):
    """
    Implements SentDebiasModel, requires the implementation of _get_embedding, _loss and _get_loss methods.

    Args:
        model (nn.Module):              language model used
        config (str):                   Optional, configuration to use when using AutoModel
        tokenizer (lm_tokenizer):       Tokenizer associated with the model
        word_pairs (list[tuple[str]]):  list of counterfactual tuples (might be words, sentences,...)
        n_components (int):             number of components of the bias subspace
        device (str):                   device to run the model on
    """

    def __init__(
        self,
        model: Union[nn.Module, str],
        config: Optional[str] = None,
        tokenizer: Optional[lm_tokenizer] = None,
        word_pairs: list[tuple] = None,
        n_components: int = 1,
        device: str = None
    ):

        super().__init__()
        
        if isinstance(model, nn.Module):
            self.model = model
            if tokenizer is None:
                raise AttributeError("You must pass a tokenizer when using a custom model.")
        elif isinstance(model, str):
            self.model_name = model
            self.model = self._load_model(self.model_name, config = config)
        else:
            raise TypeError

        self.tokenizer = tokenizer
        self.has_head = hasattr(self.model, 'classifier') or hasattr(self.model, 'head')

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.word_pairs = word_pairs
        self.n_components = n_components
        self.bias_subspace = self._compute_bias_subspace()

        self._get_loss()


    @abstractmethod
    def _get_embedding(self, **inputs):
        pass
    
    @abstractmethod
    def _get_loss(self):
        pass

    @abstractmethod
    def _loss(self, inputs):
        pass


    def _compute_bias_subspace(self):
        """
        Compute bias subspace with PCA
        """
    
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(model)

        male_tokens = self.tokenizer([male for male, _ in self.word_pairs], return_tensors="pt", padding = True)
        female_tokens = self.tokenizer([female for _, female in self.word_pairs], return_tensors="pt", padding = True)
        with torch.no_grad():
            outputs_male = self._get_embedding(**male_tokens)
            outputs_female = self._get_embedding(**female_tokens)

        diffs = (outputs_male - outputs_female).squeeze()  # shape: (n_pairs, embedding_dim)
        pca = PCA(n_components=self.n_components)
        pca.fit(diffs)
        bias_subspace = pca.components_.T  # shape: (embedding_dim, n_components)
        return torch.tensor(bias_subspace).float().to(self.device)


    def _neutralize(self, v: torch.Tensor):
        """
        Compute the projection on bias free subspace
        """
        proj_coeff = torch.matmul(v, self.bias_subspace)
        proj = torch.matmul(proj_coeff, self.bias_subspace.T)
        v_neutral = v - proj
        return v_neutral
    
    
    def forward(self,  input_ids, attention_mask=None, token_type_ids=None, labels = None):
        """
        forward pass
        """
        embeddings = self._get_embedding(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        debiased_embeddings = self._neutralize(embeddings)
        
        if self.has_head:
            if hasattr(self.model, "classifier"):
                logits = self.model.classifier(debiased_embeddings)
            elif hasattr(self.model, "head"):
                logits = self.model.head(debiased_embeddings)
            
            if labels is not None:
                loss = self._loss(logits, labels)
                return {'logits': logits, 'debiased_embedding': debiased_embeddings, 'loss': loss}
            else:
                return {'logits': logits, 'debiased_embedding': debiased_embeddings}
        
        return {'outputs': debiased_embedding}




class SentDebiasForSequenceClassification(SentDebiasModel):
    """
    Implementation ready for sequence classification, lacks _get_embedding method
    """

    def _get_loss(self):
        self.loss_fct = nn.CrossEntropyLoss()

    def _loss(self, logits, labels):
        loss = self.loss_fct(logits, labels)
        return loss

    def _load_model(self, model_name, config):
        return AutoModelForSequenceClassification(model_name, config = config)

