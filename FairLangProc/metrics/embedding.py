# Standard imports
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import TypeVar

# numpy
import numpy as np

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


TokenizerType = TypeVar("TokenizerType", bound = "PreTrainedTokenizer")


class WEAT(ABC):
    """Class for handling WEAT metric with a PyTorch model and tokenizer.
    
    Attributes
    ----------
    model : nn.Module     
        PyTorch model (e.g., BERT, GPT from HuggingFace).
    tokenizer : TokenizerType
        Tokenizer for the model.
    device : str
        Device to run the WEAT test on.

    Methods
    -------
    metric(W1_words, W2_words, A1_words, A2_words, n_perm, pval)
        Computation of the WEAT effect size between W1, W2 and A1, A2.
    _get_embedding(outputs)
        Abstract method whose implementation is required and which aims to compute the embedding of an output given
        by the model.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerType,
        device: str='cuda'
        ) -> None:
        r"""Constructor for the WEAT class

        Parameters
        ----------
        model : nn.Module     
            PyTorch model (e.g., BERT, GPT from HuggingFace).
        tokenizer : TokenizerType
            Tokenizer for the model.
        device : str
            Device to run the WEAT test on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def get_embeddings(self, words: list[str]) -> torch.Tensor:
        """Get embeddings for a list of words using the LLM."""
        embeddings = []
        for word in words:
            # Tokenize and get embeddings
            inputs = self.tokenizer(word, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get hidden states from specified layer
            word_embedding = self._get_embedding(outputs)
            
            embeddings.append(word_embedding)
        
        return torch.stack(embeddings)

    @abstractmethod
    def _get_embedding(self, outputs):
        r"""Abstract method that instructs the class on how to obtain the embedding of a given input."""
        pass

    def cosine_similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between two tensors."""
        return F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=-1)

    def effect_size(self,
        X: torch.Tensor,
        Y: torch.Tensor, 
        A: torch.Tensor,
        B: torch.Tensor
        ) -> float:
        r"""Compute WEAT effect size.
        
        Parameters
        ----------
        X : torch.Tensor
            Target concept 1 embeddings (n_X, dim)
        Y : torch.Tensor
            Target concept 2 embeddings (n_Y, dim)
        A : torch.Tensor
            Attribute 1 embeddings (n_A, dim)
        B : torch.Tensor
            Attribute 2 embeddings (n_B, dim)
            
        Returns
        -------
        Effect size : float

        Example
        -------
        >>> from transformers import AutoTokenizer, AutoModel
        >>> from FairLangProc.metrics import WEAT
        >>> class BertWEAT(WEAT):
            def _get_embedding(self, outputs):
                return outputs.last_hidden_state[:, 0, :]
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        >>> model = AutoModel.from_pretrained('bert-base-uncased')
        >>> math = ['math', 'algebra', 'geometry', 'calculus', 'equations']
        >>> arts = ['poetry', 'art', 'dance', 'literature', 'novel']
        >>> masc = ['male', 'man', 'boy', 'brother', 'he']
        >>> femn = ['female', 'woman', 'girl', 'sister', 'she']
        >>> 
        >>> weatClass = BertWEAT(model = model, tokenizer = tokenizer)
        >>> weatClass.metric(
                W1_words = math, W2_words = arts,
                A1_words = masc, A2_words = femn,
                pval = False
                )
        """
        # Compute similarities
        x_a = self.cosine_similarity(X, A).mean()
        x_b = self.cosine_similarity(X, B).mean()
        y_a = self.cosine_similarity(Y, A).mean()
        y_b = self.cosine_similarity(Y, B).mean()
        
        # Difference in mean similarities
        diff_x = x_a - x_b
        diff_y = y_a - y_b
        
        # Pooled standard deviation
        x_diffs = self.cosine_similarity(X, A) - self.cosine_similarity(X, B)
        y_diffs = self.cosine_similarity(Y, A) - self.cosine_similarity(Y, B)
        std_x = x_diffs.std(unbiased=False)
        std_y = y_diffs.std(unbiased=False)
        pooled_std = torch.sqrt((std_x**2 + std_y**2) / 2)
        
        return ((diff_x - diff_y) / pooled_std).item()

    def p_value(self, X: torch.Tensor, Y: torch.Tensor, 
               A: torch.Tensor, B: torch.Tensor, 
               n_perm: int = 10000) -> float:
        r"""Compute p-value using permutation test.
        
        Parameters
        ----------
        X, Y, A, B : torch.Tensor
            Embedding tensors
        n_perm : int
            Number of permutations
            
        Returns
        -------
        p-value : float
        """
        combined = torch.cat([X, Y])
        size_X = X.size(0)
        observed_effect = self.effect_size(X, Y, A, B)
        
        count = 0
        for _ in tqdm(range(n_perm), desc="Running permutations"):
            # Shuffle and split
            perm = combined[torch.randperm(combined.size(0))]
            X_perm = perm[:size_X]
            Y_perm = perm[size_X:]
            
            # Compute effect for this permutation
            effect = self.effect_size(X_perm, Y_perm, A, B)
            if effect > observed_effect:
                count += 1
                
        return (count + 1) / (n_perm + 1)  # Add 1 for smoothing

    def metric(
        self,
        W1_words: list[str],
        W2_words: list[str],
        A1_words: list[str],
        A2_words: list[str],
        n_perm: int = 10000,
        pval: bool = True
        ) -> dict[str, float]:
        r"""Run WEAT test.
        
        Parameters
        ----------
        W1_words : list[str]
            Target concept 1 words/sentences
        W2_words : list[str]
            Target concept 2 words
        A1_words : list[str]
            Attribute 1 words/sentences
        A2_words : list[str]
            Attribute 2 words/sentences
        n_perm : int
            Number of permutations for p-value
        pval : bool
            Whether to compute or not the p-value
            
        Returns
        -------
        results : dict[str, float]
            Dictionary with test results, namely mean similarity between W1, W2 and A1, A2; their sizes,
            the WEAT effect size and the p-value if needed.
        """
        # Get embeddings
        X = self.get_embeddings(W1_words)
        Y = self.get_embeddings(W2_words)
        A = self.get_embeddings(A1_words)
        B = self.get_embeddings(A2_words)

        # Compute mean similarities
        x_a = self.cosine_similarity(X, A).mean().item()
        x_b = self.cosine_similarity(X, B).mean().item()
        y_a = self.cosine_similarity(Y, A).mean().item()
        y_b = self.cosine_similarity(Y, B).mean().item()

        results = {
            'X-A_mean_sim': x_a,
            'X-B_mean_sim': x_b,
            'Y-A_mean_sim': y_a,
            'Y-B_mean_sim': y_b,
            'W1_size': len(W1_words),
            'W2_size': len(W2_words),
            'A1_size': len(A1_words),
            'A2_size': len(A2_words)
        }
        
        # Compute statistics
        effect = self.effect_size(X, Y, A, B)
        results['effect_size'] = effect
        if pval:
            p_val = self.p_value(X, Y, A, B, n_perm)
            results['p_value']= p_val
        return results


class BertWEAT(WEAT):
    """Class with implementation of _get_embedding for bidirectional transformers
    """
    def _get_embedding(self, outputs):
        return outputs.last_hidden_state[:, 0, :]