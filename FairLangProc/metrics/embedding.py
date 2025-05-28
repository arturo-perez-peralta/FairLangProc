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

lm_tokenizer = TypeVar("lm_tokenizer", bound = "PreTrainedTokenizer")


class WEAT(ABC):
    """
    Class for handling WEAT metric with a PyTorch model and tokenizer.
    
    Args:
        model (nn.Module):     PyTorch model (e.g., BERT, GPT from HuggingFace)
        tokenizer (tokenizer): Corresponding tokenizer
        device (str):          Device to run computations on
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: lm_tokenizer,
        device: str='cuda'
        ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def get_embeddings(self, words: list[str]) -> torch.Tensor:
        """
        Get embeddings for a list of words using the LLM.
        
        Args:
            words: List of words to embed
            
        Returns:
            Tensor of shape (num_words, embedding_dim)
        """
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
        """
        Compute WEAT effect size.
        
        Args:
            X: Target concept 1 embeddings (n_X, dim)
            Y: Target concept 2 embeddings (n_Y, dim)
            A: Attribute 1 embeddings (n_A, dim)
            B: Attribute 2 embeddings (n_B, dim)
            
        Returns:
            Effect size (float)
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
        """
        Compute p-value using permutation test.
        
        Args:
            X, Y, A, B: Embedding tensors
            n_perm: Number of permutations
            
        Returns:
            p-value (float)
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

    def run_test(
        self,
        W1_words: list[str],
        W2_words: list[str],
        A1_words: list[str],
        A2_words: list[str],
        n_perm: int = 10000,
        pval: bool = True
        ) -> dict[str, float]:
        """
        Run complete WEAT.
        
        Args:
            W1_words: Target concept 1 words
            W2_words: Target concept 2 words
            A1_words: Attribute 1 words
            A2_words: Attribute 2 words
            n_perm: Number of permutations for p-value
            pval: Whether to compute or not the p-value
            
        Returns:
            Dictionary with test results
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
    """
    class with implementation of _get_embedding for bidirectional transformers
    """
    def _get_embedding(self, outputs):
        return outputs.last_hidden_state[:, 0, :]