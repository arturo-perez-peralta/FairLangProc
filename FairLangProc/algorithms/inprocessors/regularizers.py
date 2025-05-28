# Standard libraries
from abc import ABC, abstractmethod
from typing import TypeVar

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

lm.tokenizer = TypeVar("lm.tokenizer", bound="PreTrainedTokenizer")

#===================================================================================
#              Embedding based Regularizer
#===================================================================================

class EmbeddingBasedRegularizer(nn.Module, ABC):
    """
    Class for adding a regularizer based on the embeddings of counterfactual pairs.
    Requires the implementation of the _get_embedding method

    Args:
        model (nn.Module):              A language model
        tokenizer (lm.tokenizer):       tokenizer of the model
        word_pairs (list[tuple[str]]):  List of tuples of counterfactual pairs whose embeddings should be close together
                                        (e.g. daughter and son, he and she,...)
        ear_reg_strength (float):       hyper-parameter containing the strength of the regularization term
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: lm.tokenizer,
        word_pairs: list[tuple[str]],
        ear_reg_strength: float = 0.01
        ):
        super().__init__()
        self.model = model
        self.ear_reg_strength = ear_reg_strength

        self.male_ids = tokenizer(
            [male for male, _ in self.word_pairs], return_tensors="pt", padding = True
            )
        self.female_ids = self.tokenizer(
            [female for _, female in self.word_pairs], return_tensors="pt", padding = True
            )


    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels = None
        ):
        """
        forward pass
        """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels = labels
        )

        male_embeddings = self._get_embedding(self.male_ids)
        female_embeddings = self._get_embedding(self.female_ids)

        reg_loss = torch.sum(torch.pow(torch.sum(male_embeddings - female_embeddings, dim = 1), 2), dim = 0)
        reg_loss *= self.ear_reg_strength

        loss = reg_loss + output.loss
        return {"output": output, "loss": loss}

    @abstractmethod
    def _get_embedding(self, inputs):
        pass


class BERTEmbedingReg(EmbeddingBasedRegularizer):
    def _get_embedding(self, inputs):
        return self.model(**inputs).last_hidden_state[:,0,:]




#===================================================================================
#              Entropy-based Attention Regularizer
#===================================================================================


def EntropyAttentionRegularizer(
        inputs: tuple,
        attention_mask: torch.torch,
        return_values: bool = False
        ):
    """Compute the negative entropy across layers of a network for given inputs.

    Args:
        - input: tuple. Tuple of length num_layers. Each item should be in the form: BHSS
        - attention_mask. Tensor with dim: BS


        SOURCE: https://github.com/g8a9/ear
    """
    inputs = torch.stack(inputs)  #  LayersBatchHeadsSeqlenSeqlen
    assert inputs.ndim == 5, "Here we expect 5 dimensions in the form LBHSS"

    #  average over attention heads
    pool_heads = inputs.mean(2)

    batch_size = pool_heads.shape[1]
    samples_entropy = list()
    neg_entropies = list()
    for b in range(batch_size):
        #  get inputs from non-padded tokens of the current sample
        mask = attention_mask[b]
        sample = pool_heads[:, b, mask.bool(), :]
        sample = sample[:, :, mask.bool()]

        #  get the negative entropy for each non-padded token
        neg_entropy = (sample.softmax(-1) * sample.log_softmax(-1)).sum(-1)
        if return_values:
            neg_entropies.append(neg_entropy.detach())

        #  get the "average entropy" that traverses the layer
        mean_entropy = neg_entropy.mean(-1)

        #  store the sum across all the layers
        samples_entropy.append(mean_entropy.sum(0))

    # average over the batch
    final_entropy = torch.stack(samples_entropy).mean()
    
    return final_entropy


class EARModel(torch.nn.Module):
    """
    Class for adding a regularizer based on entropy attention.

    Args:
        model (nn.Module):              A language model
        ear_reg_strength (float):       hyper-parameter containing the strength of the regularization term
    """

    def __init__(self, model, ear_reg_strength: float = 0.01):
        super().__init__()
        self.model = model
        self.ear_reg_strength = ear_reg_strength

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels = None):
        """
        forward pass
        """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels = labels,
            output_attentions=True
        )

        negative_entropy = EntropyAttentionRegularizer(
            output.attentions, attention_mask
        )
        reg_loss = self.ear_reg_strength * negative_entropy
        loss = reg_loss + output.loss

        return {"output": output, "loss": loss}

