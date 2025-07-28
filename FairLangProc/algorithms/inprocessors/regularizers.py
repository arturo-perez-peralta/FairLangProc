"""Submodule inside of the FairLangProc.algorithms.inprocessors module which stores all
processors related with the addition of regularizers.

The supported methods are embedding-based regularizers and EAR.
"""

# Standard libraries
from abc import ABC, abstractmethod
from typing import TypeVar

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom
from FairLangProc.algorithms.output import CustomOutput

TokenizerType = TypeVar("TokenizerType", bound="PreTrainedTokenizer")

#===================================================================================
#              Embedding based Regularizer
#===================================================================================

class EmbeddingBasedRegularizer(nn.Module, ABC):
    """
    Class for adding a regularizer based on the embeddings of counterfactual pairs.
    Requires the implementation of the _get_embedding method

    Example
    -------
    >>> from FairLangProc.algorithms.inprocessors import EmbeddingBasedRegularizer
    >>> class BERTEmbedingReg(EmbeddingBasedRegularizer):
    ...     def _get_embedding(self, inputs):
    ...         return self.model(**inputs).last_hidden_state[:,0,:]
    >>> model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> words = [('he', 'she'), ('his', 'hers'), ('monk', 'nun')]
    >>> EmbRegularizer = EARModel(
    ...      model = model,
    ...      tokenizer = tokenizer,
    ...      word_pairs = words, 
    ...      ear_reg_strength = 0.01
    ... )

    >>> trainer = Trainer(
    ...     model=EARRegularizer,
    ...     args=training_args,
    ...     train_dataset=train_dataset,
    ...     eval_dataset=val_dataset,
    ...     optimizers=(
    ...         AdamW(EARRegularizer.parameters(), lr=1e-5, weight_decay=0.1),
    ...         None
    ...         )
    ... )
    >>> trainer.train()
    >>> results = trainer.evaluate()
    >>> print(results)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: TokenizerType,
        word_pairs: list[tuple[str]],
        ear_reg_strength: float = 0.01
        ) -> None:
        r"""Constructor of the EmbeddingBasedRegularizer class.

        Parameters
        ----------
        model : nn.Module   
            A language model
        tokenizer : TokenizerType
            Tokenizer of the model
        word_pairs : list[tuple[str]]
            List of tuples of counterfactual pairs whose embeddings should be close together
            (e.g. daughter and son, he and she,...).
        ear_reg_strength : float
            Hyper-parameter containing the strength of the regularization term.
        """
        super().__init__()
        self.model = model
        self.ear_reg_strength = ear_reg_strength
        self.word_pairs = word_pairs

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
        r"""Forward pass
        """
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels = labels
        )

        if labels is not None:
            male_embeddings = self._get_embedding(self.male_ids)
            female_embeddings = self._get_embedding(self.female_ids)

            reg_loss = torch.sum(torch.pow(torch.sum(male_embeddings - female_embeddings, dim = 1), 2), dim = 0)
            reg_loss *= self.ear_reg_strength

            loss = reg_loss + output.loss

            return CustomOutput(
                loss = loss,
                logits = output.logits,
                last_hidden_state = output.last_hidden_state
            )
        
        return CustomOutput(
            logits = output.logits,
            last_hidden_state = output.last_hidden_state
        )

    @abstractmethod
    def _get_embedding(self, inputs):
        pass


class BERTEmbedingReg(EmbeddingBasedRegularizer):
    r"""Concrete implementation for the BERT model."""
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
    r"""Compute the negative entropy across layers of a network for given inputs.

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
    r"""Class for adding a regularizer based on entropy attention.

    Example
    -------
    >>> from FairLangProc.algorithms.inprocessors import EARModel

    >>> model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
    >>> EARRegularizer = EARModel(
    ...      model = model,
    ...      ear_reg_strength = 0.01
    ... )

    >>> trainer = Trainer(
    ...     model=EARRegularizer,
    ...     args=training_args,
    ...     train_dataset=train_dataset,
    ...     eval_dataset=val_dataset,
    ...     optimizers=(
    ...         AdamW(EARRegularizer.parameters(), lr=1e-5, weight_decay=0.1),
    ...         None
    ...         )
    ... )
    >>> trainer.train()
    >>> results = trainer.evaluate()
    >>> print(results)
    """

    def __init__(
            self,
            model: nn.Module,
            ear_reg_strength: float = 0.01
            ):
        r"""Constructor for the EARModel class

        Parameters
        ----------
        model  : nn.Module 
            A language model.
        ear_reg_strength : float
            Hyper-parameter containing the strength of the regularization term.
        """
        super().__init__()
        self.model = model
        self.ear_reg_strength = ear_reg_strength

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels = None):
        r"""Forward pass
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

        if labels is not None:
            reg_loss = self.ear_reg_strength * negative_entropy
            loss = reg_loss + output.loss
            return CustomOutput(
                loss = loss,
                logits = output.logits
            )

        return CustomOutput(
                logits = output.logits
            )

