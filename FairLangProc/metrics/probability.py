# Standard libraries
from typing import TypeVar
from transformers import PreTrainedTokenizer

# Numpy
import numpy as np

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizer)

def MaskProbability(
    model: nn.Module,
    tokenizer: TokenizerType,
    sentences: list[str],
    target_words: list[str],
    mask_indices: list[int],
    how_many: int = 2
    ) -> torch.Tensor:
    """
    Computes the probability of a list of target words in the positions of certain masks given a list
    of masked sentences (the number of masks is assumed to be constant)

    Args:
        model (nn.Module):          Language model used to compute probabilities
        tokenizer (TokenizerType):  Tokenizer associated with the model
        sentences (list[str]):      List of sentences with masks
        target_words (list[str]):   List of words whose probabilities we want to compute 
        mask_indices (list[int]):   List of indices which indicate to which mask of the sentence
                                    each word corresponds (i.e. first, second,...)
        how_many (int):             How many masks are in each sentence

    Returns:
        prob_target (torch.Tensor): Probability of target_words in the positions indicated by mask_indices
    """

    if not isinstance(mask_indices, np.ndarray):
        mask_indices = np.array(mask_indices)

    nSent = len(sentences)
    sentRange = np.arange(nSent)

    assert nSent == len(target_words)
    assert nSent == len(mask_indices)

    input_ids = tokenizer(sentences, padding=True, return_tensors="pt")
    target_ids = []
    for word in target_words:
        tokens = tokenizer.tokenize(word)
        if tokens:
            target_ids.append(tokenizer.convert_tokens_to_ids(tokens[0]))
        else:
            # Handle case where tokenization fails
            target_ids.append(tokenizer.unk_token_id)
    
    mask_index = torch.where(input_ids.input_ids == tokenizer.mask_token_id)

    with torch.no_grad():
        outputs = model(**input_ids)
        logits = outputs.logits

    probs = F.softmax(logits, dim=-1)
    
    if how_many == 1:
        mask_position = sentRange
    else:
        mask_position = how_many*sentRange + mask_indices
    
    prob_targets = probs[sentRange, mask_index[1][mask_position], target_ids]

    return prob_targets


def MaskProbabilityQuotient(
    model: nn.Module,
    tokenizer: TokenizerType,
    sentences: list[str],
    target_words: list[tuple[str]],
    fill_words: list[str],
    mask_indices: list[int]
    ) -> list[torch.Tensor]:

    """
    Assumes sentences with two masks. Computes the quotient of the probability of target_words being in
    the position of mask_indices divided by the prior probability of target_words in said position but with
    fill_words masked.

    Args:
        model (nn.Module):                  Language model used to compute probabilities
        tokenizer (TokenizerType):          Tokenizer associated with the model
        sentences (list[str]):              List of sentences with masks
        target_words (list[tuple[str]]):    List containing tuples of words whose probabilities we want to compute
        fill_words (list[str]):             List of words which replace the secondary mask
        mask_indices (list[int]):           List of indices which indicate to which mask of the sentence
                                            each target word corresponds (i.e. first (0) or second (1))
    Returns:
        probs (list[torch.Tensor]):         Quotients of probabilities given as a list of tensors
    """
    
    n_cat = len(target_words[0])
    n_sentences = len(target_words)
    try:
        fill_indices = 1 - np.array(mask_indices)
    except TypeError:
        fill_indices = 1 - np.array(mask_indices)

    filled_sentences = []
    for word, template, index in zip(fill_words, sentences, fill_indices):
        # Replace the nth occurrence of [MASK] (0-indexed)
        masks = template.split("[MASK]")
        if len(masks) > index + 1:
            filled_sentence = "[MASK]".join(masks[:index+1]) + word + "[MASK]".join(masks[index+1:])
        else:
            filled_sentence = template  # Fallback if index is out of range
        filled_sentences.append(filled_sentence)

    probs = []

    for cat in range(n_cat):
        words = [word_tuple[cat] for word_tuple in target_words]
        
        prior_probs = MaskProbability(model, tokenizer, sentences, words, mask_indices, how_many=2)
        post_probs = MaskProbability(model, tokenizer, filled_sentences, words, mask_indices, how_many=1)
        prob_quotient = post_probs/prior_probs
        probs.append(prob_quotient)

    return probs


def LPBS(
    model: nn.Module,
    tokenizer: TokenizerType,
    sentences: list[str],
    target_words: list[tuple[str]],
    fill_words: list[str],
    mask_indices: list[int] = None
    ) -> torch.Tensor:
    """
    Computes LPBS score for a list of tuples of dimension 2 of target words

    Args:
        model (nn.Module):                  Language model used to compute probabilities
        tokenizer (TokenizerType):          Tokenizer associated with the model
        sentences (list[str]):              List of sentences with masks
        target_words (list[tuple[str]]):    List containing tuples of words whose probabilities we want to compute
        fill_words (list[str]):             List of words which replace the secondary mask
        mask_indices (list[int]):           List of indices which indicate to which mask of the sentence
                                            each target word corresponds (i.e. first (0) or second (1))
    Returns:
        probs (torch.Tensor):               List of LPBS scores
    """

    assert len(target_words[0]) == 2

    if mask_indices is None:
        mask_indices = [0 for i in range(len(sentences))]

    probs = MaskProbabilityQuotient(model, tokenizer, sentences, target_words, fill_words, mask_indices)
    scores = torch.log(probs[0]) - torch.log(probs[1])
    return scores


def CBS(
    model: nn.Module,
    tokenizer: TokenizerType,
    sentences: list[str],
    target_words: list[tuple[str]],
    fill_words: list[str],
    mask_indices: list[int] = None
    ) -> torch.Tensor:
    """
    Computes CBS score for a list of tuples of dimension n of target words

    Args:
        model (nn.Module):                  Language model used to compute probabilities
        tokenizer (TokenizerType):          Tokenizer associated with the model
        sentences (list[str]):              List of sentences with masks
        target_words (list[tuple[str]]):    List containing tuples of words whose probabilities we want to compute
        fill_words (list[str]):             List of words which replace the secondary mask
        mask_indices (list[int]):           List of indices which indicate to which mask of the sentence
                                            each target word corresponds (i.e. first (0) or second (1))
    Returns:
        probs (torch.Tensor):               List of CBS scores
    """

    if mask_indices is None:
        mask_indices = [0 for i in range(len(sentences))]

    probs = MaskProbabilityQuotient(model, tokenizer, sentences, target_words, fill_words, mask_indices)
    probs = torch.stack(probs, dim=1)
    scores = torch.var(torch.log(probs), dim=1)
    return scores


def MaskedPseudoLogLikelihood(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_id: int,
    mask_id: int,
    cls_id: int,
    pad_id: int
    ) -> float:
    """
    Computes the PLL score for a sentence where all words are progressively masked with the exception of a word
    given by target_id

    Args:
        model (nn.Module):          Language model used to compute probabilities
        input_ids (torch.Tensor):   Tensor of tokens forming the sentence
        target_id (int):            Id of the token which should not be masked
        mask_id (int):              Id of the mask token
        cls_id (int):               Id of the cls token
        pad_id (int):               Id of the pad token

    Returns:
        score (float):              PLL of the masked sentence
    """

    start = 0
    end = len(input_ids)
    
    # Find start (first non-pad token)
    for i in range(len(input_ids)):
        if input_ids[i] != pad_id:
            start = i
            break

    # Find end (last non-pad token)
    for i in reversed(range(len(input_ids))):
        if input_ids[i] != pad_id:
            end = i + 1
            break  

    masked_sentences = []
    masked_words = []
    target_id_position = None

    for i in range(start, end):
        if input_ids[i] == cls_id or input_ids[i] == pad_id:
            continue
        if input_ids[i] == target_id:
            target_id_position = i
            continue
        sent_clone = input_ids.clone().detach()
        masked_words.append(input_ids[i].item())
        sent_clone[i] = mask_id
        masked_sentences.append(sent_clone)

    if not masked_sentences:
        return 0.0

    masked_sentences = torch.stack(masked_sentences, dim=0)
    masked_words = torch.tensor(masked_words)

    with torch.no_grad():
        outputs = model(masked_sentences)
        logits = outputs.logits
        logProb = torch.log(F.softmax(logits, dim=-1))

    if target_id_position is None:
        indices_dim0 = torch.arange(logProb.size(0))
        indices_dim1 = torch.arange(start, start + len(masked_words))
        indices_dim2 = masked_words
    else:
        index = target_id_position - start

        indices_dim0_seg1 = torch.arange(index)
        indices_dim1_seg1 = torch.arange(start, target_id_position)
        indices_dim2_seg1 = masked_words[:index]

        indices_dim0_seg2 = torch.arange(index, logProb.size(0))
        indices_dim1_seg2 = torch.arange(target_id_position+1, start + len(masked_words) + 1)
        indices_dim2_seg2 = masked_words[index:]

        indices_dim0 = torch.cat([indices_dim0_seg1, indices_dim0_seg2])
        indices_dim1 = torch.cat([indices_dim1_seg1, indices_dim1_seg2])
        indices_dim2 = torch.cat([indices_dim2_seg1, indices_dim2_seg2])

    score = torch.sum(logProb[indices_dim0, indices_dim1, indices_dim2])

    return score.item()


def CPS(
    model: nn.Module,
    tokenizer: TokenizerType,
    sentences: list[str],
    target_words: list[str]
    ) -> list[float]:
    """
    Computes the CPS score for list of sentences

    Args:
        model (nn.Module):          Language model used to compute probabilities
        tokenizer (TokenizerType):  Tokenizer associated with the model
        sentences (list[str]):      List of sentences for whom we will compute the CPS score
        target_words (list[str]):   List of target words which should not be masked

    Returns:
        score (list[float]):        List of CPS score of the sentences
    """
    
    input_ids = tokenizer(sentences, padding=True, return_tensors="pt")
    ids = input_ids['input_ids']
    target_ids = []
    for word in target_words:
        tokens = tokenizer.tokenize(word)
        if tokens:
            target_ids.append(tokenizer.convert_tokens_to_ids(tokens[0]))
        else:
            target_ids.append(tokenizer.unk_token_id)

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id

    scores = []

    for sentence in range(len(sentences)):
        sent = ids[sentence]
        target_id = target_ids[sentence]
        
        score = MaskedPseudoLogLikelihood(
            model=model,
            input_ids=sent,
            target_id=target_id,
            mask_id=mask_id,
            cls_id=cls_id,
            pad_id=pad_id
        )     
        scores.append(score)

    return scores


def UnMaskedPseudoLogLikelihood(
    model: nn.Module,
    input_ids: torch.Tensor,
    cls_id: int,
    pad_id: int
    ) -> float:
    """
    Computes the PLL score of an unmasked sentence

    Args:
        model (nn.Module):          Language model used to compute probabilities
        input_ids (torch.Tensor):   Tensor of tokens forming the sentence
        cls_id (int):               Id of the cls token
        pad_id (int):               Id of the pad token

    Returns:
        score (float):              PLL of the sentence
    """

    start = 0
    end = len(input_ids)
    
    # Find start (first non-pad token)
    for i in range(len(input_ids)):
        if input_ids[i] != pad_id:
            start = i
            break

    # Find end (last non-pad token)
    for i in reversed(range(len(input_ids))):
        if input_ids[i] != pad_id:
            end = i + 1
            break  

    input_ids_batch = input_ids.unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids_batch)
        logits = outputs.logits
        logProb = torch.log(F.softmax(logits, dim=-1))

    # Skip special tokens for likelihood calculation
    valid_positions = []
    valid_tokens = []
    
    for i in range(start, end):
        if input_ids[i] != cls_id and input_ids[i] != pad_id:
            valid_positions.append(i)
            valid_tokens.append(input_ids[i].item())
    
    if not valid_tokens:
        return 0.0

    indices_dim0 = torch.zeros(len(valid_tokens), dtype=torch.long)  # batch dimension (always 0)
    indices_dim1 = torch.tensor(valid_positions, dtype=torch.long)
    indices_dim2 = torch.tensor(valid_tokens, dtype=torch.long)

    score = torch.sum(logProb[indices_dim0, indices_dim1, indices_dim2])

    return score.item()


def AUL(
    model: nn.Module,
    tokenizer: TokenizerType,
    sentences: list[str]
    ) -> list[float]:
    """
    Computes the AUL score for list of sentences

    Args:
        model (nn.Module):          Language model used to compute probabilities
        tokenizer (TokenizerType):  Tokenizer associated with the model
        sentences (list[str]):      List of sentences for whom we will compute the AUL score

    Returns:
        score (list[float]):        List of AUL score of the sentences
    """
    
    input_ids = tokenizer(sentences, padding=True, return_tensors="pt")
    ids = input_ids['input_ids']

    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id

    scores = []

    for sentence in range(len(sentences)):
        sent = ids[sentence]
        
        score = UnMaskedPseudoLogLikelihood(
            model=model,
            input_ids=sent,
            cls_id=cls_id,
            pad_id=pad_id
        )     
        scores.append(score)

    return scores