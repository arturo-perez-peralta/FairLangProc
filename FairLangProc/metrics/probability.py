"""Submodule inside of the FairLangProc.metrics module which stores all methods and metrics related
with Language Modelling.

The supported metrics are LPBS, CBS, CPS, AUL.
"""

# Standard libraries
from typing import TypeVar

# Numpy
import numpy as np

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

TokenizerType = TypeVar("TokenizerType", bound="PreTrainedTokenizer")

def MaskProbability(
    model: nn.Module,
    tokenizer: TokenizerType,
    sentences: list[str],
    target_words: list[str],
    mask_indices: list[int],
    how_many: int = 2
    ) -> torch.Tensor:
    r"""Computation of masked probability with a Language Model.
    
    Computes the probability of a list of target words in the positions of certain masks given a list
    of masked sentences (the number of masks is assumed to be constant)

    Parameters
    ----------
    model : nn.Module
        Language Model used to compute probabilities.
    tokenizer : TokenizerType
        Tokenizer associated with the model.
    sentences : list[str]
        List of sentences with masks.
    target_words : list[str]
        List of words whose probabilities we want to compute.
    mask_indices : list[int]
        List of indices which indicate to which mask of the sentence
        each word corresponds to (i.e. first, second,...)
    how_many : int
        How many masks are in each sentence

    Returns
    -------
    prob_target : torch.Tensor
        Probability of target_words in the positions indicated by mask_indices.

    Example
    -------
    >>> model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> sentences = ["The [MASK] is a [MASK]", "[MASK] is such a [MASK]"]
    >>> target_words = ["engineer", "He"]
    >>> mask_indices = [0,1]
    >>> how_many = 2

    >>> probabilities = MaskProbability(model, tokenizer, sentences, target_words, mask_indices, how_many = how_many)
    """

    if not isinstance(mask_indices, np.ndarray):
        mask_indices = np.array(mask_indices)

    nSent = len(sentences)
    sentRange = np.arange(nSent)

    assert nSent == len(target_words), "Different number of sentences and target words."
    assert nSent == len(mask_indices), "Different number of sentences and mask indices."

    input_ids = tokenizer(sentences, padding = True, return_tensors="pt")
    target_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)[0]) for word in target_words]
    mask_index = torch.where(input_ids.input_ids == tokenizer.mask_token_id)

    with torch.no_grad():
        outputs = model(**input_ids)
        logits = outputs.logits

    probs = F.softmax(logits, dim = -1)
    
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
    mask_indices: list[bool]
    ) -> list[torch.Tensor]:

    r"""Computes the quotient of the probabilities of two different words in the same spot in a sentence.

    Assumes sentences with two masks. Computes the quotient of the probability of target_words being in
    the position of mask_indices divided by the prior probability of target_words in said position but with
    fill_words masked.

    Parameters
    ----------
    model : nn.Module
        Language Model used to compute probabilities.
    tokenizer : TokenizerType
        Tokenizer associated with the model.
    sentences : list[str]
        List of sentences with masks.
    target_words : list[tuple[str]]
        List containing tuples of words whose probabilities we want to compute.
    fill_words : list[str]
        List of words which replace the secondary mask.
    mask_indices : list[int]
        List of indices which indicate to which mask of the sentence each
        target word corresponds to (i.e. first (0) or second (1)).

    Returns
    -------
    probs : list[torch.Tensor]
        Quotients of probabilities given as a list of tensors

    Example
    -------
    >>> model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> sentences = ["The [MASK] is a [MASK]", "[MASK] is such a [MASK]"]
    >>> target_words = [("man", "woman"), ("He", "She")]
    >>> fill_words = ["engineer", "drag"]
    >>> mask_indices = [1,0]

    >>> quotients = MaskProbabilityQuotient(model, tokenizer, sentences, target_words, fill_word, mask_indices)
    """
    
    n_cat = len(target_words[0])
    n_sentences = len(target_words)

    try:
        fill_indices = 1 - mask_indices
    except TypeError:
        fill_indices = 1 - np.array(mask_indices)

    filled_sentences = [
        template.replace("[MASK]", word, index)
        for word, template, index in zip(fill_words, sentences, fill_indices)
    ]

    probs = []

    for cat in range(n_cat):
        words = [word_tuple[cat] for word_tuple in target_words]
        
        prior_probs = MaskProbability(model, tokenizer, sentences, words, mask_indices, how_many = 2)
        post_probs = MaskProbability(model, tokenizer, filled_sentences, words, mask_indices, how_many = 1)
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
    r"""Computes LPBS score for a list of tuples of dimension 2 of target words.

    Parameters
    ----------
    model : nn.Module                  
        Language model used to compute probabilities.
    tokenizer : TokenizerType              
        Tokenizer associated with the model.
    sentences : list[str]              
        List of sentences with masks.
    target_words : list[tuple[str]]    
        List containing tuples of words whose probabilities we want to compute.
    fill_words : list[str]             
        List of words which replace the secondary mask.
    mask_indices : list[int]           
            List of indices which indicate to which mask of the sentence each 
            target word corresponds (i.e. first (0) or second (1)).

    Returns
    -------
    probs : torch.Tensor               
        List of LPBS scores

    Example
    -------
    >>> model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> sentences = ["[MASK] is a [MASK].", "[MASK] is a [MASK].", "The [MASK] was a [MASK]."]
    >>> target_words = [("John", "Mary"), ("He", "She"), ("man", "woman")]
    >>> fill_words = ["engineer","nurse","doctor"]
    >>> mask_indices = [0, 0, 1]

    >>> LPBSscore = LPBS(
    ...     model = model,
    ...     tokenizer = tokenizer,
    ...     sentences = sentences,
    ...     target_words = target_words,
    ...     fill_words = fill_words,
    ...     mask_indices = mask_indices
    ... )
    """

    assert len(sentences) == len(fill_words), "Different number of sentences and fill words."
    assert len(sentences) == len(target_words), "Different number of sentences and target words."
    assert len(target_words[0]) == 2, "Target words must consist of pairs of words."

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
    mask_indices: list[int]
    ) -> torch.Tensor:
    r"""Computes CBS score for a list of tuples of dimension n of target words.

    Parameters
    ----------

    model : nn.Module                  
        Language model used to compute probabilities.
    tokenizer : TokenizerType              
        Tokenizer associated with the model
    sentences : list[str]     
        List of sentences with masks
    target_words : list[tuple[str]]  
        List containing tuples of words whose probabilities we want to compute
    fill_words : list[str]          
        List of words which replace the secondary mask
    mask_indices : list[int]          
        List of indices which indicate to which mask of the sentence
        each target word corresponds (i.e. first (0) or second (1))
    
    Returns
    -------
    probs : torch.Tensor
        List of CBS scores

    Example
    -------
    >>> model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> target_words = [("John", "Mamadouk", "Liu"), ("white", "black", "asian"), ("white", "black", "asian")]
    >>> sentences = ["[MASK] is a [MASK]", "The [MASK] kid got [MASK] results", "The [MASK] kid wanted to be a [MASK]"]
    >>> fill_words = ["engineer", "outstanding", "doctor"]
    >>> mask_indices = [0, 1, 1]

    >>> CBSscore = CBS(
    ...     model = model,
    ...     tokenizer = tokenizer,
    ...     sentences = sentences,
    ...     target_words = target_words,
    ...     fill_words = fill_words,
    ...     mask_indices = mask_indices
    ... )
    """

    assert len(sentences) == len(fill_words), "Different number of sentences and fill words."
    assert len(sentences) == len(target_words), "Different number of sentences and target words."

    if mask_indices is None:
        mask_indices = [0 for i in range(len(sentences))]

    probs = MaskProbabilityQuotient(model, tokenizer, sentences, target_words, fill_words, mask_indices)
    probs = torch.stack(probs, dim = 1)
    scores = torch.var(torch.log(probs), dim = 1)
    return scores



def MaskedPseudoLogLikelihood(
    model: nn.Module,
    input_ids: list[int],
    target_id: int,
    mask_id: int,
    cls_id: int,
    pad_id: int
    ) -> float:
    """Computes the PLL score for a sentence where all words are progressively masked with the exception of a word
    given by target_id.

    Parameters
    ----------
    model : nn.Module
        Language model used to compute probabilities.
    input_ids : list[int]
        List of tokens forming the sentence.
    target_id : int        
        Id of the token which should not be masked.
    mask_id : int          
        Id of the mask token.
    cls_id : int         
        Id of the cls token.
    pad_id : int       
        Id of the pad token.

    Returns
    -------
    score : float
        PLL of the masked sentence.

    Example
    --------
    >>> model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> sentence = 'The actor did a terrible job'
    >>> input_ids = tokenizer([sentence], return_tensors = 'pt')['input_ids']
    >>> target_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('actor')[0])
    >>> mask_id = tokenizer.mask_token_id
    >>> pad_id = tokenizer.pad_token_type_id
    >>> cls_id = tokenizer.cls_token_id

    >>> score = MaskedPseudoLogLikelihood(
    ...     model = model,
    ...     input_ids = input_ids,
    ...     target_id = target_id,
    ...     mask_id = mask_id,
    ...     pad_id = pad_id,
    ...     cls_id = cls_id
    ... )
    """

    for i in range(len(input_ids)):
        if input_ids[i] != cls_id:
            start = i
            break

    for i in reversed(range(len(input_ids))):
        if input_ids[i] != pad_id:
            end = i
            break  

    masked_sentences = []
    masked_words = []
    target_id_position = None

    for i in range(start, end):
        if input_ids[i] == target_id:
            target_id_position = i
            continue
        sent_clone = input_ids.clone().detach()
        masked_words.append(input_ids[i])
        sent_clone[i] = mask_id
        masked_sentences.append(sent_clone)

    masked_sentences = torch.stack(masked_sentences, dim = 0)
    masked_words = torch.tensor(masked_words)

    with torch.no_grad():
        outputs = model(masked_sentences)
        logits = outputs.logits
        logProb = torch.log(F.softmax(logits, dim = 1))

    if not target_id_position:
        indices_dim0 = torch.arange(logProb.size(0))
        indices_dim1 = torch.arange(start, end)
        indices_dim2 = masked_words


    else:
        index = target_id_position - start

        indices_dim0_seg1 = torch.arange(index)
        indices_dim1_seg1 = torch.arange(start, target_id_position)
        indices_dim2_seg1 = masked_words[:index]

        indices_dim0_seg2 = torch.arange(index, logProb.size(0))
        indices_dim1_seg2 = torch.arange(target_id_position+1, end)
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
    r"""Computes the CPS score for list of sentences.

    Parameters
    ----------
    model : nn.Module
        Language model used to compute probabilities.
    tokenizer : TokenizerType
        Tokenizer associated with the model.
    sentences : list[str]
        List of sentences for whom we will compute the CPS score.
    target_words : list[str]
        List of target words which should not be masked.

    Returns
    -------
    score : list[float]
        List of CPS score of the sentences.

    Example
    -------
    >>> model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> sentences = ['The actor did a terrible job', 'The actress did a terrible job', 'The doctor was an exemplary man', 'The doctor was an exemplary woman']
    >>> target_words = ['actor', 'actress', 'man', 'woman']

    >>> CPSscore = CPS(
    ...     model = model,
    ...     tokenizer = tokenizer,
    ...     sentences = sentences,
    ...     target_words = target_words
    ... )
    """

    assert len(sentences) == len(target_words), "Number of sentences and target words must be the same."
    assert len(sentences) != 0, "Empty sentence list."

    input_ids = tokenizer(sentences, return_tensors="pt")
    ids = input_ids['input_ids']
    target_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)[0]) for word in target_words]
    mask_index = torch.where(input_ids.input_ids == tokenizer.mask_token_id)

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_type_id
    cls_id = tokenizer.cls_token_id

    scores = []

    for sentence in range(len(sentences)):
        
        sent = ids[sentence]
        target_id = target_ids[sentence]
        score = 0

        score = MaskedPseudoLogLikelihood(
            model = model,
            input_ids = sent,
            target_id = target_id,
            mask_id = mask_id,
            cls_id = cls_id,
            pad_id = pad_id
            )     
        scores.append(score)

    return scores



def UnMaskedPseudoLogLikelihood(
    model: nn.Module,
    input_ids: list[int],
    cls_id: int,
    pad_id: int
    ) -> float:
    r"""Computes the PLL score of an unmasked sentence.

    Parameters
    ----------
    model : nn.Module      
        Language model used to compute probabilities.
    input_ids : list[int]
        List of tokens forming the sentence.
    cls_id : int
        Id of the cls token.
    pad_id : int
        Id of the pad token.

    Returns
    -------
    score : float
        PLL of the masked sentence.

    Example
    -------
    >>> model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> sentence = 'The actor did a terrible job'
    >>> input_ids = tokenizer([sentence], return_tensors = 'pt')['input_ids']
    >>> pad_id = tokenizer.pad_token_type_id
    >>> cls_id = tokenizer.cls_token_id

    >>> score = UnMaskedPseudoLogLikelihood(
    ...     model = model,
    ...     input_ids = input_ids,
    ...     pad_id = pad_id,
    ...     cls_id = cls_id
    ... )
    """

    for i in range(len(input_ids)):
        if input_ids[i] != cls_id:
            start = i
            break

    for i in reversed(range(len(input_ids))):
        if input_ids[i] != pad_id:
            end = i
            break  

    input_ids = input_ids.unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        logProb = torch.log(F.softmax(logits, dim = 1))

    indices_dim0 = torch.arange(logProb.size(0))
    indices_dim1 = torch.arange(start, end)
    indices_dim2 = input_ids.squeeze()[start:end]

    score = torch.mean(logProb[indices_dim0, indices_dim1, indices_dim2])

    return score.item()



def AUL(
    model: nn.Module,
    tokenizer: TokenizerType,
    sentences: list[str]
    ) -> list[float]:

    r"""Computes the AUL score for list of sentences.

    Parameters
    ----------
    model : nn.Module
        Language model used to compute probabilities.
    tokenizer : TokenizerType
        Tokenizer associated with the model.
    sentences : list[str]
        List of sentences for whom we will compute the AUL score.

    Returns
    -------
    score : list[float]
        List of AUL score of the sentences.

    Example
    -------
    >>> model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> sentences = ['The actor did a terrible job', 'The actress did a terrible job', 'The doctor was an exemplary man', 'The doctor was an exemplary woman']

    >>> AULscore = AUL(
    ...     model = model,
    ...     tokenizer = tokenizer,
    ...     sentences = sentences
    ... )
    """

    assert len(sentences) != 0, "Empty sentence list."
    
    input_ids = tokenizer(sentences, return_tensors="pt")
    ids = input_ids['input_ids']

    pad_id = tokenizer.pad_token_type_id
    cls_id = tokenizer.cls_token_id

    scores = []

    for sentence in range(len(sentences)):
        
        sent = ids[sentence]
        score = 0

        score = UnMaskedPseudoLogLikelihood(
            model = model,
            input_ids = sent,
            cls_id = cls_id,
            pad_id = pad_id
            )     
        scores.append(score)

    return scores