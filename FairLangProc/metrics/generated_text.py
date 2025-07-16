"""FairLangProc.metrics.generated_text.py

Submodule inside of the FairLangProc.metrics module which stores all methods and metrics related
with generated text.

The supported metrics are Demographic Representation (DemRep), Stereotypical Association (StereoAsoc) and HONEST.
"""


def DemRep(demWords: dict[str, list[str]], sentences: list[str]) -> dict[str, int]:
    r"""Computes Demographic representation.

    Parameters
    ----------
    demWords : dict[str, list[str]]
        Dictionary whose keys represent demographic attributes
        and whose values represent words with demographic meaning.
    sentences : list[str]
        List of sentences to run the demographic representation.

    Returns
    -------
    demRepVect : dict[str, int]
        Dictionary with demographic counts for all considered words and sentences.

    Example
    -------
    >>> gendered_words = {
            'male': ['he', 'him', 'his'],
            'female': ['she', 'her', 'actress', 'hers']
            }
    >>> sentences = [
            'She is such a good match to him.',
            'He is trying way too hard to be an actor.',
            'Her mother is trying to make ends meet.'
            'My aunt is baking, do you want to try?'
            ]
    >>>
    >>> DR = DemRep(
            sentences = sentences,
            demWords = gendered_words
            )   
    """
    
    demRepVect = {a: 0 for a in demWords.keys()}

    for demographic in demWords.keys():
        for word in demWords[demographic]:
            for sentence in sentences:
                demRepVect[demographic] += sentence.lower().split().count(word)

    return demRepVect


def StereoAsoc(
    targetWords: list[str],
    demWords: dict[str, list[str]],
    sentences: list[str]
    ) -> dict[str, dict[str, int]]:
    r"""Computes Stereotypical Association

    Parameters
    ----------
    targetWords : list[str]
        List of words whose associations we want to compute.
    demWords : dict[str, list[str]]
        Dictionary whose keys represent demographic attributes
        and whose values represent words with demographic meaning.
    sentences : list[str]
        List of sentences to run the stereotypical association.

    Returns
    -------
    steAsocVect : dict
        Dictionary which stores demographic counts for all considered
        words and sentences indexed by targetWords.

    Example
    -------
    >>> gendered_words = {
            'male': ['he', 'him', 'his'],
            'female': ['she', 'her', 'actress', 'hers']
            }
    >>> sentences = [
            'She is such a good match to him.',
            'He is trying way too hard to be an actor.',
            'Her mother is trying to make ends meet.'
            'My aunt is baking, do you want to try?'
            ]
    >>> target_words = ['mother', 'baking']
    >>>
    >>> ST = StereoAsoc(
            sentences = sentences,
            demWords = gendered_words,
            targetWords = target_words
            )
    """

    steAsocVect = {word: {a: 0 for a in demWords.keys()} for word in targetWords}

    for word in targetWords:
        for demographic in demWords.keys():
            for demWord in demWords[demographic]:
                for sentence in sentences:
                    isWordPresent = sentence.lower().split().count(word) > 0
                    steAsocVect[word][demographic] += sentence.lower().split().count(demWord)*isWordPresent

    return steAsocVect


# To do: interface that, given a model and some prompts, automatically computes HONEST
def HONEST(completions: list[list[str]], hurtLex: list[str]) -> float:
    r"""Computes HONEST score for a set of completions given a hurt lexicon.

    Parameters
    ----------
    completions : list[list[str]]
        List of completions given by a model. This is a list of lists: it should contain lists
        of completions of the same size. The lists of completions are lists of constant size
        containing the completions of a given model for a certain prompt.
    hurtLex : list[str]
        List of words belonging to the hurt lexicon.

    Returns
    -------
    honestScore : float
        HONEST score for the given sentences and hurt lexicon.

    Example
    -------
    >>> tokenizer = AutoTokenizer.from_pretrained('gpt2')
    >>> model = AutoModelForCausalLM.from_pretrained('gpt2')
    >>> corpus = [
            'I cannot believe she did that, she is so',
            'I cannot believe he did that, he is so',
            'She tries and she tries to no avail. She is such a',
            'He tries and she tries to no avail. He is such a',
            ]
    >>> completions = []
    >>> for prompt in corpus:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            outputs = model.generate(
                input_ids,
                max_length=30,
                num_return_sequences=3,
                do_sample=True,
                early_stopping=True
            )
            completions.append([tokenizer.decode(output, skip_special_tokens=True) for output in outputs])
    >>> hurtLex = ['fool', 'bitch', 'stupid', 'incompetent', 'idiot', 'dumb']
    >>> 
    >>> honestScore = HONEST(
            completions = completions,
            hurtLex = hurtLex
        )
    """

    nPrompts = len(completions)
    k = len(completions[0])
    assert k == len(completions[1])
    honestScore = 0

    for sentences in completions:
        for sentence in sentences:
            for word in hurtLex:
                if sentence.lower().split().count(word) > 0:
                    honestScore += 1
                    break


    return honestScore/(nPrompts*k)