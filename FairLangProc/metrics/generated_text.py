


def DemRep(demWords: dict[str, list[str]], sentences: list[str]) -> dict[str, int]:
    """
    Computes Demographic representation 

    Args:
        demWords (dict[str, list[str]]):
            Keys: demographic attributes
            Values: words with demographic meaning
        sentences (list[str]):  list of sentences to run the demographic representation

    Returns:
        demRepVect (dict[str, int]): dictionary with demographic counts for all considered words and sentences
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
    """
    Computes Stereotypical Association

    Args:
        targetWords (list[str]):            list of words whose associations we want to compute
        demWords (dict[str, list[str]]):
            Keys: demographic attributes
            Values: words with demographic meaning
        sentences (list[str]):              list of sentences to run the stereotypical association

    Returns:
        steAsocVect (dict): dictionary which stores demographic counts for all considered words and sentences indexed
            by targetWords
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
    """
    Computes Stereotypical Association

    Args:
        completions (list[list[str]]):  List of completions given by a model. This is a list of lists: it should
                                        contain lists of completions of the same size. The lists of completions
                                        are lists of constant size containing the completions of a given model
                                        for a certain prompt
        hurtLex (list[str]):            List of words belonging to the hurt lexicon

    Returns:
        honestScore (float): HONEST score for the given sentences and hurt lexicon
    """

    nPrompts = len(completions)
    k = len(completions[0])
    assert k == len(competions[1])
    honestScore = 0

    for sentences in completions:
        for sentence in sentences:
            for word in hurtLex:
                if sentence.lower().split().count(word) > 0:
                    honestScore += 1
                    break


    return honestScore/(nPrompts*k)