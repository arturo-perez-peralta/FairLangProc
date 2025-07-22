import os
import sys

import pytest

from FairLangProc.metrics import DemRep, StereoAsoc, HONEST


#=======================================
# TEST VARIABLES
#=======================================

# Demographical representation
GENDERED_WORDS = {
    'male': ['he', 'him', 'his'],
    'female': ['she', 'her', 'actress', 'hers']
    }

ATTRIBUTES = GENDERED_WORDS.keys()

SENTENCES = [
    'She is such a good match to him.',
    'He is trying way too hard to be an actor.',
    'Her mother is trying to make ends meet.'
    'My aunt is baking, do you want to try?'
]


# Stereotypical association
TARGET_WORDS = ['mother', 'baking']

# HONEST
COMPLETIONS = [
    ['he is so stupid', 'he is such a fool', 'he is so misunderstood'],
    ['she is so bossy', 'she is an incompetent manager', 'she does what is necessary'],
    ['they were so agreeable', 'they were so nice to us', 'they showed hospitality']
]

COMPLETIONS_DIFFERENT_LENGTH = [
    ['he is so stupid', 'he is such a fool', 'he is so misunderstood'],
    ['she is so bossy', 'she is an incompetent manager'],
    ['they were so agreeable', 'they were so nice to us', 'they showed hospitality']
]

COMPLETIONS_WITHOUT_LIST = [
    ['he is so stupid', 'he is such a fool', 'he is so misunderstood'],
    ('she is so bossy', 'she is an incompetent manager', 'she does what is necessary'),
    ['they were so agreeable', 'they were so nice to us', 'they showed hospitality']
]

HURTLEX = ['fool', 'stupid', 'incompetent']

#=======================================
# DEMOGRAPHIC REPRESENTATION TESTS
#=======================================


def test_demographic_representation_type():
    DR = DemRep(sentences = SENTENCES, demWords = GENDERED_WORDS)
    assert isinstance(DR, dict)

def test_demographic_representation_keys():
    DR = DemRep(sentences = SENTENCES, demWords = GENDERED_WORDS)
    assert len(DR.keys()) == 2

def test_demographic_representation_values():
    DR = DemRep(sentences = SENTENCES, demWords = GENDERED_WORDS)
    assert DR['male'] == 1
    assert DR['female'] == 2

def test_demographic_representation_empty_demwords():
    DR = DemRep(sentences = SENTENCES, demWords = {})
    assert DR == {}

def test_demographic_representation_empty_sentences():
    DR = DemRep(sentences = [], demWords = GENDERED_WORDS)
    assert len(DR.keys()) == 2
    assert DR['male'] == 0
    assert DR['female'] == 0

def test_demographic_representation_empty_demwords_sentences():
    DR = DemRep(sentences = [], demWords = {})
    assert DR == {}


#=======================================
# STEREOTYPICAL ASSOCIATION TESTS
#=======================================

def test_stereorep_type():
    ST = StereoAsoc(sentences = SENTENCES, demWords = GENDERED_WORDS, targetWords = TARGET_WORDS)
    assert isinstance(ST, dict)
    for key in ST.keys():
        assert isinstance(ST[key], dict)

def test_stereorep_keys():
    ST = StereoAsoc(sentences = SENTENCES, demWords = GENDERED_WORDS, targetWords = TARGET_WORDS)
    assert len(ST.keys()) == 2
    for key in ST.keys():
        assert len(ST[key].keys()) == 2

def test_stereorep_values():
    ST = StereoAsoc(sentences = SENTENCES, demWords = GENDERED_WORDS, targetWords = TARGET_WORDS)
    assert ST['mother']['male'] == 0
    assert ST['mother']['female'] == 1
    assert ST['baking']['male'] == 0
    assert ST['baking']['female'] == 0

def test_stereorep_empty_target():
    ST = StereoAsoc(sentences = SENTENCES, demWords = GENDERED_WORDS, targetWords = [])
    assert ST == {}

def test_stereorep_empty_dem():
    ST = StereoAsoc(sentences = SENTENCES, demWords = {}, targetWords = TARGET_WORDS)
    assert ST['mother'] == {}
    assert ST['baking'] == {}

def test_stereorep_empty_sentences():
    ST = StereoAsoc(sentences = [], demWords = GENDERED_WORDS, targetWords = TARGET_WORDS)
    assert ST['mother']['male'] == 0
    assert ST['mother']['female'] == 0
    assert ST['baking']['male'] == 0
    assert ST['baking']['female'] == 0

def test_stereorep_empty_dem_sentences():
    ST = StereoAsoc(sentences = [], demWords = {}, targetWords = TARGET_WORDS)
    assert ST['mother'] == {}
    assert ST['baking'] == {}

def test_stereorep_empty_dem_target():
    ST = StereoAsoc(sentences = SENTENCES, demWords = {}, targetWords = [])
    assert ST == {}

def test_stereorep_empty_sentences_target():
    ST = StereoAsoc(sentences = [], demWords = GENDERED_WORDS, targetWords = [])
    assert ST == {}

def test_stereorep_empty_dem_sentences_target():
    ST = StereoAsoc(sentences = [], demWords = {}, targetWords = [])
    assert ST == {}



#=======================================
# HONEST TESTS
#=======================================

def test_honest_type():
    honest = HONEST(completions = COMPLETIONS, hurtLex = HURTLEX)
    assert isinstance(honest, float)

def test_honest_value():
    honest = HONEST(completions = COMPLETIONS, hurtLex = HURTLEX)
    assert abs(honest - 1/3) < 1e-15

def test_honest_empty_hurt():
    honest = HONEST(completions = COMPLETIONS, hurtLex = [])
    assert abs(honest - 0.0) < 1e-15

def test_honest_empty_completions():
    with pytest.raises(AssertionError) as excinfo:
        honest = HONEST(completions = [], hurtLex = HURTLEX)
    assert "completions is empty" in excinfo

def test_honest_not_list():
    with pytest.raises(AssertionError) as excinfo:
        honest = HONEST(completions = {}, hurtLex = HURTLEX)
    assert "completions is not a list" in excinfo

def test_element_not_list():
    with pytest.raises(AssertionError) as excinfo:
        honest = HONEST(completions = COMPLETIONS_WITHOUT_LIST, hurtLex = HURTLEX)
    assert "completions is not a list of lists" in excinfo

def test_honest_different_length():
    with pytest.raises(AssertionError) as excinfo:
        honest = HONEST(completions = COMPLETIONS_DIFFERENT_LENGTH, hurtLex = HURTLEX)
    assert "Number of completions is not uniform" in excinfo