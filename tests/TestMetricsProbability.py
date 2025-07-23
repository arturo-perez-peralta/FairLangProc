import os
import sys

import torch
import pytest

from FairLangProc.metrics import LPBS, CBS, CPS, AUL

#=======================================
#           TEST CLASES
#=======================================


class AttrDict(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self

    def __repr__(self):
        return f"{self.__class__.__name__}({dict.__repr__(self)})"
    

class DummyModel:
    def assing_logits_mask(self, input_ids, logits, b, t):
        input_sum = input_ids[b].sum().item()

        # Logic depending on sentence and word
        logits[b, t, 200] = 5.0 + input_sum % 3     # doctor
        logits[b, t, 201] = -5.0 + input_sum % 4    # nurse
        logits[b, t, 202] = 15.0 - input_sum % 5    # engineer

        logits[b, t, 300] = 5.0 + input_sum % 2     # science
        logits[b, t, 301] = -5.0 + input_sum % 6    # art
        logits[b, t, 302] = 15.0 - input_sum % 7    # math

        logits[b, t, 400] = 10.0 + input_sum % 3    # he
        logits[b, t, 401] = -10.0 + input_sum % 4   # she
        logits[b, t, 402] = -15.0 + input_sum % 10  # it


    def assing_logits(self, input_ids, logits):
        
        ids = [200, 201, 202, 300, 301, 302, 400, 401, 402]
        batch_size, seq_len = input_ids.shape

        for b in range(batch_size):
            for t in range(seq_len):
                if input_ids[b, t] in ids:
                    logits[b, t, input_ids[b, t]] = 5.0 + input_ids[b, t] % 3 


    def __call__(self, input_ids = None, **kwargs):
        batch_size, seq_len = input_ids.shape

        logits = torch.zeros(batch_size, seq_len, 30522)

        mask_token_id = 103
        mask_positions = (input_ids == mask_token_id)
        noMask = mask_positions.sum().item() == 0  

        # introduce dependency on input_ids
        if noMask:
            self.assing_logits(input_ids, logits)
        else:
            for b in range(batch_size):
                for t in range(seq_len):
                    if mask_positions[b, t]:
                        self.assing_logits_mask(input_ids, logits, b, t)

        return AttrDict(logits=logits)
    

class DummyTokenizer:
    pad_token_type_id = 101
    pad_token_id = 101
    cls_token_id = 102
    mask_token_id = 103
    hash_map_tokens = {
        '[PAD]': pad_token_type_id,
        '[CLS]': cls_token_id,
        '[MASK]': mask_token_id,
        'doctor': 200,
        'nurse': 201,
        'engineer': 202,
        'science': 300,
        'art': 301,
        'math': 302,
        'he': 400,
        'she': 401,
        'it': 402,
    }

    def __init__(self):
        return
    
    def __call__(self, sentences, padding=True, return_tensors="pt"):
        split = [["[CLS]"] + sentence.split() for sentence in sentences]
        maxLen = max([len(sentence) for sentence in split])
        ids = [[self.convert_tokens_to_ids(word) for word in sentence] for sentence in split]
        if padding:
            for i in range(len(ids)):
                ids[i] += [self.pad_token_id] * (maxLen - len(ids[i]))

        return AttrDict(**{"input_ids": torch.tensor(ids)})

    def tokenize(self, word):
        return [word]

    def convert_tokens_to_ids(self, token):
        return self.hash_map_tokens.get(token, 100)


#=======================================
#           TEST VARIABLES
#=======================================

MODEL = DummyModel()
TOKENIZER = DummyTokenizer()

SENTENCES = [
    "[MASK] is a [MASK]",
    "Is [MASK] a [MASK] ?",
    "[MASK] teaches [MASK]"
]

TARGET_WORDS_LPBS = [
    ("he", "she"),
    ("he", "she"),
    ("he", "she")
]

TARGET_WORDS_CBS = [
    ("he", "she", "it"),
    ("he", "she", "it"),
    ("he", "she", "it")
]

FILL_WORDS = [
    'engineer',
    'doctor',
    'math',
]

MASK_INDICES = [0, 0, 0]

PLL_SENTENCES = [
    'he is an exemplary doctor',
    'she is an exemplary doctor',
    'it is an exemplary doctor',
]

TARGET_WORDS_CPS = [
    'doctor',
    'doctor',
    'doctor',
]

#=======================================
#           CBS TESTS
#=======================================

def test_lpbs_type():
    LPBSscore = LPBS(
        model = MODEL,
        tokenizer = TOKENIZER,
        sentences = SENTENCES,
        target_words = TARGET_WORDS_LPBS,
        fill_words = FILL_WORDS,
        mask_indices = MASK_INDICES
    )
    assert isinstance(LPBSscore, torch.Size), f"Wrong format: expected {torch.Size}, got {type(LPBSscore)}"
    assert LPBSscore.shape == torch.Size([3]), f"Wrong output shape ({LPBSscore.shape}), expected value ({torch.Size([3])})"

def test_lpbs_value():
    LPBSscore = LPBS(
        model = MODEL,
        tokenizer = TOKENIZER,
        sentences = SENTENCES,
        target_words = TARGET_WORDS_LPBS,
        fill_words = FILL_WORDS,
        mask_indices = MASK_INDICES
    )
      
    assert abs(LPBSscore[0].item() - 1.0) < 1e-5, f"Mismatch between output ({LPBSscore[0].item()}) and expected value ({1.0})"
    assert abs(LPBSscore[1].item() - (-3.0)) < 1e-5, f"Mismatch between output ({LPBSscore[1].item()}) and expected value ({-3.0})"
    assert abs(LPBSscore[2].item() - (-2.0)) < 1e-5, f"Mismatch between output ({LPBSscore[2].item()}) and expected value ({-2.0})"

def test_lpbs_less_target():
    with pytest.raises(AssertionError) as excinfo:
        LPBSscore = LPBS(
            model = MODEL,
            tokenizer = TOKENIZER,
            sentences = SENTENCES[:1],
            target_words = TARGET_WORDS_LPBS,
            fill_words = FILL_WORDS,
            mask_indices = MASK_INDICES
        )
    assert "Different number of sentences" in excinfo, f"Wrong error trace"

def test_lpbs_less_target():
    with pytest.raises(AssertionError) as excinfo:
        LPBSscore = LPBS(
            model = MODEL,
            tokenizer = TOKENIZER,
            sentences = SENTENCES,
            target_words = TARGET_WORDS_LPBS[:1],
            fill_words = FILL_WORDS,
            mask_indices = MASK_INDICES
        )
    assert "Different number of sentences and target words" in excinfo, f"Wrong error trace"

def test_lpbs_less_fill():
    with pytest.raises(AssertionError) as excinfo:
        LPBSscore = LPBS(
            model = MODEL,
            tokenizer = TOKENIZER,
            sentences = SENTENCES,
            target_words = TARGET_WORDS_LPBS,
            fill_words = FILL_WORDS[:1],
            mask_indices = MASK_INDICES
        )
    assert "Different number of sentences and fill words" in excinfo, f"Wrong error trace"

def test_lpbs_less_masks():
    with pytest.raises(AssertionError) as excinfo:
        LPBSscore = LPBS(
            model = MODEL,
            tokenizer = TOKENIZER,
            sentences = SENTENCES,
            target_words = TARGET_WORDS_LPBS,
            fill_words = FILL_WORDS,
            mask_indices = MASK_INDICES[:1]
        )
    assert "Different number of sentences and mask indices" in excinfo, f"Wrong error trace"

def test_lpbs_no_pairs():
    with pytest.raises(AssertionError) as excinfo:
        LPBSscore = LPBS(
            model = MODEL,
            tokenizer = TOKENIZER,
            sentences = SENTENCES,
            target_words = [('he', 'she', 'it')]*3,
            fill_words = FILL_WORDS,
            mask_indices = MASK_INDICES
        )
    assert "Target words must consist of pairs of words" in excinfo, f"Wrong error trace"



#=======================================
#           CBS TESTS
#=======================================

def test_cbs_type():
    CBSscore = CBS(
        model = MODEL,
        tokenizer = TOKENIZER,
        sentences = SENTENCES,
        target_words = TARGET_WORDS_CBS,
        fill_words = FILL_WORDS,
        mask_indices = MASK_INDICES
    )
    assert isinstance(CBSscore, torch.Size), f"Wrong format: expected {torch.Size}, got {type(CBSscore)}"
    assert CBSscore.shape == torch.Size([3]), f"Wrong output shape ({CBSscore.shape}), expected value ({torch.Size([3])})"

def test_cbs_value():
    CBSscore = CBS(
        model = MODEL,
        tokenizer = TOKENIZER,
        sentences = SENTENCES,
        target_words = TARGET_WORDS_CBS,
        fill_words = FILL_WORDS,
        mask_indices = MASK_INDICES
    )
      
    assert abs(CBSscore[0].item() - 1/3) < 1e-5, f"Mismatch between output ({CBSscore[0].item()}) and expected value ({1/3})"
    assert abs(CBSscore[1].item() - 13/3) < 1e-5, f"Mismatch between output ({CBSscore[1].item()}) and expected value ({13/3})"
    assert abs(CBSscore[2].item() - 4.0) < 1e-5, f"Mismatch between output ({CBSscore[2].item()}) and expected value ({4.0})"

def test_cbs_less_target():
    with pytest.raises(AssertionError) as excinfo:
        CBSscore = CBS(
            model = MODEL,
            tokenizer = TOKENIZER,
            sentences = SENTENCES[:1],
            target_words = TARGET_WORDS_CBS,
            fill_words = FILL_WORDS,
            mask_indices = MASK_INDICES
        )
    assert "Different number of sentences" in excinfo, f"Wrong error trace"

def test_cbs_less_target():
    with pytest.raises(AssertionError) as excinfo:
        CBSscore = CBS(
            model = MODEL,
            tokenizer = TOKENIZER,
            sentences = SENTENCES,
            target_words = TARGET_WORDS_CBS[:1],
            fill_words = FILL_WORDS,
            mask_indices = MASK_INDICES
        )
    assert "Different number of sentences and target words" in excinfo, f"Wrong error trace"

def test_cbs_less_fill():
    with pytest.raises(AssertionError) as excinfo:
        CBSscore = CBS(
            model = MODEL,
            tokenizer = TOKENIZER,
            sentences = SENTENCES,
            target_words = TARGET_WORDS_CBS,
            fill_words = FILL_WORDS[:1],
            mask_indices = MASK_INDICES
        )
    assert "Different number of sentences and fill words" in excinfo, f"Wrong error trace"

def test_cbs_less_masks():
    with pytest.raises(AssertionError) as excinfo:
        CBSscore = CBS(
            model = MODEL,
            tokenizer = TOKENIZER,
            sentences = SENTENCES,
            target_words = TARGET_WORDS_CBS,
            fill_words = FILL_WORDS,
            mask_indices = MASK_INDICES[:1]
        )
    assert "Different number of sentences and mask indices" in excinfo, f"Wrong error trace"

#=======================================
#           CPS TESTS
#=======================================

def test_cps_type():
    CPSscore = CPS(
        model = MODEL,
        tokenizer = TOKENIZER,
        sentences = PLL_SENTENCES,
        target_words = TARGET_WORDS_CPS[:1]
    )
    assert isinstance(CPSscore, list), f"Wrong format: expected {list}, got {type(CPSscore)}"
    assert len(CPSscore) == 3, f"Wrong output shape ({len(CPSscore)}), expected value ({3})"

def test_cps_value():
    CPSscore = CPS(
        model = MODEL,
        tokenizer = TOKENIZER,
        sentences = PLL_SENTENCES,
        target_words = TARGET_WORDS_CPS[:1]
    )
    assert abs(CPSscore[0] - ( -5.3755054473)) < 1e-6, f"Mismatch between output ({CPSscore[0].item()}) and expected value ({-5.3755})"
    assert abs(CPSscore[1] - (-15.9847412109)) < 1e-6, f"Mismatch between output ({CPSscore[1].item()}) and expected value ({-15.9847})"
    assert abs(CPSscore[2] - (-16.9847259521)) < 1e-6, f"Mismatch between output ({CPSscore[2].item()}) and expected value ({-16.9847})"

def test_cps_less_target():
    with pytest.raises(AssertionError) as excinfo:
        CPSscore = CPS(
            model = MODEL,
            tokenizer = TOKENIZER,
            sentences = PLL_SENTENCES,
            target_words = TARGET_WORDS_CPS[:1]
        )
    assert "Number of sentences and target words must be the same" in excinfo, f"Wrong error trace"

def test_cps_less_sentences():
    with pytest.raises(AssertionError) as excinfo:
        CPSscore = CPS(
            model = MODEL,
            tokenizer = TOKENIZER,
            sentences = PLL_SENTENCES[:1],
            target_words = TARGET_WORDS_CPS
        )
    assert "Number of sentences and target words must be the same" in excinfo, f"Wrong error trace"

def test_cps_empty_sentences():
    with pytest.raises(AssertionError) as excinfo:
        CPSscore = CPS(
            model = MODEL,
            tokenizer = TOKENIZER,
            sentences = [],
            target_words = []
        )
    assert "Empty sentence list" in excinfo, f"Wrong error trace"

#=======================================
#           AUL TESTS
#=======================================

def test_aul_type():
    AULscore = AUL(
        model = MODEL,
        tokenizer = TOKENIZER,
        sentences = PLL_SENTENCES
    )
    assert isinstance(AULscore, list), f"Wrong format: expected {list}, got {type(AULscore)}"
    assert len(AULscore) == 3, f"Wrong output shape ({len(AULscore)}), expected value ({3})"

def test_aul_value():
    AULscore = AUL(
        model = MODEL,
        tokenizer = TOKENIZER,
        sentences = PLL_SENTENCES
    )
    assert abs(AULscore[0] - (-1.3468990325)) < 1e-6, f"Mismatch between output ({AULscore[0].item()}) and expected value ({-1.34689})"
    assert abs(AULscore[1] - (-1.3449568748)) < 1e-6, f"Mismatch between output ({AULscore[1].item()}) and expected value ({-1.34495})"
    assert abs(AULscore[2] - (-1.3521032333)) < 1e-6, f"Mismatch between output ({AULscore[2].item()}) and expected value ({-1.35210})"

def test_aul_empty_sentences():
    with pytest.raises(AssertionError) as excinfo:
        AULscore = AUL(
            model = MODEL,
            tokenizer = TOKENIZER,
            sentences = []
        )
    assert "Empty sentence list" in excinfo, f"Wrong error trace"