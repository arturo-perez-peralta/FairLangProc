import os
import sys
from math import sqrt

import torch
import pytest

from FairLangProc.metrics import WEAT

#================================
#   TEST CLASSES
#================================


class AttrDict(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__dict__ = self

    def __repr__(self):
        return f"{self.__class__.__name__}({dict.__repr__(self)})"
    
    def to(self, device):
        return self
    

class DummyModel:
    privileged_tokens = [200, 201, 202, 300, 301, 302, 400, 401, 402, 500, 501, 502]

    def primitive_embedding(self, id):
        remainder = id % 100
        output = torch.Tensor([0,0])
        if remainder == 2:
            output = torch.Tensor([1, 0])
        elif remainder == 3:
            output = torch.Tensor([-1,0])
        elif remainder == 4:
            output = torch.Tensor([0, 1])
        elif remainder == 5:
            output = torch.Tensor([0,-1])
        return output

    def __call__(self, input_ids = None, output_hidden_states=True, **kwargs):
        batch_size, seq_len = input_ids.shape
        embedding = torch.zeros(batch_size, 2)
        for b in range(batch_size):
            for s in range(seq_len):
                embedding[b] += self.primitive_embedding(input_ids[b, s])
        return AttrDict(embedding=embedding)
    
    def to(self, device):
        return self
    
    def eval(self):
        return
    

class DummyTokenizer:
    pad_token_type_id = 101
    pad_token = 101
    cls_token_id = 102
    mask_token_id = 103
    hash_map_tokens = {
        '[PAD]': pad_token_type_id,
        '[CLS]': cls_token_id,
        '[MASK]': mask_token_id,
        'secretary': 200,
        'nurse': 201,
        'teacher':202,
        'engineer': 300,
        'firefighter': 301,
        'banker': 302,
        'he': 400,
        'actor': 401,
        'son': 402,
        'she': 500,
        'actress': 501,
        'daughter': 502
    }

    def __init__(self):
        return
    
    def __call__(self, sentences, padding=True, return_tensors="pt"):
        split = [sentence.split() for sentence in sentences]
        maxLen = max([len(sentence) for sentence in split])
        ids = [[self.convert_tokens_to_ids(word) for word in sentence] for sentence in split]
        if padding:
            for i in range(len(ids)):
                lenId = len(ids[i])
                if lenId < maxLen:
                    ids[i] = ids[i] + [self.pad_token_id for _ in range(maxLen - lenId)] 
        return AttrDict(input_ids = torch.tensor(ids))

    def tokenize(self, word):
        return [word]

    def convert_tokens_to_ids(self, token):
        return self.hash_map_tokens.get(token, 100)
    
    def to(self, device):
        return self

class DummyWEAT(WEAT):
    def _get_embedding(self, outputs):
        return outputs.embedding[0]



#================================
#   TEST VARIABLES
#================================

MODEL = DummyModel()
TOKENIZER = DummyTokenizer()
TEST_WEAT = DummyWEAT(model = MODEL, tokenizer = TOKENIZER)

X = torch.tensor([[1]*12+[0]*6, [0]*6+[1]*12], dtype = float).transpose(0, 1)
Y = torch.tensor([[1,-1]*2+[0]*2, [0]*2+[1]*3 + [-1]], dtype = float).transpose(0, 1)

COSXY = [[1.0, -1.0, 1/sqrt(2), -1/sqrt(2), 0.0, 0.0]]*6 \
    + [[1/sqrt(2), -1/sqrt(2), 1.0, 0.0, 1/sqrt(2), -1/sqrt(2)]]*6 \
    + [[0.0, 0.0, 1/sqrt(2), 1/sqrt(2), 1.0, -1.0]]*6
COSXY = torch.tensor(COSXY)

XEFFECT =  torch.tensor([[1]*4+[0]*2, [0]*2+[1]*4], dtype = float).transpose(0, 1)
YEFFECT = -torch.tensor([[1]*4+[0]*2, [0]*2+[1]*4], dtype = float).transpose(0, 1)
AEFFECT =  torch.tensor([[-1]*4+[0]*2, [0]*2+[1]*4], dtype = float).transpose(0, 1)
BEFFECT = -torch.tensor([[1]*4+[0]*2, [0]*2+[1]*4], dtype = float).transpose(0, 1)

WORDSX = ['he', 'actor', 'son']
WORDSY = ['she', 'actress', 'daughter']
WORDSA = ['banker', 'engineer', 'firefighter']
WORDSB = ['secretary', 'nurse', 'teacher']

#================================
#   EMBEDDING TESTS
#================================

def test_type_cosine_similarity():
    X = torch.tensor([[1]*12+[0]*6, [0]*6+[1]*12], dtype = float).transpose(0,1)
    Y = torch.tensor([([1,-1]*2+[0]*2)*3, ([0]*2+[1]*4)*3], dtype = float).transpose(0,1)
    output = TEST_WEAT.cosine_similarity(X, Y)
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 18
    assert output.shape[1] == 6

def test_value_cosine_similarity():
    X = torch.tensor([[1]*12+[0]*6, [0]*6+[1]*12], dtype = float).transpose(0,1)
    Y = torch.tensor([([1,-1]*2+[0]*2)*3, ([0]*2+[1]*4)*3], dtype = float).transpose(0,1)
    output = TEST_WEAT.cosine_similarity(X, Y)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            assert abs(output[i,j] - COSXY[i,j]) < 1e-7, f"Mismatch between output ({output[i,j]}) and expected value ({COSXY[i,j]})"

def test_type_effect_size():
    result = TEST_WEAT.effect_size(XEFFECT, YEFFECT, AEFFECT, BEFFECT)
    assert isinstance(result, float), f"Wrong format: expected {float}, got {type(result)}"

def test_value_effect_size():
    result = TEST_WEAT.effect_size(XEFFECT, YEFFECT, AEFFECT, BEFFECT)
    assert abs(result - 1.699794717779) < 1e-7, f"Mismatch between output ({result}) and expected value ({1.69979})"

def test_type_metric():
    result = TEST_WEAT.metric(WORDSX, WORDSY, WORDSA, WORDSB)
    assert isinstance(result, dict), f"Wrong format: expected {dict}, got {type(result)}"