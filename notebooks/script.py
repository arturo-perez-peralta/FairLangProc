#!/usr/bin/env python
# coding: utf-8


#---------------------------------------------------------
#=========================================================
#                   FAIRLANGPROC SCRIPT
#=========================================================
#---------------------------------------------------------

# The following script is a compilation of the demo code found in the companion paper of the FairLangProc package.
#   Preprint: https://arxiv.org/abs/2508.03677
#   Repo: https://github.com/arturo-perez-peralta/FairLangProc/tree/main
# This code is derived from the notebooks in the /notebooks folder in the companion repository, and we encourage using  the notebooks instead,
# although this script provides a single, unified version of the whole code.
# The script is divided in four sections in the same way as the original paper.
#   1. Fairness Datasets.
#   2. Fairness Metrics. 
#   3. Fairness Processors.
#   4. Debiasing BERT.
# Each section is independent of the rest in order to facilitate copy-pasting concrete sections.

# In order to run the imports from a local folder set LOCAL to False and change ROOT_PATH with the location of the FairLangProc package.
LOCAL = True
if LOCAL:
    import os
    import sys
    ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) \
        if "__file__" in globals() else os.path.abspath("..")
    sys.path.insert(0, ROOT_PATH)

import gc
import torch

#=========================================================
#                   FAIRNESS DATASETS
#=========================================================

# This section showcases the datasets available in the module. We will take a glance at the task each one of them proposes as well as the format.
# The data sets were downloaded from the github repository https://github.com/i-gallegos/Fair-LLM-Benchmark.
# In order to run the code, one needs to first download the data. This can be done with the command 
# 'git clone https://github.com/i-gallegos/Fair-LLM-Benchmark.git'

# Reference: Gallegos, I. O., Rossi, R. A., Barrow, J., Tanjim, M. M., Kim, S., Dernoncourt, F., ... & Ahmed, N. K. (2024). Bias and fairness in large language models: A survey. Computational Linguistics, 1-79.
# Preprint: https://arxiv.org/abs/2309.00770.

#---------------------------------------------------------
#                   BiasDataLoader
#---------------------------------------------------------

from FairLangProc.datasets import BiasDataLoader

# The `BiasDataLoader` allows to interact with the datasets. 
# The arguments of `BiasDataLoader` are:
#   `dataset`: the name of the dataset (shown above).
#   `config`: which further specifies the dataset.
#   `format`: accepts either `raw` for raw pd/txt format, `pt` for PyTorch dataset, `hf` for hugging face data set.


# The empty initialization of `BiasDataLoader` method shows the available datasets:
BiasDataLoader()

# An empty config shows available configs (if there are more any):
BiasDataLoader(dataset = 'BBQ')


# We can now insert any of the configurations to extract the corresponding data set:
ageBBQ = BiasDataLoader(dataset = 'BBQ', config = 'Age')
print(ageBBQ['data'][0])


# We can also change the format of the outputs with the `format` parameter:
ageBBQraw = BiasDataLoader(dataset = 'BBQ', config = 'Age', format = 'raw')
print(ageBBQraw['data'].head())
print(type(ageBBQraw['data']))
print(type(ageBBQ['data']))



#=========================================================
#                   FAIRNESS METRICS
#=========================================================

# This section showcases the metrics available in the `FairnessMetrics` submodule. Basically, there are three different types of metrics to assess bias in LLMS:
#   1. Embedding based: based on association tests on the embeddings of both sensitive words and words with certain attributes (professions, occupations,...) 
#   2. Probability based: computed using a masked language model to compute the probabilities of masked tokens.
#   3. Generated text based: counts the lexicon used in the generations of certain models.

#---------------------------------------------------------
#                   Embedding based
#---------------------------------------------------------

# Embedding based metrics boil down to the `WEAT` metric in one form or another. Our implementation is flexible enough to allow for association tests at the word, sentence and contextualized levels.
# The implementation of the text can be accessed through the `FairnessMetrics.Embedding` subfolder.
# The association test assumes to sets of words, $W_1, W_2$ and two sets of attributes, $A_1, A_2$. It then computes the association by computing averages of the cosine similarities of elements of the two groups.
# A simple demostration can be found in the cell code below:

# Imports
from transformers import AutoTokenizer, AutoModel
from FairLangProc.metrics import WEAT

# Concrete implementation of the `WEAT` abstract class.
class BertWEAT(WEAT):
    def _get_embedding(self, outputs):
        return outputs.last_hidden_state[:, 0, :]

# Initialize tokenizer, model, weat test class, association words.
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
weatClass = BertWEAT(model = model, tokenizer = tokenizer)
math = ['math', 'algebra', 'geometry', 'calculus', 'equations']
arts = ['poetry', 'art', 'dance', 'literature', 'novel']
masc = ['male', 'man', 'boy', 'brother', 'he']
femn = ['female', 'woman', 'girl', 'sister', 'she']

# Compute WEAT test.
weatVal = weatClass.metric(
    W1_words = math, W2_words = arts,
    A1_words = masc, A2_words = femn,
    pval = False
    )
print(weatVal)

del model, tokenizer, weatClass
gc.collect()
torch.cuda.empty_cache()

#---------------------------------------------------------
#                   Probability based
#---------------------------------------------------------

#                   Masked tokens 
#---------------------------------------------------------

# These metrics aim to measure bias by computing the probability of certain tokens inside a sentence. In the package we have opted to implement `LPBS` and its generalization to non-binary sensitiva variables, `CBS`. 
# Our implementation assumes that the masked sentence only has two masks, one which should be substituted by a sensitive word (suchs as "man" or "woman")
# and another one which should be replaced by the fill word (such as the occupation of the person), but it is flexible enough so the user may specify
# (using a list of indices) which of the masks goes where. In particular, the user should specify the position of the masks corresponding to sensitive words.
# If the mask indices are not introduced, the program assumes that it should always consider the first mask of each sentence.

# Imports
from transformers import AutoTokenizer, AutoModelForMaskedLM
from FairLangProc.metrics import LPBS, CBS

# Masked sentences
sentences = [
    "[MASK] is a [MASK].",
    "[MASK] is a [MASK].",
    "The [MASK] was a [MASK]."
]

# Target words (with demographic information)
target_words = [
    ("John", "Mary"),
    ("He", "She"),
    ("man", "woman")
]
# Fill words
fill_words = [
    "engineer",
    "nurse",
    "doctor"
]

# Position of the mask associated with the target word in each sentence
mask_indices = [0, 0, 1]

# Initialize model and tokenizer
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Compute LPBS
LPBSscore = LPBS(
    model = model,
    tokenizer = tokenizer,
    sentences = sentences,
    target_words = target_words,
    fill_words = fill_words,
    mask_indices = mask_indices
)

# CBS has a similar format
target_words = [
    ("John", "Mamadouk", "Liu"),
    ("white", "black", "asian"),
    ("white", "black", "asian")
]
sentences = [
    "[MASK] is a [MASK]",
    "The [MASK] kid got [MASK] results",
    "The [MASK] kid wanted to be a [MASK]"
]
fill_words = [
    "engineer",
    "outstanding",
    "doctor"
]
mask_indices = [0, 1, 1]

CBSscore = CBS(
    model = model,
    tokenizer = tokenizer,
    sentences = sentences,
    target_words = target_words,
    fill_words = fill_words,
    mask_indices = mask_indices
)

# Show LPBS and CBS:
print(LPBSscore)
print(CBSscore)

del model, tokenizer
gc.collect()
torch.cuda.empty_cache()


#                   Pseudo-loglikelihood
#---------------------------------------------------------

# These methods are based on the computation of the pseudo-loglikelihood of a sentence $S$.
# In particular, we have implemented:
#   `CPS`: computes the pseudo-loglikelihood of the unmodified tokens of two versions of the same sentence (one stereotyping and the other one anti-stereotyping).
#   `AUL`: computes the all unmasked likelihood of all tokens.

# Imports
from transformers import AutoModelForMaskedLM, AutoTokenizer
from FairLangProc.metrics import CPS, AUL

# Model and tokenizer
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Setences with a stereotyping and and an anti-stereotyping version
sentences = [
    'The actor did a terrible job',
    'The actress did a terrible job',
    'The doctor was an exemplary man',
    'The doctor was an exemplary woman'
]

# Target words for CPS (unmodified tokens)
target_words = [
    'actor',
    'actress',
    'man',
    'woman'
]

# Compute CPS and AUL
CPSscore = CPS(
    model = model,
    tokenizer = tokenizer,
    sentences = sentences,
    target_words = target_words
)
AULScore = AUL(
    model = model,
    tokenizer = tokenizer,
    sentences = sentences
)

# Show metrics
print(CPSscore)
print(AULScore)

del model, tokenizer
gc.collect()
torch.cuda.empty_cache()

#---------------------------------------------------------
#                   Generated text
#---------------------------------------------------------

#                   Pseudo-loglikelihood
#---------------------------------------------------------

# These metrics count the amount of times certain words appear in the generated text. Then these counts can be normalized to compare them to a certain benchmark distribution (e.g. uniform).
# We have implemented Demographic Representation and Stereotypical Association through the `DemRep` and `StereoAsoc` methods, respectively.

# Imports
from FairLangProc.metrics import DemRep, StereoAsoc

# Demographic words
gendered_words = {
    'male': ['he', 'him', 'his'],
    'female': ['she', 'her', 'actress', 'hers']
    }

# Sentences (morally they should be generated by an LLM)
sentences = [
    'She is such a good match to him.',
    'He is trying way too hard to be an actor.',
    'Her mother is trying to make ends meet.'
    'My aunt is baking, do you want to try?'
    ]

# Target words for stereotypical association
target_words = ['mother', 'baking']

# Compute the metrics
DR = DemRep(
    sentences = sentences,
    demWords = gendered_words
    )
ST = StereoAsoc(
    sentences = sentences,
    demWords = gendered_words,
    targetWords = target_words
    )

# Print metrics
print(DR)
print(ST)


#                   Lexicon-based
#---------------------------------------------------------

# These metrics measure the ammount of times certain lexicon (denominated the *hurt lexicon*) appears in the top k completions.
# The HONEST metric is specially relevant having its own data set associated which allows for quick evaluation of a given LM.

# Improts
from transformers import AutoModelForCausalLM, AutoTokenizer
from FairLangProc.metrics import HONEST

# Model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

# Compile a hurt lexicon
hurtLex = ['fool', 'bitch', 'stupid', 'incompetent', 'idiot', 'dumb']

# Generate sentences with GPT-2 from a given corpus
corpus = [
    'I cannot believe she did that, she is so',
    'I cannot believe he did that, he is so',
    'She tries and she tries to no avail. She is such a',
    'He tries and she tries to no avail. He is such a',
    ]

completions = []

for prompt in corpus:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids,
        max_length=15,
        num_return_sequences=2,
        do_sample=True,
        early_stopping=True
    )
    completions.append([tokenizer.decode(output, skip_special_tokens=True) for output in outputs])

# Compute HONEST
honestScore = HONEST(
    completions = completions,
    hurtLex = hurtLex
    )

# Print metric
print(honestScore)

del model, tokenizer
gc.collect()
torch.cuda.empty_cache()


#=========================================================
#                   FAIRNESS PROCESSORS
#=========================================================

# This section showcases the different fairness processors we have implemented, showing a simple use case in which we debias the BERT model.
# Fairness processors can be classified according to the part of the machine learning pipeline they are introduced in:
#   1. Pre processors: if they are introduced before the model has been trained.
#   2. In processors: if they are introduced during the process of training the model.
#   3. Post processors: if they are introduced after the training step.
#   4. Intra processors: aditionally, we speak of *intra processors* when refering to fairness methods that do not modify a model's parameters. This notion overlaps with that of post processors and can be deemed equivalent.
# 
# To showcase the implementation of these methods we will run then on the imdb data set without further considerations as it is only intended to serve as a proof of concept.

#---------------------------------------------------------
#                   Preliminaries
#---------------------------------------------------------

#                   Imports
#---------------------------------------------------------

# Standard libraries
import sys
import os

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Hugging face
from transformers import (
    BertForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import (
    load_dataset,
    Dataset
)


#                   Model
#---------------------------------------------------------

# Use GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load BERT
def get_bert():
    return BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
        )

TOKENIZER = AutoTokenizer.from_pretrained('bert-base-uncased')
HIDDEN_DIM_BERT = get_bert().config.hidden_size


#                   Processing the data
#---------------------------------------------------------

# load dataset
imdb = load_dataset("imdb")

# define the tokenize function
def tokenize_function(example):
    return TOKENIZER(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=128
        )

dataset = imdb.map(tokenize_function, batched=True)
dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "label"]
    )

# Train test split
train_dataset = dataset["train"].select(range(min(100, len(dataset["train"]))))
val_dataset = dataset["test"].select(range(min(100, len(dataset["test"]))))

# Trainer configuration
training_args = TrainingArguments(
    output_dir="./checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    fp16=True,
    save_safetensors=False, 
    weight_decay=0.1,
    logging_dir="./logs",
    logging_steps=10,
)


#                   Base model
#---------------------------------------------------------

BERT = get_bert()
trainer = Trainer(
    model=BERT,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(
        AdamW(BERT.parameters(), lr=1e-5, weight_decay=0.1),
        None
        )
)

trainer.train()
results = trainer.evaluate()
print(results)

del trainer, BERT
gc.collect()
torch.cuda.empty_cache()


#---------------------------------------------------------
#                   Pre-processors
#---------------------------------------------------------

# Pre processors are those methods that only affect the model's inputs and do not change their parameters. We have implemented:
#   1. Counterfactual Data Augmentation (CDA).
#   2. Projection based debiasing.
#   3. BLIND debiasing.

#                   CDA
#---------------------------------------------------------

# CDA is based on the idea of augmenting the data by flipping words with information of the sensitive attribute (e.g. feminine vs. masculine words).
# This procedure is implemented with the `transform_batch` function which is applied to a hugging face data set.

# Import
from FairLangProc.algorithms.preprocessors import CDA

# Counterfactual pairs
gendered_pairs = [
    ('he', 'she'),
    ('him', 'her'),
    ('his', 'hers'),
    ('actor', 'actress'),
    ('priest', 'nun'),
    ('father', 'mother'),
    ('dad', 'mom'),
    ('daddy', 'mommy'),
    ('waiter', 'waitress'),
    ('James', 'Jane')
    ]

# Run CDA
cda_train = Dataset.from_dict(
        CDA(imdb['train'].select(range(min(100, len(imdb['train']))))[:], pairs = dict(gendered_pairs))
)
train_CDA = cda_train.map(tokenize_function, batched=True)
train_CDA.set_format(
    type="torch", columns=["input_ids", "attention_mask", "label"]
)
train_CDA = train_CDA.select(range(min(100, len(train_CDA))))

text_key = 'text'

# Check differences
print(f'Lenght of original train data set: {len(train_dataset[text_key])}')
print(f'Lenght of CDA augmented train data set: {len(cda_train[text_key])}')

# Train model
CDAModel = get_bert()

trainer = Trainer(
    model=CDAModel,
    args=training_args,
    train_dataset=train_CDA,
    eval_dataset=val_dataset,
    optimizers=(
        AdamW(CDAModel.parameters(), lr=2e-5, weight_decay=0.01),
        None
        )
)

trainer.train()
results = trainer.evaluate()
print(results)

del trainer, CDAModel
gc.collect()
torch.cuda.empty_cache()

#                   BLIND debiasing
#---------------------------------------------------------


# BLIND debiasing incoroporates a classifier whom is tasked with identifying whether the base model will succeed in the task for a given training instance.
# The model then reweights each training instance depending on the probability that this auxiliary model, $g_{B}$, assigns to the base model that it will correctly perform the task. The loss is modified accordingly.

# The implementation of BLIND is given by the `BLINDModel` abstract class, which requires the implementation of three abstract methods: 
#   1. `_get_loss`: sets the `self.loss_fct` attribute to the desired loss.
#   2. `_loss`: computes the value of `self.loss_fct` for a training instance.
#   3. `_get_embedding`: which retrieves the hidden representation of a given input.
# 
# We have implemented the `BLINDModelForClassification` to handle classification tasks, which sets the loss function to the usual cross-entropy loss and only requires the definition of `_get_embedding`.
# Below we implement a custom class for the BERT model.

# Import
from FairLangProc.algorithms.preprocessors import BLINDTrainer

# Get BERT and construct a classifier
BLINDModel = get_bert()
BLINDClassifier = nn.Sequential(
      nn.Linear(HIDDEN_DIM_BERT, HIDDEN_DIM_BERT),
      nn.ReLU(),
      nn.Linear(HIDDEN_DIM_BERT, 2)
)

# Implement _get_embedding
class BLINDBERTTrainer(BLINDTrainer):
    def _get_embedding(self, inputs):
        return self.model.bert(
            input_ids = inputs.get("input_ids"),
            attention_mask = inputs.get("attention_mask"),
            token_type_ids = inputs.get("token_type_ids")
            ).last_hidden_state[:,0,:]

# Set up the trainer
trainer = BLINDBERTTrainer(
    blind_model = BLINDClassifier,
    blind_optimizer = lambda x: AdamW(x, lr=1e-5, weight_decay=0.1),
    temperature = 1.0,
    gamma = 2.0,
    alpha = 1.0,
    model = BLINDModel,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    optimizers=(
        AdamW(BLINDModel.parameters(), lr=1e-5, weight_decay=0.1),
        None
        )
)

# Train the model
trainer.train()
results = trainer.evaluate()
print(results)

del trainer, BLINDModel, BLINDClassifier
gc.collect()
torch.cuda.empty_cache()

#                   Projection-based debiasing
#---------------------------------------------------------


# Projection based debiasing identifies a bias subpsace by performing PCA on the difference of the hidden representation of counterfactual pairs of words or sentences.
# Then, the hidden representation of a given input is debiased by computing its projection on the bias-free subspace that's orthogonal to the bias subspace.

# The implementation of projection based debiasing is given by the `SentDebiasModel` abstract class, which requires the implementation of three abstract methods:
#   1. `_get_loss`: sets the `self.loss_fct` attribute to the desired loss.
#   2. `_loss`: computes the value of `self.loss_fct` for a training instance.
#   3. `_get_embedding`: which retrieves the hidden representation of a given input.

# We have implemented the `SentDebiasForSequenceClassification` to handle classification tasks, which sets the loss function to the usual cross-entropy loss and only requires the definition of `_get_embedding`.
# Below we implement a custom class for the BERT model.

# Imports
from FairLangProc.algorithms.preprocessors\
import SentDebiasForSequenceClassification

# Counterfactual pairs
gendered_pairs = [('he', 'she'), ('his', 'hers'), ('monk', 'nun')]

# Implement _get_embedding
model = get_bert()
class SentDebiasBert(SentDebiasForSequenceClassification):        
    def _get_embedding(
            self,
            input_ids,
            attention_mask = None,
            token_type_ids = None
            ):
        return self.model.bert(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
            ).last_hidden_state[:,0,:]

# Train model
EmbedModel = SentDebiasBert(
    model = model,
    config = None,
    tokenizer = TOKENIZER,
    word_pairs = gendered_pairs,
    n_components = 1,
    n_labels = 2
)

trainer = Trainer(
    model=EmbedModel,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(
        AdamW(EmbedModel.parameters(), lr=1e-5, weight_decay=0.1),
        None
        )
)

trainer.train()
results = trainer.evaluate()
print(results)

del trainer, EmbedModel, model
gc.collect()
torch.cuda.empty_cache()


#---------------------------------------------------------
#                   In-processors
#---------------------------------------------------------

# In processors are those methods that change the way the model is trained. In particular we have implemented:
#   1. ADELE (adapter based debiasing).
#   2. Selective updating.
#   3. Regularizers.

#                   ADELE
#---------------------------------------------------------

# The ADELE method adopts an adapter-based approach where they include an adapter layer after each FNN layer of the transformer architecture.
# The adapter layer consists of a linear layer with an activation function and a bias, $r$. This layer has a smaller dimension than the corresponding FNN,
# compressing the data and providing a information bottleneck so the bias information gets discarded after carefully training the model.

# Imports
from adapters import AdapterTrainer
from FairLangProc.algorithms.inprocessors import DebiasAdapter

# Train model
DebiasAdapterInst = DebiasAdapter(
    model = get_bert(),
    adapter_config = "seq_bn"
    )
AdeleModel = DebiasAdapterInst.get_model()

trainer = AdapterTrainer(
    model=AdeleModel,
    args=training_args,
    train_dataset=train_CDA,
    eval_dataset=val_dataset,
    optimizers=(
        AdamW(AdeleModel.parameters(),lr=1e-5, weight_decay=0.1),
        None
        )
)

trainer.train()
results = trainer.evaluate()
print(results)

del trainer, AdeleModel, DebiasAdapterInst
gc.collect()
torch.cuda.empty_cache()


#                   Selective updating
#---------------------------------------------------------


# Selective updating aims to selectively update some of the model's parameters. The method `selective_unfreezing` allows to freeze all of the model's parameters with the exception of certain parameters
# specificied by their names.


# Imports
from FairLangProc.algorithms.inprocessors import selective_unfreezing

# Freeze the whole model and unfreeze the attention layers
FrozenBert = get_bert()
selective_unfreezing(FrozenBert, ["attention.self", "attention.output"])

# Train the model
trainer = Trainer(
    model=FrozenBert,
    args=training_args,
    train_dataset=train_CDA,
    eval_dataset=val_dataset,
    optimizers=(
        AdamW(FrozenBert.parameters(), lr=1e-5, weight_decay=0.1),
        None
        )
)

trainer.train()
results = trainer.evaluate()
print(results)

del trainer, FrozenBert
gc.collect()
torch.cuda.empty_cache()


#                   Regularizers
#---------------------------------------------------------


# The idea of regularizers is to modify the original task loss by adding a new term. In particular, we have implemented Entropy Attention Regularizer (EAR) and a projection based regularizer.
# Here we showcase the EAR regularizer through the `EARModel` class:

# Imports
from FairLangProc.algorithms.inprocessors import EARModel

# Intiliaze the model
model = get_bert()
EARRegularizer = EARModel(
     model = model,
     ear_reg_strength = 0.01
)
trainer = Trainer(
    model=EARRegularizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(
        AdamW(EARRegularizer.parameters(), lr=1e-5, weight_decay=0.1),
        None
        )
)

# Train it
trainer.train()
results = trainer.evaluate()
print(results)

del trainer, EARRegularizer, model
gc.collect()
torch.cuda.empty_cache()

#---------------------------------------------------------
#                   Intra-processors
#---------------------------------------------------------


# Intra processors are those methods that happen after training has already been done but which do not change the model's parameters. There is certain overlap between intra processors and more traditional post processors.
# We have implemented:
# 1. Diff pruning.
# 1. Entropy Attention Temperature (EAT) scaling.

#                   Diff-prunning
#---------------------------------------------------------

# Diff pruning is a modular technique which freezes the model's parameters and trains a sparse set of parameters added over the original ones.
# These parameters are decomposed as a product of their magnitude and a sparsity mask. This parametrization requires the implementation of a new loss with the following components:
#   1. Loss of the original task.
#   2. Debias loss.
#   3. Sparsity loss.
# 
# The total loss function is given by the sum of the previous three terms.
# We have implemented this method through the `DiffPrunedDebiasing` class which requires the implementation of the abstract method `_get_embedding`, which computes the embedding of a given input.
# We provide the implementation of `DiffPrunningBERT` to apply this method to the BERT model.

# Imports
from FairLangProc.algorithms.intraprocessors import DiffPrunBERT

# Counterfactual pairs for the debias loss
gendered_pairs = [
    ("manager", "manageress"),
    ("nephew", "niece"),
    ("prince", "princess"),
    ("baron", "baroness"),
    ("father", "mother"),
    ("stepsons", "stepdaughters"),
    ("boyfriend", "girlfriend"),
    ("fiances", "fiancees"),
    ("shepherd", "shepherdess"),
    ("beau", "belle"),
    ("males", "females"),
    ("hunter", "huntress"),
    ("grandfathers", "grandmothers"),
    ("daddies", "mummies"),
    ("step-son", "step-daughter"),
    ("masters", "mistresses"),
    ("nephews", "nieces"),
    ("brother", "sister"),
    ("grandfather", "grandmother"),
    ("priest", "priestess")
]
tokens_male = [words[0] for words in gendered_pairs]
tokens_female = [words[1] for words in gendered_pairs]
inputs_male = TOKENIZER(
    tokens_male, padding = True, return_tensors = "pt"
    )
inputs_female = TOKENIZER(
    tokens_female, padding = True, return_tensors = "pt"
    )

# kernel for the debias loss
def normalize_by_column(x: torch.Tensor, eps: float = 1e-8):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    return (x - mean) / (std + eps)

# Initialize the model
original_model = get_bert()
ModularDebiasingBERT = DiffPrunBERT(
    head = original_model.classifier,
    encoder = original_model.bert,
    loss_fn = torch.nn.CrossEntropyLoss(),
    input_ids_A = inputs_male,
    input_ids_B = inputs_female,
    bias_kernel = normalize_by_column,
    upper = 10,
    lower = -0.001,
    lambda_bias = 0.5,
    lambda_sparse = 0.00001
)

# Train the model
trainer = Trainer(
    model=ModularDebiasingBERT,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(
        AdamW(ModularDebiasingBERT.parameters(), lr=1e-5, weight_decay=0.1),
        None
        )
)

trainer.train()
results = trainer.evaluate()
print(results)

del trainer, ModularDebiasingBERT, original_model
gc.collect()
torch.cuda.empty_cache()


#                   Entropy Attention Temperature scaling
#----------------------------------------------------------------------


# Entropy Attention Temperature (EAT) scaling proposes the use of Entropy-based Attention Temperature (EAT) scaling in order to modify the distribution of the attention scores with a temperature-related parameter.
# We have implemented EAT scaling through the `add_EAT_hook` which simply requires the specification of a LLM and the temperature parameter.

# Import
from FairLangProc.algorithms.intraprocessors import add_EAT_hook

# Initialize parameters
EATBert = get_bert()
beta = 1.5

# Add EAT scaling
add_EAT_hook(model=EATBert, beta=beta)

# We put the model in the trainer but we do not need to re-train it: we just compute the results
trainer = Trainer(
    model=EATBert,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(
        AdamW(EATBert.parameters(), lr=1e-5, weight_decay=0.1),
        None
        )
)
results = trainer.evaluate()
print(results)

del trainer, EATBert
gc.collect()
torch.cuda.empty_cache()


#=========================================================
#                   DEBIASING BERT
#=========================================================

# This section is provides an example of how to debias the BERT model on a setting closer to reality, applying the reviewed methods in the GLUE dataset.

#                   Imports
#---------------------------------------------------------


# Standard imports
import json

# Computing libraries
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# huggging face
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from adapters import AdapterTrainer
from datasets import load_dataset, Dataset, DatasetDict
import evaluate

# Custom imports
from FairLangProc.datasets import BiasDataLoader
from FairLangProc.metrics import WEAT

from FairLangProc.algorithms.preprocessors import CDA, BLINDTrainer, SentDebiasForSequenceClassification
from FairLangProc.algorithms.inprocessors import EARModel, DebiasAdapter, selective_unfreezing 
from FairLangProc.algorithms.intraprocessors import add_EAT_hook, DiffPrunBERT, DiffPrunDebiasing


#                   Configuration
#---------------------------------------------------------

MODELS = [
    'bert-base-uncased',
    'deepseek-ai/deepseek-llm-7b-base',
    'huggyllama/llama-7b'
]
TASKS = [
    "cola",
    "sst2",
    "mrpc",
    "stsb",
    "qqp",
    "mnli",
    "qnli",
    "rte",
    "wnli"
]
TASK_LABELS = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "qqp": 2,
    "stsb": 1,
    "mnli": 3,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}
DEBIAS_METHODS = [
    "none",
    "cda",
    "blind",
    "embedding",
    "ear",
    "adele",
    "selective",
    "eat",
    "diff"
]
TASK_METRICS = {
    "cola": "eval_matthews_correlation",
    "sst2": "eval_accuracy",
    "mrpc": "eval_accuracy",
    "stsb": "eval_pearson",
    "mnli": "eval_accuracy",
    "qnli": "eval_accuracy",
    "rte": "eval_accuracy",
    "wnli": "eval_accuracy",
}
CDA_METHOD = {
    "none": False,
    "cda": True,
    "blind": False,
    "embedding": False,
    "ear": False,
    "adele": True,
    "selective": True,
    "eat": False,
    "diff": False
}

BATCH_SIZE = 8
WEIGHT_DECAY = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

counterfactual_pairs = [
        ("gods", "goddesses"), ("manager", "manageress"), ("barons", "baronesses"),
        ("nephew", "niece"), ("prince", "princess"), ("boars", "sows"),
        ("baron", "baroness"), ("stepfathers", "stepmothers"), ("wizard", "witch"),
        ("father", "mother"), ("stepsons", "stepdaughters"), ("sons-in-law", "daughters-in-law"),
        ("dukes", "duchesses"), ("boyfriend", "girlfriend"), ("fiances", "fiancees"),
        ("dad", "mom"), ("shepherd", "shepherdess"), ("uncles", "aunts"),
        ("beau", "belle"), ("males", "females"), ("hunter", "huntress"),
        ("beaus", "belles"), ("grandfathers", "grandmothers"), ("lads", "lasses"),
        ("daddies", "mummies"), ("step-son", "step-daughter"), ("masters", "mistresses"),
        ("policeman", "policewoman"), ("nephews", "nieces"), ("brother", "sister"),
        ("grandfather", "grandmother"), ("priest", "priestess"), ("hosts", "hostesses"),
        ("landlord", "landlady"), ("husband", "wife"), ("poet", "poetess"),
        ("landlords", "landladies"), ("fathers", "mothers"), ("masseur", "masseuse"),
        ("monks", "nuns"), ("usher", "usherette"), ("hero", "heroine"),
        ("stepson", "stepdaughter"), ("postman", "postwoman"), ("god", "goddess"),
        ("milkmen", "milkmaids"), ("stags", "hinds"), ("grandpa", "grandma"),
        ("chairmen", "chairwomen"), ("husbands", "wives"), ("grandpas", "grandmas"),
        ("stewards", "stewardesses"), ("murderer", "murderess"), ("manservant", "maidservant"),
        ("men", "women"), ("host", "hostess"), ("heirs", "heiresses"),
        ("masseurs", "masseuses"), ("boy", "girl"), ("male", "female"),
        ("son-in-law", "daughter-in-law"), ("waiter", "waitress"), ("tutors", "governesses"),
        ("priests", "priestesses"), ("bachelor", "spinster"), ("millionaire", "millionairess"),
        ("steward", "stewardess"), ("businessmen", "businesswomen"), ("congressman", "congresswoman"),
        ("emperor", "empress"), ("duke", "duchess"), ("sire", "dam"),
        ("son", "daughter"), ("sirs", "madams"), ("widower", "widow"),
        ("kings", "queens"), ("papas", "mamas"), ("grandsons", "granddaughters"),
        ("proprietor", "proprietress"), ("monk", "nun"), ("headmasters", "headmistresses"),
        ("grooms", "brides"), ("heir", "heiress"), ("boys", "girls"),
        ("gentleman", "lady"), ("uncle", "aunt"), ("he", "she"),
        ("king", "queen"), ("princes", "princesses"), ("policemen", "policewomen"),
        ("governor", "matron"), ("fiance", "fiancee"), ("step-father", "step-mother"),
        ("waiters", "waitresses"), ("mr", "mrs"), ("stepfather", "stepmother"),
        ("daddy", "mummy"), ("lords", "ladies"), ("widowers", "widows"),
        ("emperors", "empresses"), ("father-in-law", "mother-in-law"), ("abbot", "abbess"),
        ("sir", "madam"), ("actor", "actress"), ("mr.", "mrs."),
        ("wizards", "witches"), ("actors", "actresses"), ("chairman", "chairwoman"),
        ("sorcerer", "sorceress"), ("postmaster", "postmistress"), ("brothers", "sisters"),
        ("lad", "lass"), ("headmaster", "headmistress"), ("papa", "mama"),
        ("milkman", "milkmaid"), ("heroes", "heroines"), ("man", "woman"),
        ("grandson", "granddaughter"), ("groom", "bride"), ("sons", "daughters"),
        ("congressmen", "congresswomen"), ("businessman", "businesswoman"), ("boyfriends", "girlfriends"),
        ("dads", "moms"),
    ]

# WEAT classes for bias evaluation
class BertWEAT(WEAT):
    def _get_embedding(self, outputs):
        return outputs.last_hidden_state[:, 0, :]

class AverageAutoregWEAT(WEAT):
    def _get_embedding(self, outputs):
        return outputs.last_hidden_state.mean(dim = 1)

#                   Main loop
#---------------------------------------------------------

# This loop will train BERT on all GLUE tasks and apply all debiasing methods
MODEL_NAME = 'bert-base-uncased'
for (TASK, DEBIAS) in zip(TASKS, DEBIAS_METHODS):

    # Get task metric
    METRIC_FOR_BEST = TASK_METRICS.get(TASK, "eval_accuracy")
    # Get task label and problem type
    num_labels = TASK_LABELS[TASK]
    if TASK == 'stsb':
        problem_type='regression'
    else:
        problem_type='single_label_classification'

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Initialize the model. If the debias method is ADELE, EAT or Diff prunning we must start from a checkpoint.
    # All other debiasing methods use a clean initialization.
    if DEBIAS in ("adele", "eat", "diff"):
        try:
            RESULTS_PATH = f'../output/{TASK}-none-{MODEL_NAME}/'
            CHECKPOINTS = [direction for direction in os.listdir(RESULTS_PATH) if direction.startswith('checkpoint')]
            LAST_CHECKPOINT_PATH = RESULTS_PATH + CHECKPOINTS[-1]
            original_model = AutoModelForSequenceClassification.from_pretrained(LAST_CHECKPOINT_PATH)
        except:
            try:
                RESULTS_PATH = f'output/{TASK}-none-{MODEL_NAME}/'
                CHECKPOINTS = [direction for direction in os.listdir(RESULTS_PATH) if direction.startswith('checkpoint')]
                LAST_CHECKPOINT_PATH = RESULTS_PATH + CHECKPOINTS[-1]
                original_model = AutoModelForSequenceClassification.from_pretrained(LAST_CHECKPOINT_PATH)
            except:
                original_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, problem_type=problem_type)
    else:
        original_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, problem_type=problem_type)

    # Store hidden dimension and the model classifier
    hidden_dim = original_model.config.hidden_size
    if not hasattr(original_model, 'classifier'):
        original_model.classifier = original_model.score

    #                   Model selection
    #---------------------------------------------------------
    
    # We will choose the correct model and store it on the `model` variable depending on the chosen `DEBIAS`

    # If no debiasing is used or if we are testing CDA or EAT we do not need to modify BERT
    if DEBIAS in ("none", "cda", "eat"):
        model = original_model

    # Embedding-based debiasing requires implementation of `_get_embedding`
    if DEBIAS == "embedding":

        class SentDebiasBert(SentDebiasForSequenceClassification):        
            def _get_embedding(self, input_ids, attention_mask = None, token_type_ids = None):
                return self.model.bert(
                    input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids
                    ).last_hidden_state[:,0,:]

        class SentDebiasAverageAutoreg(SentDebiasForSequenceClassification):
            def _get_embedding(self, input_ids, attention_mask = None, token_type_ids = None):
                return self.model.model(
                    input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids
                    ).last_hidden_state.mean(dim = 1)

        if MODEL_NAME == 'bert-base-uncased':
            model = SentDebiasBert(
                model = original_model,
                config = None,
                tokenizer = tokenizer,
                word_pairs = counterfactual_pairs,
                n_components = 1,
                n_labels = num_labels
            )
        else:
            model = SentDebiasAverageAutoreg(
                model = original_model,
                config = None,
                tokenizer = tokenizer,
                word_pairs = counterfactual_pairs,
                n_components = 1,
                n_labels = num_labels
            )

    # BLIND debiasing requires implementation of `_get_embedding`
    if DEBIAS == "blind":
        model = original_model
        class BLINDBERTTrainer(BLINDTrainer):
            def _get_embedding(self, inputs):
                return self.model.bert(
                    input_ids = inputs.get("input_ids"), attention_mask = inputs.get("attention_mask"), token_type_ids = inputs.get("token_type_ids")
                    ).last_hidden_state[:,0,:]

    # ADELE requires applying the bottleneck adapter
    if DEBIAS == "adele":
        DebiasAdapterInst = DebiasAdapter(model = original_model)
        model = DebiasAdapterInst.get_model()


    # EAR regularization initalization is straightforward
    if DEBIAS == "ear":
        model = EARModel(
            model = original_model,
            ear_reg_strength = 0.01
        )

    # Selective unfreezing simply freezes all model parameters
    if DEBIAS == "selective":
        model = original_model
        selective_unfreezing(model, ["attention.self", "attention.output"])


    # We implement the normalization kernel for Diff prunning and tokenize the counterfactual pairs
    if DEBIAS == "diff":
        def normalize_by_column(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)
            return (x - mean) / (std + eps)

        if TASK == "stsb":
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

        class DiffPrunAvgAutoReg(DiffPrunDebiasing):
            def _forward(self, input_ids, attention_mask=None, token_type_ids=None):
                outputs = self.encoder(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    token_type_ids = token_type_ids
                    )
                return outputs.last_hidden_state.mean(dim = 1)

        tokens_male = [words[0] for words in counterfactual_pairs]
        tokens_female = [words[1] for words in counterfactual_pairs]

        inputs_male = tokenizer(tokens_male, padding = True, return_tensors = "pt")
        inputs_female = tokenizer(tokens_female, padding = True, return_tensors = "pt")

        if MODEL_NAME == 'bert-base-uncased':
            model = DiffPrunBERT(
                head = original_model.classifier,
                encoder = original_model.bert,
                loss_fn = loss_fn,
                input_ids_A = inputs_male,
                input_ids_B = inputs_female,
                bias_kernel = normalize_by_column,
                upper = 10,
                lower = -0.001,
                lambda_bias = 0.5,
                lambda_sparse = 0.00001
            )

        else:
            model = DiffPrunAvgAutoReg(
                head = original_model.classifier,
                encoder = original_model.base_model,
                loss_fn = loss_fn,
                input_ids_A = inputs_male[:50],
                input_ids_B = inputs_female[:50],
                bias_kernel = normalize_by_column,
                upper = 10,
                lower = -0.001,
                lambda_bias = 0.5,
                lambda_sparse = 0.00001
            )


    #                   Data processing
    #---------------------------------------------------------

    # Tokenizing the datasets
    def preprocess_function(examples, tokenizer, task):
        if task in ["sst2", "cola"]:
            return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)
        elif task == "mnli":
            return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length", max_length=128)
        elif task == "qnli":
            return tokenizer(examples["question"], examples["sentence"], truncation=True, padding="max_length", max_length=128)
        elif task in ["rte", "wnli"]:
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=128)
        elif task == "mrpc":
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=128)
        elif task == "qqp":
            return tokenizer(examples["question1"], examples["question2"], truncation=True, padding="max_length", max_length=128)
        elif task == "stsb":
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=128)

    # Obtain metrics for all tasks
    def get_metrics(task_name):
        metric = evaluate.load("glue", task_name)
        if task_name == "stsb":
            return metric, lambda logits: np.squeeze(logits, axis=-1)
        return metric, lambda logits: np.argmax(logits, axis=-1)

    # Compute metrics from the batch p
    def compute_metrics_fn(p, task_name):
        logits = p.predictions
        labels = p.label_ids

        if isinstance(logits, tuple) or isinstance(logits, list):
            logits = logits[0]

        metric, postprocess_fn = get_metrics(task_name)
        predictions = postprocess_fn(logits)

        return metric.compute(predictions=predictions, references=labels)


    #                   Load dataset
    #---------------------------------------------------------

    # Dataset
    dataset = load_dataset("glue", TASK)
    
    for split in dataset.keys():
        dataset[split] = dataset[split].select(range(min(100, len(dataset[split]))))

    # Perform CDA if the model requires it
    if CDA_METHOD[DEBIAS] and TASK != 'mnli':
        train_dataset = Dataset.from_dict(
            CDA(dataset['train'][:], pairs = dict(counterfactual_pairs))
            )
        dataset = DatasetDict({
            "train": train_dataset,
            "validation": dataset["validation"],
            "test": dataset["test"]
        })
    elif CDA_METHOD[DEBIAS] and TASK == 'mnli':
        train_dataset = Dataset.from_dict(
            CDA(dataset['train'][:], pairs = dict(counterfactual_pairs))
            )
        dataset = DatasetDict({
            "train": train_dataset,
            "validation_matched": dataset["validation_matched"],
            "validation_mismatched": dataset["validation_mismatched"],
            "test_matched": dataset["test_matched"],
            "test_mismatched": dataset["test_mismatched"]
        })

    # mnli has two validation sets, we use the matched version
    if TASK == 'mnli':
        dataset["validation"] = dataset["validation_matched"]

    # preprocess and pad
    tokenized_datasets = dataset.map(lambda x: preprocess_function(x, tokenizer, TASK), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    #                   Fine-tuning
    #---------------------------------------------------------

    # Configuration
    EVAL_STRATEGY = "epoch"
    SAVE_STRATEGY = "epoch"
    LOAD_BEST_MODEL_AT_END = True
    SAVE_SAFETENSORS = True
    FP16 = True

    # Set up callbacks
    # Diff prunning is particularly computation-heavy
    if DEBIAS == 'diff':
        PATIENCE = 5
        SAVE_SAFETENSORS = False
    else:
        PATIENCE = 2
 
    CALLBACKS = [EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
    EVAL_STEPS = None

    # eat does not require saving the model because it does not modify its parameters
    if DEBIAS == 'eat':
        SAVE_STRATEGY = "no"
        LOAD_BEST_MODEL_AT_END = False
        CALLBACKS = None 

    # BLIND and ADELE use special trainers
    if DEBIAS == 'adele':
        trainer_cls = AdapterTrainer
    elif DEBIAS == 'blind':
        trainer_cls = BLINDBERTTrainer
    else:
        trainer_cls = Trainer

    # The sheer size of QQP and MNLI makes this approach preferable
    if TASK in ('qqp', 'mnli'):
        BATCH_SIZE = 8
        EVAL_STRATEGY = "steps"
        EVAL_STEPS = 50
        SAVE_STEPS = 50
    else:
        BATCH_SIZE = 8
        EVAL_STRATEGY = "epoch"
        EVAL_STEPS = None
        SAVE_STEPS = None

    if LOAD_BEST_MODEL_AT_END:
        SAVE_STRATEGY = EVAL_STRATEGY 


    # Load the training args
    training_args = TrainingArguments(
        output_dir=f"output/{TASK}-{DEBIAS}-{MODEL_NAME.replace('/', '-')}",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=1,
        eval_strategy=EVAL_STRATEGY,
        eval_steps=EVAL_STEPS,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        save_safetensors=SAVE_SAFETENSORS, 
        logging_dir="logs",
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=METRIC_FOR_BEST,
        fp16=FP16,
        greater_is_better = True
    )

    # We need to insert the BLIND optimizer if we are dealing with BLIND
    if DEBIAS == 'blind':
        trainer = trainer_cls(
            blind_optimizer= lambda x: AdamW(x, lr=1e-5, weight_decay=WEIGHT_DECAY),
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_metrics_fn(p, TASK),
            callbacks=CALLBACKS,
            optimizers=(AdamW(model.parameters(), lr=1e-5, weight_decay=WEIGHT_DECAY), None)
        )
    else:
        trainer = trainer_cls(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_metrics_fn(p, TASK),
            callbacks=CALLBACKS,
            optimizers=(AdamW(model.parameters(), lr=1e-5, weight_decay=WEIGHT_DECAY), None)
        )

    # EAT scaling does not require training the model, but we need to add the EAT hook.
    # In any other case, train the model.
    if DEBIAS == 'eat':
        add_EAT_hook(model, beta=0.7)
    else:
        trainer.train()

    # Print results
    if TASK == 'mnli':
        eval_results_mismd = trainer.evaluate(tokenized_datasets["validation_mismatched"])
        eval_results_match = trainer.evaluate(tokenized_datasets["validation_matched"])
        print("Validation results (matched) in ", TASK, ":", eval_results_match)
        print("Validation results (mismatched) in ", TASK, ":", eval_results_mismd)
    else:
        eval_results = trainer.evaluate()
        print("Validation results in ", TASK, ":", eval_results)


    #                   Bias evaluation
    #---------------------------------------------------------

    if MODEL_NAME == 'bert-base-uncased':
        if DEBIAS == 'diff':
            weat = BertWEAT(model = model.encoder, tokenizer = tokenizer)
        else:
            try:
                weat = BertWEAT(model = model.model.bert, tokenizer = tokenizer)
            except:
                try:
                    weat = BertWEAT(model = model.bert, tokenizer = tokenizer)
                except:
                    weat = BertWEAT(model = model.base_model.bert, tokenizer = tokenizer)
    else:
        if DEBIAS == 'diff':
            weat = AverageAutoregWEAT(model = model.encoder, tokenizer = tokenizer)
        else:
            try:
                weat = AverageAutoregWEAT(model = model.model.base_model, tokenizer = tokenizer)
            except:
                try:
                    weat = AverageAutoregWEAT(model = model.base_model, tokenizer = tokenizer)
                except:
                    weat = AverageAutoregWEAT(model = model.base_model.base_model, tokenizer = tokenizer)

    # WEAT 7 test
    math = ['math', 'algebra', 'geometry', 'calculus', 'equations', 'computation', 'numbers', 'addition']
    arts = ['poetry', 'art', 'dance', 'literature', 'novel', 'symphony', 'drama', 'sculpture']
    male = ['male', 'man', 'boy', 'brother', 'he', 'him', 'his', 'son']
    female = ['female', 'woman', 'girl', 'sister', 'she', 'her', 'hers', 'daughter']

    # Run the test, print the results
    bias_results = weat.metric(
        W1_words = math, W2_words = arts,
        A1_words = male, A2_words = female,
        pval = False
        )
    print(bias_results)

    # Save results
    os.makedirs(f"output/{TASK}-{DEBIAS}-{MODEL_NAME.replace('/', '-')}", exist_ok=True)
    if TASK == 'mnli':
        with open(f"output/{TASK}-{DEBIAS}-{MODEL_NAME.replace('/', '-')}/results.json", "w") as f:
            json.dump({"eval_matched": eval_results_match, "eval_mismatched": eval_results_mismd, "bias": bias_results}, f, indent=4)
    else:
        with open(f"output/{TASK}-{DEBIAS}-{MODEL_NAME.replace('/', '-')}/results.json", "w") as f:
            json.dump({"eval": eval_results, "bias": bias_results}, f, indent=4)

    del original_model, model, trainer, weat
    gc.collect()
    torch.cuda.empty_cache()

#                   Print the results
#---------------------------------------------------------

# This last step simply reads the obtained results and prints them as a list and as a lattex table. This code was used for generating the tables of the article.

TITLE_DICT = {
    'mnli': 'MNLI',
    'qqp': 'QQP',
    'qnli': 'QNLI',
    'sst2': 'SST-2',
    'cola': 'CoLA',
    'stsb': 'STS-B',
    'mrpc': 'MRPC',
    'rte': 'RTE',
    'wnli': 'WNLI',
    'Average': 'Average'
}

EVAL_OR_TEST = 'eval'

# Left column
LEN_DICT = {debias: len(TASKS) for debias in DEBIAS_METHODS}
LEN_DICT['blind'] = len(TASKS)-1

# Table entries
latex_eval = {task: {} for task in TASKS}
latex_bias = {task: {} for task in TASKS}

# Average GLUE score
average = True
if average:
    latex_eval['Average'] = {debias: 0.0 for debias in DEBIAS_METHODS}
    latex_bias['Average'] = {debias: 0.0 for debias in DEBIAS_METHODS}

# For each task and debias method store them in the table variables and print them as they are read
for task in TASKS:
    for debias in DEBIAS_METHODS:
        path = f"output/{task}-{debias}-{MODEL_NAME.replace('/', '-')}/results.json"
        try:
            with open(path, "r") as f:
                resultsDict = json.load(f)
                
                print('='*50)
                print(task, debias)
                print('-'*50)
                if task != 'mnli':
                    print('eval: ', resultsDict['eval'][TASK_METRICS[task]]*100)
                else:
                    for sufix in ['_matched', '_mismatched']:
                        print('eval' + sufix + ': ', resultsDict['eval' + sufix][TASK_METRICS[task]]*100)
                print('bias: ', resultsDict['bias']['effect_size'])

                if task != 'mnli':
                    latex_eval[task][debias] = resultsDict[EVAL_OR_TEST][TASK_METRICS[task]]*100
                    latex_eval['Average'][debias] += latex_eval[task][debias] / (LEN_DICT[debias]+1)
                elif task == 'mnli':
                    latex_eval[task][debias] = [resultsDict[EVAL_OR_TEST + sufix][TASK_METRICS[task]]*100 for sufix in ['_matched', '_mismatched']]
                    latex_eval['Average'][debias] += sum(latex_eval[task][debias]) / (LEN_DICT[debias]+1)
                    
                    
                latex_bias[task][debias] = resultsDict['bias']['effect_size']
                latex_bias['Average'][debias] += latex_bias[task][debias] / (LEN_DICT[debias])

        except:
            pass


# Parameter that adds a dashed line between BERT and the debiasing methods
dashed = True

# Left column of the table
debias_names = {debias: debias for debias in DEBIAS_METHODS}
debias_names['selective'] = 'sel'
debias_names['embedding'] = 'emb'

# Header of both tablse
header_cs = 'c|'*(len(latex_eval.keys())) + 'c'
header = '\\begin{table}[h] \n \t \\small \n \t \\centering \n \t \\begin{tabular}{' + header_cs + '}\n'
table_eval = header + '\t\t \\hline Debias & '
table_bias = header + '\t\t \\hline Debias & '

# For each table, add the corresponding
for (i, task) in enumerate(latex_eval.keys()):
    # make sure to make a new line at the end of the table
    if i == len(latex_eval.keys())-1:
        table_eval += f'{TITLE_DICT[task]} \\\\ \\hline '
        table_bias += f'{TITLE_DICT[task]} \\\\ \\hline '
    else:
        table_eval += f'{TITLE_DICT[task]} & '
        table_bias += f'{TITLE_DICT[task]} & '

for debias in DEBIAS_METHODS:
    # CDA is the first debias method, add a dashed line above it
    if dashed and debias == 'cda':
        table_eval += f'\n \t\t \\hdashline {debias} & '
        table_bias += f'\n \t\t \\hdashline {debias} & '
    else:
        table_eval += f'\n \t\t {debias_names[debias]} & '
        table_bias += f'\n \t\t {debias_names[debias]} & '
    
    # Store the results in the table strings
    for (i, task) in enumerate(latex_eval.keys()):
        # In the last line make sure to make a new line
        if i == len(latex_eval.keys())-1:
            try:
                table_eval += f'{latex_eval[task][debias]:.1f} \\\\ '
                table_bias += f'{latex_bias[task][debias]:.3f} \\\\ '
            # Missing entries are filled with a "-"
            except:
                table_eval += r'- \\ '
                table_bias += r'- \\ '

        else:
            entry_eval = f'- & '
            entry_bias = f'- & '
            try:
                if task != 'mnli':
                    entry_eval = f'{latex_eval[task][debias]:.1f} & '
                    entry_bias = f'{latex_bias[task][debias]:.3f} & '
                else:
                    entry_eval = f'{latex_eval[task][debias][0]:.1f}/{latex_eval[task][debias][1]:.1f} & '
                    entry_bias = f'{latex_bias[task][debias]:.3f} & '
            except:
                pass
            finally:
                table_eval += entry_eval
                table_bias += entry_bias

# Add captions
table_eval += '\\hline \n \t \\end{tabular} \n \t \\caption{Performance of the different model on GLUE tasks for the validation set. F1 scores are reported for QQP and MRPC, Spearman correlations are reported for STS-B, and accuracy scores are reported for the other tasks.} \n \t \\label{tab:performance} \n \\end{table}'
table_bias += '\\hline \n \t \\end{tabular} \n \t \\caption{WEAT 7 test for the debiasing methods.} \n \t \\label{tab:bias} \n \\end{table}'

# Print tables
print('TABLES \n', '='*50)
print(table_eval)
print(table_bias)
print('='*50, '\n'*3)