import torch
import pytest
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments

from FairLangProc.algorithms.preprocessors import CDA, BLINDTrainer, SentDebiasForSequenceClassification
from FairLangProc.algorithms.inprocessors import EARModel, DebiasAdapter, selective_unfreezing 
from FairLangProc.algorithms.intraprocessors import add_EAT_hook, DiffPrunBERT

#==================================
#    TEST VARIABLES
#==================================

SENTENCES = [
    'he is a good father',
    'the actor gave a staggering performance',
    'she tries very hard'
]
LABELS = [0, 1, 0]
BATCH = {'sentence': SENTENCES, 'label': LABELS}
PAIRS = {'he': 'she', 'actor': 'actress', 'father': 'mother'}

LAYERS = ["attention.self", "attention.output"]

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2

TEST_CASES_OUTPUT = [
    "emb",
    "ear",
    "selective",
    "adele",
    "diff",
    "eat"
]

PAIRS = [('actor', 'actress'), ('son', 'daughter'), ('father', 'mother'), ('he', 'she')]




#==================================
#    TEST CLASSES
#==================================

class BLINDBERTTrainer(BLINDTrainer):
    def _get_embedding(self, inputs):
        return self.model.bert(
            input_ids = inputs.get("input_ids"), attention_mask = inputs.get("attention_mask"), token_type_ids = inputs.get("token_type_ids")
            ).last_hidden_state[:,0,:]
    
class SentDebiasBert(SentDebiasForSequenceClassification):        
    def _get_embedding(self, input_ids, attention_mask = None, token_type_ids = None):
        return self.model.bert(
            input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids
            ).last_hidden_state[:,0,:]

def get_model(base_model, tokenizer, debias):
    
    if debias == "emb":
        model = SentDebiasBert(
            model = base_model,
            config = None,
            tokenizer = tokenizer,
            word_pairs = PAIRS,
            n_components = 1,
            n_labels = NUM_LABELS
        )
    
    elif debias == "ear":
        model = EARModel(
            model = base_model,
            ear_reg_strength = 0.01
        )

    elif debias == "selective":
        model = base_model
        selective_unfreezing(model, LAYERS)

    elif debias == "adele":
        DebiasAdapter = DebiasAdapter(model = base_model)
        model = DebiasAdapter.get_model()

    elif debias == "diff":
        tokens_male = [words[0] for words in PAIRS]
        tokens_female = [words[1] for words in PAIRS]
        inputs_male = tokenizer(tokens_male, padding = True, return_tensors = "pt")
        inputs_female = tokenizer(tokens_female, padding = True, return_tensors = "pt")
        model = DiffPrunBERT(
            head = base_model.classifier,
            encoder = base_model.bert,
            loss_fn = torch.nn.CrossEntropyLoss(),
            input_ids_A = inputs_male,
            input_ids_B = inputs_female,
            bias_kernel = None,
            upper = 10,
            lower = -0.001,
            lambda_bias = 0.5,
            lambda_sparse = 0.00001
        )

    elif debias == "eat":
        model = base_model
        add_EAT_hook(model, beta=0.7)
    
    return model

#==================================
#    TEST MODELS
#==================================

def test_cda_bidirectional():
    result = CDA(batch = BATCH, pairs = PAIRS, bidirectional = True)
    assert isinstance(result, dict), f"Wrong type: expected {dict}, got {type(result)}"
    assert len(result['sentence']) == 5, f"Expected 5 sentences, got {len(result['sentence'])}"
    assert len(result['label']) == 5, f"Expected 5 labels, got {len(result['label'])}"

def test_cda_no_bidirectional():
    result = CDA(batch = BATCH, pairs = PAIRS, bidirectional = False)
    assert isinstance(result, dict), f"Wrong type: expected {dict}, got {type(result)}"
    assert len(result['sentence']) == 3, f"Expected 3 sentences, got {len(result['sentence'])}"
    assert len(result['label']) == 3, f"Expected 3 labels, got {len(result['label'])}"

def test_blind_model_output_shape():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    sample = tokenizer("A test sentence.", return_tensors="pt")
    sample["labels"] = torch.tensor([1])

    # Dummy blind classifier
    blind_classifier = torch.nn.Sequential(
        torch.nn.Linear(768, 768),
        torch.nn.ReLU(),
        torch.nn.Linear(768, 2)
    )

    # Dummy training args
    training_args = TrainingArguments(
        output_dir="/tmp/test_blind",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        no_cuda=True
    )

    # Instantiate trainer
    trainer = BLINDBERTTrainer(
        model=model,
        blind_model=blind_classifier,
        args=training_args,
        train_dataset=[sample],
        eval_dataset=[sample],
    )

    # Forward pass
    embedding = trainer._get_embedding(sample)
    logits_blind = trainer.blind_model(embedding)

    assert logits_blind.shape == (1, 2), f"Expected blind logits of shape (1, 2), got {logits_blind.shape}"


def test_unfreezing():
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    selective_unfreezing(base_model, LAYERS)
    for name, param in base_model.named_parameters():
        if any(layer_key in name for layer_key in LAYERS):
            assert param.requires_grad, f"Expected param '{name}' to be trainable"
        else:
            assert not param.requires_grad, f"Expected param '{name}' to be frozen"


def test_hook():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    pre_hook_counts = [
        len(layer.attention.self._forward_hooks)
        for layer in model.base_model.encoder.layer
    ]

    add_EAT_hook(model, beta=1.2)
    post_hook_counts = [
        len(layer.attention.self._forward_hooks)
        for layer in model.base_model.encoder.layer
    ]

    for pre, post in zip(pre_hook_counts, post_hook_counts):
        assert post == pre + 1, f"Expected 1 new hook, but got {post - pre}"

@pytest.mark.parametrize("debias", TEST_CASES_OUTPUT)
def test_sequence_classification_output_shape(debias):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model = get_model(base_model, tokenizer, debias)

    inputs = tokenizer("This is a test sentence.", return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    assert outputs.logits.shape == (1, NUM_LABELS), \
        f"Expected logits shape (1, {NUM_LABELS}), but got {outputs.logits.shape}"