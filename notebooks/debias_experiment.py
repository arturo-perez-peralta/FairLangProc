# debias_experiment.py
"""
Contains the complete logic for running a single k-fold debiasing experiment.
Converted from DemoDebiasing.ipynb.
"""

# Standard imports
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.model_selection import KFold
from pathlib import Path

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
from datasets import load_dataset, load_metric, concatenate_datasets

# --- Local Package Imports ---
# (This assumes FairLangProc is importable)
# We will set up the path inside run_experiment if needed

# --- Mappings ---
TASK_TO_KEYS = {
    "cola": ("sentence", None), "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"), "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"), "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None), "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
GLUE_METRICS = {
    "cola": "matthews_correlation", "mnli": "accuracy", "mrpc": "f1",
    "qnli": "accuracy", "qqp": "f1", "rte": "accuracy",
    "sst2": "accuracy", "stsb": "pearson", "wnli": "accuracy",
}

# --- Custom Trainers ---
class DebiasTrainer(Trainer):
    def __init__(self, processor=None, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
    
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if self.processor:
            loss = self.processor.compute_loss(inputs, outputs, model)
        return (loss, outputs) if return_outputs else loss

class DebiasAdapterTrainer(AdapterTrainer):
    def __init__(self, processor=None, **kwargs):
        super().__init__(**kwargs)
        self.processor = processor
    
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        if self.processor:
            loss = self.processor.compute_loss(inputs, outputs, model)
        return (loss, outputs) if return_outputs else loss

# --- Helper Functions ---
def get_embeddings(model, tokenizer):
    """Extracts word embeddings from the model."""
    try:
        embeddings = model.get_input_embeddings().weight.data
    except:
        embeddings = model.bert.get_input_embeddings().weight.data
    return embeddings

def compute_weat(model, tokenizer, device):
    """Computes the WEAT score."""
    from FairLangProc.metrics import weat, effect_size
    
    # ... [Logic from the notebook to load WEAT words] ...
    # This logic should be part of your package, e.g., FairLangProc.metrics.load_weat_words()
    # For this demo, I'll assume they are loaded.
    
    # Placeholder: In a real scenario, you'd load the word sets
    X_words = ['mechanic', 'engineer']
    Y_words = ['nurse', 'teacher']
    A_words = ['he', 'man']
    B_words = ['she', 'woman']

    # Get embeddings for word sets
    embeddings = get_embeddings(model, tokenizer)
    vocab = tokenizer.get_vocab()
    
    X, Y, A, B = [], [], [], []
    for word_set, embed_set in [(X_words, X), (Y_words, Y), (A_words, A), (B_words, B)]:
        for word in word_set:
            if word in vocab:
                embed_set.append(embeddings[vocab[word]].cpu().numpy())
    
    if not all([X, Y, A, B]):
        print("Warning: Could not find all WEAT words in vocab. Skipping bias calc.")
        return 0.0

    X_embed = torch.tensor(np.array(X)).to(device)
    Y_embed = torch.tensor(np.array(Y)).to(device)
    A_embed = torch.tensor(np.array(A)).to(device)
    B_embed = torch.tensor(np.array(B)).to(device)
    
    score, s_X_Y_diff = weat(X_embed, Y_embed, A_embed, B_embed)
    return effect_size(s_X_Y_diff)

# --- Main Experiment Function ---
def run_experiment(
    model_name="bert-base-uncased",
    task_name="stsb",
    debias_method="diff",
    k_folds=5,
    output_path_base="../output",
    local_path_setup=True
):
    """
    Runs a full k-fold debiasing experiment for a given
    model, task, and debias method.
    """
    
    print(f"\n--- Starting Experiment ---")
    print(f"  Model: {model_name}")
    print(f"  Task: {task_name}")
    print(f"  Debias: {debias_method}")
    print(f"  K-Folds: {k_folds}")
    print(f"  Output Base: {output_path_base}")
    print("-" * 27)

    # --- 1. Setup Paths and Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if local_path_setup:
        ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if ROOT_PATH not in sys.path:
            sys.path.insert(0, ROOT_PATH)
            print(f"Added {ROOT_PATH} to sys.path")

    # Import processors *after* setting up path
    from FairLangProc.processors import CDA, Blind, Embedding, EAR, Adele, Selective, EAT, Diff
    PROCESSORS = {
        'cda': CDA, 'blind': Blind, 'embedding': Embedding, 'ear': EAR,
        'adele': Adele, 'selective': Selective, 'eat': EAT, 'diff': Diff,
    }

    # --- 2. Load Datasets ---
    print("Loading datasets...")
    dataset = load_dataset("glue", task_name)
    # crows_ds = load_dataset("crows_pairs", "default")
    # stereo_ds = load_dataset("stereoset", "intrasence")
    
    sentence1_key, sentence2_key = TASK_TO_KEYS[task_name]
    metric_name = GLUE_METRICS[task_name]
    num_labels = 3 if task_name == "mnli" else (1 if task_name == "stsb" else 2)

    # --- 3. Prepare for K-Fold ---
    data_collator = DataCollatorWithPadding(tokenizer=AutoTokenizer.from_pretrained(model_name))
    dataset_train_val = concatenate_datasets([dataset['train'], dataset['validation']])
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    metrics_per_fold = []
    bias_per_fold = []

    print(f"Starting K-Fold Cross-Validation (k={k_folds})...")
    
    for fold, (train_index, val_index) in enumerate(kf.split(dataset_train_val)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        
        # --- 4. Load Model and Tokenizer (fresh for each fold) ---
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

        # --- 5. Initialize Processor ---
        processor = None
        TrainerClass = Trainer
        if debias_method != 'none':
            if debias_method in ['diff', 'adele', 'selective']:
                model.add_adapter("debias_adapter")
                model.train_adapter("debias_adapter")
                TrainerClass = DebiasAdapterTrainer
            elif debias_method not in ['cda', 'blind']:
                TrainerClass = DebiasTrainer
            
            processor = PROCESSORS[debias_method](
                tokenizer=tokenizer, model=model, device=device, task=task_name
            )

        # --- 6. Preprocess Data for this Fold ---
        train_dataset_fold = dataset_train_val.select(train_index)
        val_dataset_fold = dataset_train_val.select(val_index)
        
        if debias_method in ['cda', 'blind']:
            train_dataset_fold = train_dataset_fold.map(processor, batched=True, batch_size=100)

        def preprocess_function(examples):
            args = (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            return tokenizer(*args, padding=False, max_length=128, truncation=True)
        
        tokenized_train = train_dataset_fold.map(preprocess_function, batched=True)
        tokenized_val = val_dataset_fold.map(preprocess_function, batched=True)

        # --- 7. Define Metrics Computation ---
        def compute_metrics(eval_pred):
            metric = load_metric("glue", metric_name)
            preds = eval_pred.predictions
            if metric_name == "stsb":
                preds = preds[:, 0]
            else:
                preds = np.argmax(preds, axis=1)
            labels = eval_pred.label_ids
            return metric.compute(predictions=preds, references=labels)

        # --- 8. Train ---
        model_name_slug = model_name.replace("/", "-")
        output_dir = f"{output_path_base}/fold_checkpoints/{task_name}-{debias_method}-{model_name_slug}/fold_{fold+1}"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
            learning_rate=2e-5,
            save_total_limit=1,
        )

        trainer = TrainerClass(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            processor=processor if debias_method not in ['cda', 'blind'] else None
        )

        print("Starting training for fold...")
        trainer.train()

        # --- 9. Evaluate ---
        print("Evaluating fold...")
        eval_metrics = trainer.evaluate()
        print(f"Fold {fold+1} metrics: {eval_metrics}")
        metrics_per_fold.append(eval_metrics)

        # --- 10. Compute Bias ---
        print("Computing bias for fold...")
        bias_score = compute_weat(model, tokenizer, device)
        print(f"Fold {fold+1} bias (WEAT effect size): {bias_score}")
        bias_per_fold.append(bias_score)

    # --- 11. Aggregate and Save Results ---
    print("\n--- K-Fold Validation Complete ---")
    
    # Aggregate metrics
    mean_eval = 0
    if task_name != 'mnli':
        metric_key = f"eval_{metric_name}"
        all_metrics = [m[metric_key] for m in metrics_per_fold]
        mean_eval = np.mean(all_metrics)
        std_eval = np.std(all_metrics)
        print(f"Overall {metric_key} mean/std: {mean_eval:.4f}/{std_eval:.4f}")
    # ... (add 'mnli' matched/mismatched logic if needed) ...

    mean_bias = np.mean(bias_per_fold)
    std_bias = np.std(bias_per_fold)
    print(f"Overall Bias mean/std: {mean_bias:.4f}/{std_bias:.4f}")

    # Save results
    output_dir = Path(f"{output_path_base}/{task_name}-{debias_method}-{model_name_slug}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        "metrics_per_fold": metrics_per_fold,
        "bias_per_fold": bias_per_fold,
        "mean_metrics": mean_eval, # Simplify for now
        "std_metrics": std_eval, # Simplify for now
        "mean_bias": mean_bias,
        "std_bias": std_bias
    }

    results_file = output_dir / "results_kfold.json"
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=4)

    print(f"--- Experiment Finished. Results saved to {results_file} ---")

# --- Test Block ---
if __name__ == "__main__":
    """
    This allows the script to be run directly for testing.
    Example:
    python debias_experiment.py --task_name sst2 --debias_method cda
    """
    import argparse
    parser = argparse.ArgumentParser(description="Run a single debiasing experiment.")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--task_name", type=str, default="stsb")
    parser.add_argument("--debias_method", type=str, default="diff")
    parser.add_argument("--k_folds", type=int, default=2) # Default to 2 for fast testing
    args = parser.parse_args()
    
    run_experiment(
        model_name=args.model_name,
        task_name=args.task_name,
        debias_method=args.debias_method,
        k_folds=args.k_folds
    )