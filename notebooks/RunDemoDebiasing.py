import papermill as pm

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

param_grid = [
    {
        "MODEL_NAME": model_name,
        "TASK": task,
        "DEBIAS": debias
    } for model_name in MODELS for task in TASKS for debias in DEBIAS_METHODS
]

for i, params in enumerate(param_grid, 1):
    print('='*50)
    print(params["MODEL_NAME"])
    print(params["TASK"])
    print(params["DEBIAS"])
    print('='*50)
    try:
        pm.execute_notebook(
            "notebooks/DemoDebiasing.ipynb",
            "notebooks/tmp.ipynb",    # notebook de salida
            parameters=params               
        )
    except Exception as e:
        print(e)
