import papermill as pm

MODELS = [
    'bert-base-uncased',
#    'deepseek-ai/deepseek-llm-7b-base',
#    'huggyllama/llama-7b'
]

TASKS = [
    "cola",
    "sst2",
    "mrpc",
    "stsb",
    "wnli",
    "rte",
    "qnli",
    "mnli",
    "qqp"
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

DEBIAS_METHODS = [
    "blind",
    "adele",
    "eat"
]

param_grid = [
    {
        "MODEL_NAME": model_name,
        "TASK": task,
        "DEBIAS": debias
    } for model_name in MODELS for task in TASKS for debias in DEBIAS_METHODS
]

param_grid = [
    {'MODEL_NAME': 'bert-base-uncased', 'TASK': 'mnli', 'DEBIAS': 'blind'},
    {'MODEL_NAME': 'bert-base-uncased', 'TASK': 'mnli', 'DEBIAS': 'diff'},
    {'MODEL_NAME': 'bert-base-uncased', 'TASK': 'qqp', 'DEBIAS': 'diff'}
]

for i, params in enumerate(param_grid, 1):
    print('='*50)
    print(params["MODEL_NAME"])
    print(params["TASK"])
    print(params["DEBIAS"])
    print('='*50)
    try:
        pm.execute_notebook(
            "notebooks/DemoDebiasing.ipynb",    # input
            "notebooks/tmp.ipynb",              # output
            parameters=params               
        )
    except Exception as e:
        print(e)
