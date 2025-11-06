import json

def load_config(config_path="config_results.json"):
    """Loads the JSON configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)

def load_results(config):
    """
    Loads results from JSON files based on the configuration
    and calculates evaluation and bias metrics.
    """
    TASKS = config['TASKS']
    DEBIAS_METHODS = config['DEBIAS_METHODS']
    MODEL_NAME = config['MODEL_NAME']
    EVAL_OR_TEST = config['EVAL_OR_TEST']
    TASK_METRICS = config['TASK_METRICS']

    latex_eval = {task: {} for task in TASKS}
    latex_bias = {task: {} for task in TASKS}

    # Calculate LEN_DICT dynamically from config
    LEN_DICT = {debias: len(TASKS) for debias in DEBIAS_METHODS}
    LEN_DICT['blind'] = len(TASKS) - 1  # Special adjustment as in the notebook

    if config.get('average', False):
        latex_eval['Average'] = {debias: 0.0 for debias in DEBIAS_METHODS}
        latex_bias['Average'] = {debias: 0.0 for debias in DEBIAS_METHODS}

    for task in TASKS:
        for debias in DEBIAS_METHODS:
            path = f"../output/{task}-{debias}-{MODEL_NAME.replace('/', '-')}/results.json"
            try:
                with open(path, "r") as f:
                    resultsDict = json.load(f)
                    
                    print('='*50)
                    print(task, debias)
                    print('-'*50)
                    if task != 'mnli':
                        print('eval: ', resultsDict['eval'][TASK_METRICS[task]]*100)
                    else:
                        for suffix in ['_matched', '_mismatched']:
                            print('eval' + suffix + ': ', resultsDict['eval' + suffix][TASK_METRICS[task]]*100)
                    print('bias: ', resultsDict['bias']['effect_size'])

                    if task != 'mnli':
                        latex_eval[task][debias] = resultsDict[EVAL_OR_TEST][TASK_METRICS[task]]*100
                        if config.get('average', False):
                            latex_eval['Average'][debias] += latex_eval[task][debias] / (LEN_DICT[debias]+1)
                    elif task == 'mnli':
                        latex_eval[task][debias] = [resultsDict[EVAL_OR_TEST + suffix][TASK_METRICS[task]]*100 for suffix in ['_matched', '_mismatched']]
                        if config.get('average', False):
                            latex_eval['Average'][debias] += sum(latex_eval[task][debias]) / (LEN_DICT[debias]+1)
                        
                    latex_bias[task][debias] = resultsDict['bias']['effect_size']
                    if config.get('average', False):
                        latex_bias['Average'][debias] += latex_bias[task][debias] / (LEN_DICT[debias])
            except FileNotFoundError:
                print(f"Warning: Results file not found for {task} - {debias}")
            except Exception as e:
                print(f"Error processing {task} - {debias}: {e}")
    
    return latex_eval, latex_bias

def generate_latex_tables(latex_eval, latex_bias, config):
    """Generates and prints the LaTeX tables."""
    
    dashed = config.get('dashed', False)
    debias_names = config.get('debias_names', {d: d for d in config['DEBIAS_METHODS']})
    TITLE_DICT = config['TITLE_DICT']
    DEBIAS_METHODS = config['DEBIAS_METHODS']

    header_cs = 'c|'*(len(latex_eval.keys())) + 'c'
    header = '\\begin{table}[h] \\n \\t \\small \\n \\t \\centering \\n \\t \\begin{tabular}{' + header_cs + '}\\n'
    
    table_eval = header + '\\t\\t \\hline Debias & '
    table_bias = header + '\\t\\t \\hline Debias & '

    tasks_order = list(latex_eval.keys()) # ['cola', 'sst2', ..., 'Average']

    for (i, task) in enumerate(tasks_order):
        if i == len(tasks_order)-1:
            table_eval += f'{TITLE_DICT[task]} \\\\\\\\ \\\\hline '
            table_bias += f'{TITLE_DICT[task]} \\\\\\\\ \\\\hline '
        else:
            table_eval += f'{TITLE_DICT[task]} & '
            table_bias += f'{TITLE_DICT[task]} & '

    for debias in DEBIAS_METHODS:
        if dashed and debias == 'cda':
            table_eval += f'\\n \\t\\t \\\\hdashline {debias} & '
            table_bias += f'\\n \\t\\t \\\\hdashline {debias} & '
        else:
            table_eval += f'\\n \\t\\t {debias_names.get(debias, debias)} & '
            table_bias += f'\\n \\t\\t {debias_names.get(debias, debias)} & '
        
        for (i, task) in enumerate(tasks_order):
            if i == len(tasks_order)-1:
                try:
                    table_eval += f'{latex_eval[task][debias]:.1f} \\\\\\\\ '
                    table_bias += f'{latex_bias[task][debias]:.3f} \\\\\\\\ '
                except (KeyError, TypeError):
                    table_eval += r'- \\\\ '
                    table_bias += r'- \\\\ '
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
                except (KeyError, TypeError):
                    pass
                finally:
                    table_eval += entry_eval
                    table_bias += entry_bias

    # Note: The LaTeX captions were already in English in the original notebook.
    table_eval += '\\\\hline \\n \\t \\\\end{tabular} \\n \\t \\\\caption{Performance of the different model on GLUE tasks for the validation set. F1 scores are reported for QQP and MRPC, Spearman correlations are reported for STS-B, and accuracy scores are reported for the other tasks.} \\n \\t \\\\label{tab:performance} \\n \\\\end{table}'
    table_bias += '\\\\hline \\n \\t \\\\end{tabular} \\n \\t \\\\caption{WEAT 7 test for the debiasing methods.} \\n \\t \\\\label{tab:bias} \\n \\\\\\\\end{table}'

    print('TABLES \\n', '='*50)
    print(table_eval)
    print(table_bias)
    print('='*50, '\\n'*3)

def main():
    """Main function to run the script."""
    config = load_config()
    latex_eval, latex_bias = load_results(config)
    generate_latex_tables(latex_eval, latex_bias, config)

if __name__ == "__main__":
    main()