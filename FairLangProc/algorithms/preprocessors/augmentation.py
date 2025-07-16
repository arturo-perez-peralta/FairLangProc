import re
from typing import Optional


#======================================================================
#           Counterfactual Data Augmentation
#======================================================================

def CDAPairs_transform(
    example: dict,
    pairs: dict[str, str],
    columns: Optional[list[str]] = None
    ) -> tuple[dict, bool]:
    r"""Given an example (dictionary with texts in its various fields) and list of counterfactual
    pairs, perform CDA on the specified columns.

    Parameters
    ----------
    example : dict
        Training instance.  
    pairs : dict
        Dictionary of counterfactual pairs
    columns : list[str]
        List of columns on which CDA should be performed.
        If none, applies CDA to all columns.

    Returns
    -------
    transformed_example : dict
        Augmented training instance
    modified : dict
        Whether or not the training instance was augmented        
    """

    # Define the pattern
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, pairs.keys())) + r')\b', flags=re.IGNORECASE)
    
    def replace_match(match):
        word = match.group(0)
        replacement = pairs.get(word.lower(), word)
        return (
            replacement.upper() if word.isupper() else
            replacement.title() if word.istitle() else
            replacement
        )
    
    transformed_example = example.copy()
    modified = False
    columns_to_transform = columns if columns is not None else example.keys()
    
    for col in columns_to_transform:
        if col in example and isinstance(example[col], str):
            new_value = pattern.sub(replace_match, example[col])
            if new_value != example[col]:
                transformed_example[col] = new_value
                modified = True
                
    return transformed_example, modified


def CDA(
    batch: dict,
    pairs: dict[str, str],
    columns: list[str] = None,
    bidirectional: bool = True
    ) -> dict:
    r"""Perform CDA on a batch of training instances.

    Parameters
    ----------
    batch : dict
        Batch of training instances     
    pairs : dict
        Dictionary of counterfactual pairs
    columns : list[str]
        List of columns on which CDA should be performed. If none, applies CDA to all columns.
    bidirectional : bool
        If true, applies bidirectional CDA (preserves original training instance).
        If false, deletes original training instance.

    Returns
    -------
    output : dict
        Augmented training instance.
    modified : dict           
        Whether or not the training instance was augmented.
        
    Example
    -------
    >>> from FairLangProc.algorithms.preprocessors import CDA
    >>> gendered_pairs = [('he', 'she'), ('him', 'her'), ('his', 'hers'), ('actor', 'actress'), ('priest', 'nun'),
        ('father', 'mother'), ('dad', 'mom'), ('daddy', 'mommy'), ('waiter', 'waitress'), ('James', 'Jane')]
    >>>     
    >>> cda_train = Dataset.from_dict(CDA(imdb['train'][:], pairs = dict(gendered_pairs)))
    >>> train_CDA = cda_train.map(tokenize_function, batched=True)
    >>> train_CDA.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    """

    output = {key: [] for key in batch.keys()}
    num_examples = len(next(iter(batch.values())))
    
    for i in range(num_examples):
        # Reconstruct each batch instance
        example = {key: batch[key][i] for key in batch.keys()}
        transformed_example, modified = CDAPairs_transform(example, pairs, columns)

        if bidirectional and modified:
            for key in batch.keys():
                output[key].append(example[key])
                output[key].append(transformed_example[key])

        elif not bidirectional and modified:
            for key in batch.keys():
                output[key].append(transformed_example[key])
        
        elif not modified:
            for key in batch.keys():
                output[key].append(example[key])
                
    return output