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
    """
    function that, given an example (dictionary with texts in its various fields) and list of counterfactual
    pairs, performs CDA on the specified columns

    Args:
        example (dict):         training instance    
        pairs (dict):           dictionary of counterfactual pairs
        columns (list[str]):    list of columns on which CDA should be performed.
                                If none, applies CDA to all columns

    Returns:
        transformed_example (dict): Augmented training instance
        modified (dict):            Whether or not the training instance was augmented        
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
    """
    function that performs CDA on a batch of training instances

    Args:
        batcg (dict):         batch of training instances     
        pairs (dict):           dictionary of counterfactual pairs
        columns (list[str]):    list of columns on which CDA should be performed.
                                If none, applies CDA to all columns
        bidirectional (bool):   If true, applies bidirectional CDA (preserves original training instance).
                                If false, deletes original training instance.

    Returns:
        output (dict):          Augmented training instance
        modified (dict):            Whether or not the training instance was augmented        
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




# Maybe I should erase all of this things below? I mean it was not a fruitful approach...


# class CDATransformer:
#     """
#     A class to perform counterfactual data augmentation for gender debiasing.
#     Given a dictionary of counterfactual pairs, for every occurrence of a gendered word in an input text,
#     this class can generate a new training instance by replacing that occurrence with its counterfactual counterpart.
#     """
# 
#     def __init__(self, pairs: Dict[str, str]):
#         """
#         Initialize the augmenter with a dictionary of counterfactual pairs.
#         
#         Args:
#             pairs (Dict[str, str]): A dictionary where keys are gendered words in lowercase and
#                                     values are their counterfactual counterparts.
#         """
#         self.pairs = pairs
#         # Create a regex pattern that matches any of the keys as a whole word, ignoring case.
#         pattern_str = r'\b(' + '|'.join(map(re.escape, pairs.keys())) + r')\b'
#         self.pattern = re.compile(pattern_str, flags=re.IGNORECASE)
# 
#     def _preserve_case(self, original: str, replacement: str) -> str:
#         """
#         Preserve the capitalization of the original token in the replacement.
#         
#         Args:
#             original (str): The original token.
#             replacement (str): The replacement token in lowercase.
#             
#         Returns:
#             str: The replacement token with the same capitalization as the original.
#         """
#         if original.isupper():
#             return replacement.upper()
#         elif original.istitle():
#             return replacement.title()
#         else:
#             return replacement
# 
#     def augment_instance(self, text: str) -> List[str]:
#         """
#         For a given text, create one augmented instance per occurrence of a gendered word.
#         Each new instance is generated by replacing a single occurrence of a gendered word (found via regex)
#         with its counterfactual counterpart.
#         
#         Args:
#             text (str): The original text.
#             
#         Returns:
#             List[str]: A list of augmented texts (one per gendered word occurrence).
#         """
#         # Find all matches (each match is a gendered word occurrence).
#         matches = list(self.pattern.finditer(text))
#         augmented_texts = []
#         for match in matches:
#             start, end = match.span()
#             original_word = match.group(0)
#             # Look up the replacement using the lowercase key.
#             replacement_word = self._preserve_case(original_word, self.pairs[original_word.lower()])
#             # Construct a new text instance by replacing only this occurrence.
#             new_text = text[:start] + replacement_word + text[end:]
#             augmented_texts.append(new_text)
#         return augmented_texts
#     
#     def _repl(self, match):
#         word = match.group(0)
#         return self._preserve_case(word, self.pairs[word.lower()])
# 
#     def augment_text(self, text: str) -> str:
#         """
#         Augment a given text by creating a new training instance where
#         all occurrences of gendered words are replaced with their counterfactual counterparts
#         
#         Args:
#             text (str): The original text.
#             include_original (bool): Whether to include the original text in the output.
#             
#         Returns:
#             all_replaced (str): augmented text
#         """
#         
#         # Create one instance with all occurrences replaced.
#         all_replaced = self.pattern.sub(self._repl, text)
#         return all_replaced
#     
# 
# 
# def CDAPairs(
#         pairs: dict[str, str],
#         config: str = 'one-sided',
#         columns: list[str] = None,
#         overwrite: bool = False,
#         type: str = 'batch'
#     ):
#     """
#     Load the CDA function.
#     
#     Args:
#         pairs (Dict[str, str]): A dictionary where keys are gendered words in lowercase and
#                                 values are their counterfactual counterparts.
#         config (str): 'one-sided' (default) or 'two-sided'. In 'two-sided', key-value pairs are swapped.
#         columns (list): List of column names to transform. If None, all columns are transformed.
#         overwrite (bool): Whether to overwrite existing examples with the transformed ones or add new ones.
#     """
# 
#     if config == 'two-sided':
#         # If config is 'two-sided', swap the key-value pairs
#         new_pairs = pairs.copy()
#         for word in pairs.keys():
#             new_pairs[pairs[word]] = word
#         pairs = new_pairs
# 
#     # Create the regex pattern for the gendered words
#     pattern_str = r'\b(' + '|'.join(map(re.escape, pairs.keys())) + r')\b'
#     pattern = re.compile(pattern_str, flags=re.IGNORECASE)
# 
#     def replace_match(match):
#         word = match.group(0)
#         replacement = pairs.get(word.lower(), word)
#         # Preserve the case of the original word
#         return (
#             replacement.upper() if word.isupper() else
#             replacement.title() if word.istitle() else
#             replacement
#         )
# 
#     def transform_example(example: dict):
#         try:
#             transformed_example = example.copy()  # Copy original example to modify it
#         except:
#             transformed_example = example
#         modified = False  # Track if any column was changed
# 
#         # If columns is None, transform all columns
#         columns_to_transform = columns if columns is not None else example.keys()
# 
#         # Process each column
#         for column in columns_to_transform:
#             if column in example:
#                 original_text = example[column]
#                 # Ensure that the value being transformed is a string
#                 if isinstance(original_text, str):
#                     transformed_value = pattern.sub(replace_match, original_text)
#                     if transformed_value != original_text:  # Check if the value was modified
#                         transformed_example[column] = transformed_value
#                         modified = True
#                 else:
#                     # If it's not a string, don't transform it
#                     transformed_example[column] = original_text
#         
#         return transformed_example, modified
#         
#     def transform_batch(batch):
#         try:
#             features = batch.features
#         except:
#             features = batch.keys()
#         output = {column: [] for column in features}
# 
#         for example in batch:
#             proc_example, modified = transform_example(example)
#             for column in features:
#                 output[column].append(example[column])
#             if modified:
#                 for column in features:
#                     output[column].append(proc_example[column])
#             
#         return output
# 
#     return transform_batch