import re
from typing import Optional, List, Dict, Tuple

# ======================================================================
#           Counterfactual Data Augmentation
# ======================================================================

class CounterfactualAugmenter:
    """
    A class to perform counterfactual data augmentation on text data.

    This augmenter replaces specified words with their counterfactual counterparts
    while preserving the original word's case. It can operate on individual examples
    or batches of examples, offering bidirectional augmentation to include both
    original and transformed instances.
    """

    def __init__(self, pairs: Dict[str, str]):
        """
        Initializes the CounterfactualAugmenter with a dictionary of counterfactual pairs.

        Args:
            pairs (Dict[str, str]): A dictionary where keys are words to be replaced
                                    (in lowercase) and values are their counterfactual
                                    replacements.
        Raises:
            ValueError: If 'pairs' is not a dictionary or is empty.
        """
        if not isinstance(pairs, dict) or not pairs:
            raise ValueError("The 'pairs' argument must be a non-empty dictionary.")

        self.pairs = {k.lower(): v for k, v in pairs.items()} # Ensure all keys are lowercase
        # Compile the regex pattern once for efficiency.
        # It matches whole words from the keys of the pairs dictionary, ignoring case.
        pattern_str = r'\b(' + '|'.join(map(re.escape, self.pairs.keys())) + r')\b'
        self.pattern = re.compile(pattern_str, flags=re.IGNORECASE)

    def _preserve_case(self, original_word: str, replacement_word: str) -> str:
        """
        Preserves the capitalization of the original word in the replacement word.

        Args:
            original_word (str): The original word found in the text.
            replacement_word (str): The counterfactual replacement word (typically in lowercase).

        Returns:
            str: The replacement word with its case adjusted to match the original.
        """
        if original_word.isupper():
            return replacement_word.upper()
        elif original_word.istitle():
            return replacement_word.title()
        else:
            return replacement_word

    def _replace_match(self, match: re.Match) -> str:
        """
        Callback function for `re.sub` to perform word replacement with case preservation.

        Args:
            match (re.Match): A match object from the regex pattern.

        Returns:
            str: The replaced word with preserved case.
        """
        original_word = match.group(0)
        # Get the replacement using the lowercase version of the original word
        replacement = self.pairs.get(original_word.lower(), original_word)
        return self._preserve_case(original_word, replacement)

    def _transform_example(
        self, example: Dict, columns: Optional[List[str]] = None
    ) -> Tuple[Dict, bool]:
        """
        Transforms a single training instance (example) using counterfactual pairs.

        Args:
            example (Dict): A dictionary representing a single training instance,
                            where keys are column names and values are text strings.
            columns (Optional[List[str]]): A list of column names on which CDA should be performed.
                                           If None, applies CDA to all columns that are strings.

        Returns:
            Tuple[Dict, bool]:
                transformed_example (Dict): The augmented training instance.
                modified (bool): True if the training instance was modified, False otherwise.
        """
        transformed_example = example.copy()
        modified = False
        # Determine which columns to attempt to transform
        columns_to_transform = columns if columns is not None else example.keys()

        for col in columns_to_transform:
            if col in example and isinstance(example[col], str):
                original_value = example[col]
                # Perform the substitution using the pre-compiled pattern and _replace_match callback
                new_value = self.pattern.sub(self._replace_match, original_value)
                if new_value != original_value:
                    transformed_example[col] = new_value
                    modified = True
        return transformed_example, modified

    def augment_batch(
        self,
        batch: Dict[str, List],
        columns: Optional[List[str]] = None,
        bidirectional: bool = True,
    ) -> Dict[str, List]:
        """
        Performs Counterfactual Data Augmentation (CDA) on a batch of training instances.

        Args:
            batch (Dict[str, List]): A dictionary representing a batch of training instances.
                                     Keys are column names, and values are lists of texts
                                     (or other data) for that column.
            columns (Optional[List[str]]): A list of column names on which CDA should be performed.
                                           If None, applies CDA to all columns that are strings.
            bidirectional (bool): If True, preserves the original training instance and
                                  appends the transformed instance if modified.
                                  If False, only includes the transformed instance if modified,
                                  otherwise includes the original (i.e., modified examples replace
                                  the original, unmodified examples are kept as is).

        Returns:
            Dict[str, List]: An augmented batch of training instances. The lists within
                             the dictionary will have an increased length if augmentation occurred.
        Raises:
            ValueError: If the input 'batch' is not a dictionary or is empty.
        """
        if not isinstance(batch, dict) or not batch:
            raise ValueError("The 'batch' argument must be a non-empty dictionary.")

        # Initialize output dictionary with empty lists for each column
        output = {key: [] for key in batch.keys()}
        
        # Determine the number of examples in the batch by checking the length of any column list
        # We assume all column lists have the same length for a valid batch.
        try:
            num_examples = len(next(iter(batch.values())))
        except StopIteration: # Handle case of empty batch dictionary but valid structure
            return output

        for i in range(num_examples):
            # Reconstruct each example from the batch for processing by _transform_example
            example = {key: batch[key][i] for key in batch.keys()}
            transformed_example, modified = self._transform_example(example, columns)

            if bidirectional and modified:
                # Append both the original and transformed example
                for key in batch.keys():
                    output[key].append(example[key])
                    output[key].append(transformed_example[key])
            elif not bidirectional and modified:
                # Only append the transformed example if modified
                for key in batch.keys():
                    output[key].append(transformed_example[key])
            else:  # Not modified
                # Always append the original example if not modified, regardless of bidirectional flag
                for key in batch.keys():
                    output[key].append(example[key])

        return output