import os, sys, pytest
import pandas as pd
from torch.utils.data import Dataset as PtDataset
from datasets import Dataset as HfDataset

from FairLangProc.datasets.fairness_datasets import BiasDataLoader

IMPLEMENTED = [
    "BBQ",
    "BEC-Pro",
    "BOLD",
    "BUG",
    "CrowS-Pairs",
    "GAP",
    "StereoSet",
    "UnQover",
    "WinoBias+",
    "WinoBias",
    "Winogender"
]

DATASETS = [
    "BBQ",
    "BEC-Pro",
    "BOLD",
    "BUG",
    "Bias-NLI",
    "CrowS-Pairs",
    "GAP",
    "Grep-BiasIR",
    "HONEST",
    "HolisticBias",
    "PANDA",
    "RedditBias",
    "StereoSet",
    "TrustGPT",
    "UnQover",
    "WinoBias",
    "WinoBias+",
    "WinoQueer",
    "Winogender",
]

REMAINING = [dataset for dataset in DATASETS if dataset not in IMPLEMENTED]

CONFIGURATIONS = {
    "BBQ": ["Age", "Disability_Status", "Gender_identity", "Nationality", "Physical_appearance", "Race_ethnicity", "Race_x_gender", "Race_x_SES", "Religion", "SES", "Sexual_orientation", "all"],
    "BEC-Pro": ["english", "german", "all"],
    "BOLD": ["prompts", "wikipedia", "all"],
    "BUG": ["balanced", "full", "gold", "all"],
    "Bias-NLI": ["process", "load", "all"],
    "CrowS-Pairs": [""],
    "GAP": [""],
    "Grep-BiasIR": ["queries", "documents", "relevance", "all"],
    "HolisticBias": ["noun_phrases", "sentences", "all"],
    "PANDA": ["train", "test", "dev", "all"],
    "RedditBias": ["posts", "comments", "annotations", "all"],
    "StereoSet": ["word", "sentence", "all"],
    "TrustGPT": ["process", "load", "all", "benchmarks"],
    "UnQover": ["questions", "answers", "annotations"],
    "WinoBias": ["pairs", "WinoBias"],
    "WinoBias+": [""],
    "WinoQueer": ["sentences", "templates", "annotations", "all"],
    "Winogender": [""],
}

FORMATS = ["hf", "pt", "raw"]

CLASS_DICT = {
    "hf": HfDataset,
    "pt": PtDataset,
    "raw": pd.DataFrame
}

TEST_CASES_FORMAT = [
    (dataset, config, format)
    for dataset in CONFIGURATIONS.keys()
    for config in CONFIGURATIONS[dataset] 
    for format in FORMATS if dataset in IMPLEMENTED
]

@pytest.mark.parametrize("dataset, config, format", TEST_CASES_FORMAT)
def test_format(dataset, config, format):
    result = BiasDataLoader(dataset = dataset, config = config, format = format)
    assert isinstance(result, dict)
    for key in result:
        assert isinstance(result[key], CLASS_DICT[format])


COLUMNS = {
    "BBQ": ['example_id', 'question_index', 'question_polarity', 'context_condition', 'category', 'answer_info', 'additional_metadata', 'context', 'question', 'ans0', 'ans1', 'ans2', 'label'],
    "BEC-Pro": ['Unnamed: 0', 'Sentence', 'Sent_TM', 'Sent_AM', 'Sent_TAM', 'Template', 'Person', 'Gender', 'Profession', 'Prof_Gender'],
    "BOLD": ['gender_prompt.json', 'political_ideology_prompt.json', 'profession_prompt.json', 'race_prompt.json', 'religious_ideology_prompt.json'],
    "BUG": ['Unnamed: 0', 'sentence_text', 'tokens', 'profession', 'g', 'profession_first_index', 'g_first_index', 'predicted gender', 'stereotype', 'distance', 'num_of_pronouns', 'corpus', 'data_index'],
    "CrowS-Pairs": ['Unnamed: 0', 'sent_more', 'sent_less', 'stereo_antistereo', 'bias_type', 'annotations', 'anon_writer', 'anon_annotators'],
    "GAP": ['ID', 'Text', 'Pronoun', 'Pronoun-offset', 'A', 'A-offset', 'A-coref', 'B', 'B-offset', 'B-coref', 'URL'],
    "HolisticBias": None,
    "StereoSet": ['options', 'context', 'target', 'bias_type', 'labels'],
    "WinoBias+": ['gendered', 'neutral'],
    "WinoBias": ['sentence', 'entity', 'pronoun'],
    "Winogender": ['sentid', 'sentence']
}

TEST_CASES_COLUMNS = list(COLUMNS.keys())

@pytest.mark.parametrize("dataset", TEST_CASES_COLUMNS)
def test_columns(dataset):
    result = BiasDataLoader(dataset = dataset, config = 'all', format = 'raw')
    data = result[list(dataset.keys())[0]]
    for column in COLUMNS[dataset]:
        assert column in data.columns


ROWS = {
    "BBQ": {"Age.jsonl": 3680, "Disability_status.jsonl": 1556, "Gender_identity.jsonl": 5672, "Nationality.jsonl": 3080, "Physical_appearance.jsonl": 1576, "Race_ethnicity.jsonl": 6880, "Race_x_SES.jsonl": 11160, "Race_x_gender.jsonl": 15960, "Religion.jsonl": 1200, "SES.jsonl": 6864, "Sexual_orientation.jsonl": 864, "additional_metadata.csv": 58556, },
    "BEC-Pro": {"english": 5400, "german": 5400, }, 
    "BUG": {"balanced_BUG.csv": 25504, "full_BUG.csv": 105687, "gold_BUG.csv": 1717, }, 
    "CrowS-Pairs": {"data": 1508, }, 
    "GAP": {"gap-development.tsv": 2000, "gap-test.tsv": 2000, "gap-validation.tsv": 454, }, 
    "StereoSet": {"test_sentence": 6374, "test_word": 6392, "dev_sentence": 2123, "dev_word": 2106, }, 
    "WinoBias": {"anti_stereotyped_type1.txt.dev": 396, "anti_stereotyped_type1.txt.test": 396, "anti_stereotyped_type2.txt.dev": 396, "anti_stereotyped_type2.txt.test": 396, "pro_stereotyped_type1.txt.dev": 396, "pro_stereotyped_type1.txt.test": 396, "pro_stereotyped_type2.txt.dev": 396, "pro_stereotyped_type2.txt.test": 396, }, 
    "WinoBias+": {"data": 3167, }, 
    "Winogender": {"data": 720, }, 
}

TEST_CASES_ROWS = list(ROWS.keys())

@pytest.mark.parametrize("dataset", TEST_CASES_ROWS)
def test_row_number(dataset):
    result = BiasDataLoader(dataset = dataset, config = 'all', format = 'raw')
    for key in result:
        if isinstance(result[key], pd.Dataframe):
            assert len(result[key].index) == ROWS[dataset][key]
        elif isinstance(result[key], list):
            assert len(result[key]) == ROWS[dataset][key]