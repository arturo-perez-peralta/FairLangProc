# Standard libraries
import os, subprocess, re
from typing import Union, List, Dict, Any, Optional

# data handling libraries
import pandas as pd
from torch.utils.data import Dataset as pt_Dataset
from datasets import Dataset as hf_Dataset


#=======================================================================
#                               VARIABLES
#=======================================================================

# File path
file_path = os.path.dirname(os.path.abspath(__file__))

# Allowed file extensions
extensions = [
    'tsv',
    'csv',
    'json',
    'jsonl',
    'txt',
    'tgz',
    'zip',
    'py',
    'default'
]

# Available but not accessible from fair-llm-benchmark
not_redirect = [
    'Equity-Evaluation-Corpus',
    'RealToxicityPrompts'
]

# Need to run a python file 
need_python = [
    'HolisticBias',
    'Bias-NLI',
    'TrustGPT'
]

# Path of the data sets
path_benchmark = file_path + '/Fair-LLM-Benchmark'

# Get folders in path_benchmark
def getDatasets() -> List[str]:
    """Get list of available datasets"""
    if not os.path.exists(path_benchmark):
        return []
    datasets = [dir for dir in os.listdir(path_benchmark) 
               if os.path.isdir(os.path.join(path_benchmark, dir)) and 'git' not in dir]
    return datasets


dirs_benchmark = getDatasets()

argsDict = {
    name: ['path', 'output_dir'] if name in need_python else ['path']
    for name in dirs_benchmark
}


#=======================================================================
#                               READERS
#=======================================================================

def csvReader(path: str) -> pd.DataFrame:
    """Read CSV file"""
    return pd.read_csv(path)

def tsvReader(path: str) -> pd.DataFrame:
    """Read TSV file"""
    return pd.read_csv(path, sep='\t')

def jsonReader(path: str) -> pd.DataFrame:
    """Read JSON file"""
    return pd.read_json(path)

def jsonlReader(path: str) -> pd.DataFrame:
    """Read JSONL file"""
    return pd.read_json(path, lines=True)

def txtReader(path: str) -> str:
    """Read text file"""
    with open(path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data

def zipReader(path: str) -> pd.DataFrame:
    """Read ZIP compressed CSV"""
    return pd.read_csv(path, compression='zip')

def tgzReader(path: str) -> None:
    """Placeholder for TGZ reader"""
    return None

def pyReader(path: str, *args) -> None:
    """Execute Python file with arguments"""
    # Path variables
    program = path.split('/')[-1]
    name = path.split('/')[1]
    program_folder = '/'.join(path.split('/')[:-1])

    # Create files folder if it doesn't exist
    files_dir = os.path.join(program_folder, 'files')
    if not os.path.exists(files_dir):
        os.makedirs(files_dir)

    # Create command list
    process = ['python', program] + list(args)

    # Run the command
    subprocess.run(process, cwd=os.path.abspath(program_folder))

def defaultReader(path: str) -> str:
    """Generic data reader"""
    with open(path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data

readers = {ext: globals()[ext + 'Reader'] for ext in extensions}


#=======================================================================
#                               HANDLER
#=======================================================================

def dataReader(path: str, *args) -> Union[pd.DataFrame, dict, str]:
    """Use appropriate reader based on file extension"""
    extension = path.split('.')[-1]
    
    if extension not in extensions:
        extension = 'default'

    return readers[extension](path, *args)

def readFolder(path_folder: str) -> Dict[str, Any]:
    """Recursively read folder contents"""
    files = {}
    if not os.path.exists(path_folder):
        return files
        
    names = os.listdir(path_folder)
    for file in names:
        path_file = os.path.join(path_folder, file)
        if os.path.isdir(path_file):
            files[file] = readFolder(path_file)
        else:
            try:
                files[file] = dataReader(path_file)
            except Exception as e:
                print(f"Error reading {path_file}: {e}")
                files[file] = None
    
    return files

def obtainPath(name: str) -> str:
    """Get dataset path"""
    return os.path.join(path_benchmark, name, 'data')

def downloadData(name: str, trace: bool = True) -> Optional[Dict[str, Any]]:
    """Download and load dataset contents"""
    if name.lower() in ('', 'h', 'help'):
        print('List of available datasets:')
        for dataset in getDatasets():
            print(dataset)
        print('all')
        return None
        
    if name in not_redirect:
        print('DATA UNAVAILABLE')
        return None
        
    if trace: 
        print('Loading ' + name)
    
    path = obtainPath(name)
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return None
        
    if trace: 
        print('Data set ' + name + ' loaded')
    
    return readFolder(path)

def runProcess(name: str, *args) -> None:
    """Run Python process for dataset"""
    path = obtainPath(name)
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return
        
    files = os.listdir(path)
    py_files = [f for f in files if f.endswith('.py')]
    
    if not py_files:
        print(f"No Python files found in {path}")
        return
        
    proc = py_files[0]  # Use first Python file found
    pyReader(os.path.join(path, proc), *args)


#=======================================================================
#                               HANDLERS
#=======================================================================

def BBQHandler(data: str = '') -> Union[pd.DataFrame, Dict[str, Any], None]:
    """Handle BBQ dataset"""
    path = obtainPath('BBQ')
    
    if data.lower() in ('', 'h', 'help'):
        print('List of available datasets:')
        if os.path.exists(path):
            for dataset in os.listdir(path):
                if dataset.endswith('.jsonl'):
                    print(dataset[:-6])
                else:
                    print(dataset)
        print('all')
        return None
    elif data.lower() == 'all':
        return readFolder(path)
    elif 'template' in data:
        file_path = os.path.join(path, 'templates', data + '.csv')
        return csvReader(file_path) if os.path.exists(file_path) else None
    else:
        file_path = os.path.join(path, data + '.jsonl')
        return jsonlReader(file_path) if os.path.exists(file_path) else None

def BECProHandler(config: str = '') -> Union[pd.DataFrame, Dict[str, pd.DataFrame], None]:
    """Handle BEC-Pro dataset"""
    path = obtainPath('BEC-Pro')
    # Default to 'all' if no config is provided
    current_config = config if config else 'all'

    files = {
        'english': path / 'BEC-Pro_EN.tsv',
        'german': path / 'BEC-Pro_DE.tsv'
    }
    
    if current_config == 'all':
        return {lang: tsvReader(file_path) 
            for lang, file_path in files.items() 
            if file_path.exists()}
    
    if current_config in files and files[current_config].exists():
        return tsvReader(files[current_config])
    
    print('Available options: english, german, all')
    return None

def BiasNLIHandler(data: str = '') -> None:
    """Handle Bias-NLI dataset - not implemented"""
    print("BiasNLI handler not implemented")
    return None

def BOLDHandler(data: str = '') -> Union[pd.DataFrame, Dict[str, Any], None]:
    """Handle BOLD dataset"""
    path = obtainPath('BOLD')
    
    if data.lower() in ('', 'h', 'help'):
        print('List of available datasets:')
        prompts_path = os.path.join(path, 'prompts')
        if os.path.exists(prompts_path):
            for dataset in os.listdir(prompts_path):
                if dataset.endswith('.json'):
                    print(dataset[:-5])
        print('all')
        return None
    elif data.lower() == 'all':
        return readFolder(path)
    elif data == 'prompts':
        return readFolder(os.path.join(path, 'prompts'))
    elif data == 'wikipedia':
        return readFolder(os.path.join(path, 'wikipedia'))
    elif 'prompt' in data:
        file_path = os.path.join(path, 'prompts', data + '.json')
        if not os.path.exists(file_path):
            file_path = os.path.join(path, 'prompts', data)
        return jsonReader(file_path) if os.path.exists(file_path) else None
    elif 'wiki' in data:
        file_path = os.path.join(path, 'wikipedia', data + '.json')
        if not os.path.exists(file_path):
            file_path = os.path.join(path, 'wikipedia', data)
        return jsonReader(file_path) if os.path.exists(file_path) else None
    else:
        file_path = os.path.join(path, data + '.csv')
        return csvReader(file_path) if os.path.exists(file_path) else None

def BUGHandler(data: str = '') -> Union[pd.DataFrame, Dict[str, Any], None]:
    """Handle BUG dataset"""
    path = obtainPath('BUG')
    
    if data.lower() in ('', 'h', 'help'):
        print('List of available datasets:')
        if os.path.exists(path):
            for dataset in os.listdir(path):
                print(dataset)
        print('all')
        return None
    elif data.lower() == 'all':
        return readFolder(path)
    elif 'csv' in data:
        return csvReader(path)
    elif 'BUG' not in data:
        file_path = os.path.join(path, data + '_BUG.csv')
        return csvReader(file_path) if os.path.exists(file_path) else None
    else:
        file_path = os.path.join(path, data + '.csv')
        return csvReader(file_path) if os.path.exists(file_path) else None

def CrowSPairsHandler(data: str = '') -> Optional[pd.DataFrame]:
    """Handle CrowS-Pairs dataset"""
    file_path = os.path.join(obtainPath('CrowS-pairs'), 'crows_pairs_anonymized.csv')
    return csvReader(file_path) if os.path.exists(file_path) else None

def EquityEvaluationCorpusHandler() -> Optional[pd.DataFrame]:
    """Handle Equity Evaluation Corpus"""
    file_path = obtainPath('Equity-Evaluation-Corpus') / 'equity_evaluation_corpus.csv'
    return csvReader(file_path) if file_path else None

def GAPHandler() -> Optional[Dict[str, Any]]:
    """Handle GAP dataset"""
    return readFolder(obtainPath('GAP'))

def GrepBiasIRHandler(config: str = '') -> Optional[Dict[str, Any]]:
    """Handle Grep-BiasIR dataset"""
    path = obtainPath('Grep-BiasIR')
        
    # Default to 'all' if no config is provided or requested
    current_config = config if config else 'all'
    
    if current_config.lower() in ('', 'h', 'help'):
        print('Grep-BiasIR dataset configurations:')
        print('  queries - Load search queries')
        print('  documents - Load document collection')
        print('  relevance - Load relevance judgments')
        print('  all - Load all data')
        return None

    data_dict = {}
    
    if current_config in ['queries', 'all']:
        queries_file = path / 'queries.tsv'
        if queries_file.exists():
            data_dict['queries'] = tsvReader(queries_file)

    if current_config in ['documents', 'all']:
        docs_file = path / 'documents.tsv'
        if docs_file.exists():
            data_dict['documents'] = tsvReader(docs_file)

    if current_config in ['relevance', 'all']:
        rel_file = path / 'relevance.tsv'
        if rel_file.exists():
            data_dict['relevance'] = tsvReader(rel_file)
    
    return data_dict if data_dict else None

def HolisticBiasHandler(data: str = '') -> Union[pd.DataFrame, Dict[str, Any], None]:
    """Handle HolisticBias dataset"""
    path = os.path.join(obtainPath('HolisticBias'), 'files')
    
    if data == 'all':
        return readFolder(path)
    elif 'sentences' in data:
        file_path = os.path.join(path, 'sentences.csv')
        return csvReader(file_path) if os.path.exists(file_path) else None
    elif 'phrases' in data:
        file_path = os.path.join(path, 'noun_phrases.csv')
        return csvReader(file_path) if os.path.exists(file_path) else None
    else:
        print('Allowed data sets: noun_phrases, sentences, all')
        return None

def HONESTHandler() -> None:
    """Handle HONEST - not implemented"""
    print("HONEST handler not implemented")
    return None

def PANDAHandler() -> None:
    """Handle PANDA - not implemented"""
    print("PANDA handler not implemented")
    return None

def RealToxicityPromptsHandler(data: str = '') -> None:
    """Handle Real Toxicity Prompts"""
    print('We do not include the data set due to its size. See https://allenai.org/data/real-toxicity-prompts.')
    return None

def RedditBiasHandler() -> None:
    """Handle RedditBias - not implemented"""
    print("RedditBias handler not implemented")
    return None

def StereoSetHandler(data: str = '') -> Union[Dict[str, pd.DataFrame], None]:
    """Handle StereoSet dataset"""
    path = obtainPath('StereoSet')
    
    if data == 'word':
        rows = [1]
    elif data == 'sentence':
        rows = [0]
    elif data == 'all':
        rows = [0, 1]
    else:
        print('Available options: word, sentence, all')
        return None

    dataframes = {}
    rowDict = {0: 'sentence', 1: 'word'}

    for dataset in ['test', 'dev']:
        file_path = os.path.join(path, dataset + '.json')
        if not os.path.exists(file_path):
            continue
            
        raw_data = jsonReader(file_path)
        
        for row in rows:
            target = []
            bias_type = []
            context = []
            labels = []
            options = []
            
            if 'data' in raw_data.columns and len(raw_data) > row:
                item = raw_data.iloc[row]['data']
                for item2 in item:
                    sentences = []
                    label = []
                    for item3 in item2['sentences']:
                        label.append(item3['gold_label'])
                        sentences.append(item3['sentence'])

                    labels.append('[' + '/'.join(label) + ']')
                    options.append('[' + '/'.join(sentences) + ']')
                    target.append(item2['target'])
                    bias_type.append(item2['bias_type'])
                    context.append(item2['context'])

                dataframes[dataset + '_' + rowDict[row]] = pd.DataFrame({
                    'options': options,
                    'context': context,
                    'target': target,
                    'bias_type': bias_type,
                    'labels': labels
                })

    return dataframes

def TrustGPTHandler() -> None:
    """Handle TrustGPT - not implemented"""
    print("TrustGPT handler not implemented")
    return None

def UnQoverHandler() -> None:
    """Handle UnQover - not implemented"""
    print("UnQover handler not implemented")
    return None

def WinoBiasHandler(data: str = '') -> Union[List, Dict, None]:
    """Handle WinoBias dataset"""
    files = downloadData('WinoBias', trace=False)
    if not files:
        return None

    if data in ('h', 'help'):
        print('Available datasets: pairs, gender_words, WinoBias')
        return None
    
    if data == 'pairs':
        sets = [files.get('generalized_swaps.txt', ''), files.get('extra_gendered_words.txt', '')]
        pairs = [tuple(word.strip() for word in pair.split('\t')) 
                for file_content in sets if file_content
                for pair in file_content.split('\n')[:-1] if pair.strip()]
        return pairs
    
    if 'gender' in data:
        gendered_words_path = os.path.join(file_path, 'GenderSwaps', 'gendered_words_unidirectional.txt')
        lists = {'male': [], 'female': []}
        
        if 'male_occupations.txt' in files:
            lists['male'] = files['male_occupations.txt'].split('\n')
        if 'female_occupations.txt' in files:
            lists['female'] = files['female_occupations.txt'].split('\n')

        if os.path.exists(gendered_words_path):
            gendered_words = txtReader(gendered_words_path)
            for pair in gendered_words.split('\n')[:-1]:
                if '\t' in pair:
                    male_word, female_word = pair.split('\t')
                    lists['male'].append(male_word.strip())
                    lists['female'].append(female_word.strip())

        return lists

    else:
        names = [f"{prefix}_stereotyped_type{number}.txt.{settype}"
                for prefix in ['anti', 'pro'] 
                for number in ['1', '2'] 
                for settype in ['dev', 'test']]
        dataframes = {}

        for name in names:
            if name not in files or not files[name]:
                continue
                
            sentences = [' '.join(word.split()[1:]) 
                        for word in files[name].split('\n')[:-1]]
            entities = []
            pronouns = []

            for i, sentence in enumerate(sentences):
                match = re.findall(r'\[(.*?)\]', sentence)
                if len(match) >= 2:
                    entities.append(match[0])
                    pronouns.append(match[1])
                    sentences[i] = sentence.replace('[', '').replace(']', '')
                else:
                    entities.append('')
                    pronouns.append('')

            dataframes[name] = pd.DataFrame({
                'sentence': sentences,
                'entity': entities,
                'pronoun': pronouns
            })

        return dataframes

def WinoBiasPlusHandler(data: str = '') -> Optional[pd.DataFrame]:
    """Handle WinoBias+ dataset"""
    raw = readFolder(obtainPath('WinoBias+'))
    if not raw:
        return None
        
    gendered_data = raw.get('WinoBias+.preprocessed', '')
    neutral_data = raw.get('WinoBias+.references', '')
    
    gendered_split = gendered_data.split('\n') if gendered_data else []
    neutral_split = neutral_data.split('\n') if neutral_data else []
    
    # Ensure both lists have the same length
    max_len = max(len(gendered_split), len(neutral_split))
    gendered_split.extend([''] * (max_len - len(gendered_split)))
    neutral_split.extend([''] * (max_len - len(neutral_split)))
    
    return pd.DataFrame({
        'gendered': gendered_split,
        'neutral': neutral_split
    })

def WinogenderHandler() -> Optional[pd.DataFrame]:
    """Handle Winogender dataset"""
    file_path = os.path.join(obtainPath('Winogender'), 'all_sentences.tsv')
    return tsvReader(file_path) if os.path.exists(file_path) else None

def WinoQueerHandler() -> None:
    """Handle WinoQueer - not implemented"""
    print("WinoQueer handler not implemented")
    return None


#=======================================================================
#                               DATA LOADER 
#=======================================================================

class CustomDataset(pt_Dataset):
    """Custom PyTorch Dataset wrapper"""
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index].to_numpy()
        features = row[1:]
        label = row[0]
        return features, label

    def __len__(self) -> int:
        return len(self.dataframe)

def BiasDataLoader(
        dataset: Optional[str] = None,
        config: Optional[str] = None,
        format: str = 'hf'
    ) -> Optional[Dict[str, Union[pd.DataFrame, List[str], pt_Dataset, hf_Dataset]]]:
    """
    Load specified bias evaluation dataset

    Args:
        dataset: name of the dataset
        config: dataset configuration if applicable
        format: output format - 'raw', 'hf' (hugging face), or 'pt' (pytorch)
    
    Returns:
        Dictionary with datasets in the appropriate format
    """
    
    # Create handlers mapping
    _handlers = {}
    for name in getDatasets():
        handler_name = re.sub('[^a-zA-Z]', '', name) + 'Handler'
        if handler_name in globals():
            _handlers[name] = globals()[handler_name]

    _too_big = ['RealToxicityPrompts']
    _not_implemented = [
        'Bias-NLI', 'Equity-Evaluation-Corpus', 'Grep-BiasIR', 
        'HONEST', 'PANDA', 'RealToxicityPrompts', 'RedditBias', 
        'TrustGPT', 'UnQover', 'WinoGender', 'WinoQueer'
    ]
    _need_config = ['BBQ', 'BEC-Pro', 'BOLD', 'BUG', 'HolisticBias', 'StereoSet', 'WinoBias']
    
    _configs = {
        'BBQ': ['Age', 'Disability_Status', 'Gender_identity', 'Nationality', 
                'Physical_appearance', 'Race_ethnicity', 'Race_x_gender', 
                'Race_x_SES', 'Religion', 'SES', 'Sexual_orientation', 'all'],
        'BEC-Pro': ['english', 'german', 'all'],
        'BOLD': ['prompts', 'wikipedia', 'all'],
        'BUG': ['balanced', 'full', 'gold', 'all'],
        'CrowS-pairs': None,
        'GAP': None,
        'HolisticBias': ['noun_phrases', 'sentences', 'all'],
        'StereoSet': ['word', 'sentence', 'all'],
        'WinoBias': ['pairs', 'gender_words', 'WinoBias'],
        'WinoBias+': None
    }

    if not dataset:
        print('Available datasets:')
        print('=' * 20)
        for name in getDatasets():
            if name not in _not_implemented:
                print(name)
        return None
    
    if dataset in _not_implemented:
        raise NotImplementedError(f'Dataset {dataset} not implemented')
    
    if dataset in _too_big:
        raise NotImplementedError(f'Dataset {dataset} too big to load')
    
    if dataset not in _handlers:
        raise ValueError(f'Dataset {dataset} not found')
    
    if dataset in _need_config and dataset in _configs:
        if config is None:
            print('Available configurations:')
            print('=' * 20)
            for conf in _configs[dataset]:
                print(conf)
            return None
    
    # Load raw data
    try:
        dataRaw = _handlers[dataset](config)
    except Exception as e:
        print(f"Error loading dataset {dataset}: {e}")
        return None
    
    if dataRaw is None:
        return None
    
    dataDict = {}

    if format == 'hf':
        if isinstance(dataRaw, dict):
            for key, value in dataRaw.items():
                if isinstance(value, pd.DataFrame):
                    dataDict[key] = hf_Dataset.from_pandas(value)
                else:
                    print(f"Skipping {key}: not a DataFrame")
        elif isinstance(dataRaw, pd.DataFrame): 
            dataDict['data'] = hf_Dataset.from_pandas(dataRaw)
        else:
            raise TypeError("Data must be a pandas DataFrame or dict of DataFrames")

    elif format == 'pt':
        if isinstance(dataRaw, dict):
            for key, value in dataRaw.items():
                if isinstance(value, pd.DataFrame):
                    dataDict[key] = CustomDataset(value)
                else:
                    print(f"Skipping {key}: not a DataFrame")
        elif isinstance(dataRaw, pd.DataFrame): 
            dataDict['data'] = CustomDataset(dataRaw)
        else:
            raise TypeError("Data must be a pandas DataFrame or dict of DataFrames")
    
    elif format == 'raw':
        dataDict = dataRaw if isinstance(dataRaw, dict) else {'data': dataRaw}

    else:
        raise ValueError('Formats supported: "hf", "pt", "raw"')

    return dataDict

def RunProcessAndDownload(name: str, **kwargs) -> None:
    """Run process and download dataset"""
    if name in need_python:
        args = [str(v) for v in kwargs.values()]
        runProcess(name, *args)
    
    return downloadData(name)