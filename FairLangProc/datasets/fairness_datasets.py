# Standard libraries
import os
import sys
import json
import subprocess
import re
import zipfile
from typing import Union

# data handling libraries
import numpy as np
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

# Available but not accesible from fair-llm-benchmark
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

# Get folders in path_benckmark
def getDatasets() -> list[str]:
    datasets = [dir for dir in os.listdir(path_benchmark) if os.path.isdir(path_benchmark + "/" + dir) and 'git' not in dir]
    return datasets


dirs_benchmark = getDatasets()


argsDict = {
    name:
    ['path', 'output_dir'] if name in need_python
    else ['path']
    for name in dirs_benchmark
}


#=======================================================================
#                               READERS
#=======================================================================

# Define different ways of reading data and store them in a dictionary

def csvReader(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def tsvReader(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep = '\t')

def jsonReader(path: str) -> pd.DataFrame:
    return pd.read_json(path)

def jsonlReader(path: str) -> dict:
    return pd.read_json(path, lines = True)

def txtReader(path: str) -> str:
    with open(path) as file:
        data = file.read()
    return data

def zipReader(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression='zip')

def tgzReader(path: str) -> None:
    return

def pyReader(path: str, *args) -> None:
    """
    Runs python file
    """
    # Path variables
    program = path.split('/')[-1]
    name = path.split('/')[1]
    program_folder = '/'.join(path.split('/')[:-1])

    # Create files folder if it doesn't exist
    try:
        os.listdir(program_folder + '/files')
    except:
        os.mkdir(program_folder + '/files')

    # Create a list with the command we want to run
    process = ['python', program]
    process += [arg for arg in args]

    # Run the command
    subprocess.run(process, cwd = os.path.abspath(program_folder))

    return


def defaultReader(path: str) -> pd.DataFrame:
    """
    Generic data reader
    """
    with open(path) as file:
        data = file.read()
    return data


this_module = sys.modules[__name__]
readers = {ext: getattr(this_module, ext + 'Reader') for ext in extensions}


#=======================================================================
#                               HANDLER
#=======================================================================


def dataReader(path: str, *args) -> Union[pd.DataFrame, dict, str]:
    """
    Use the appropriate reader depending on the extension
    """
    name = path.split('/')[1]
    extension = path.split('.')[-1]

    if extension not in extensions:
        extension = 'default'

    data = readers[extension](path, *args)

    return data


def readFolder(path_folder: str) -> list[pd.DataFrame | dict | str]:
    """
    Recursively read the contents of a folder
    """
    files = dict()
    names = os.listdir(path_folder)
    for file in names:
        path_file = path_folder + '/' + file
        if os.path.isdir(path_file):
            files[file] = readFolder(path_file)
        else:
            files[file] = dataReader(path_file)
    
    return files


def obtainPath(name: str) -> str:
    return path_benchmark + '/' + name + '/data'


def downloadData(name: str, trace: bool = True) -> list[Union[pd.DataFrame, dict, str]]:
    """
    Given a name of a data set, obtain its path and download its contents
    """
    if name.lower() in ('', 'h', 'help'):
        print('List of available datasets:')
        for dataset in getDatasets():
            print(dataset)
        print('all')
        return
    if name in not_redirect:
        print('DATA UNAVAILABLE')
        return None
    if trace: print('Loading ' + name)
    path = obtainPath(name)
    if trace: print('Data set ' + name + ' loaded')
    return readFolder(path)


def runProcess(name: str, *args) -> None:
    """
    Run python process
    """
    path = obtainPath(name)
    files = os.listdir(path)
    for file in files:
        if file.endswith('.py'):
            proc = file
    pyReader(proc, *args)
    return 

#=======================================================================
#                               HANDLERS
#=======================================================================


#            IMPORTANT: THESE DATA SETS NEED TO BE NORMALIZED AND CLASSIFIED...
#--------------------------------------------------------------------------------

#            BBQ
#-----------------------------------------------

def BBQHandler(data: str = '') -> pd.DataFrame:
    path = obtainPath('BBQ')
    if data.lower() in ('', 'h', 'help'):
        print('List of available datasets:')
        for dataset in os.listdir(path):
            if dataset.endswith('.jsonl'):
                print(dataset[:-6])
            else:
                print(dataset)
        print('all')
        return
    elif data.lower() == 'all':
        return readFolder(path)
    elif 'template' in data:
        path += '/templates/' + data + '.csv'
        return csvReader(path)
    else:
        path += '/' + data + '.jsonl'
        return jsonlReader(path)


#            BEC
#-----------------------------------------------

def BECProHandler(data: str = '') -> pd.DataFrame:
    path = obtainPath('BEC-Pro')
    path_en = path + '/BEC-Pro_EN.tsv'
    path_de = path + '/BEC-Pro_DE.tsv'

    if data == 'english':
        return tsvReader(path_en)
    elif data == 'german':
        return tsvReader(path_de)
    elif data == 'all':
        dataframes = dict()
        dataframes['english'] = tsvReader(path_en)
        dataframes['german'] = tsvReader(path_de)
        return dataframes
    else:
        print('Available options: \n English \n German \n All')
        return


#            TO DO 
#===============================================

#            Bias-NLI
#-----------------------------------------------

def BiasNLIHandler(data: str = '') -> pd.DataFrame:
    return


#            BOLD
#-----------------------------------------------

def BOLDHandler(data: str = '') -> pd.DataFrame:
    path = obtainPath('BOLD')
    if data.lower() in ('', 'h', 'help'):
        print('List of available datasets:')
        for dataset in os.listdir(path + '/prompts'):
            print(dataset[:-12])
        print('all')
        return
    elif data.lower() == 'all':
        return readFolder(path)
    elif data == 'prompts':
        return readFolder(path + '/prompts')
    elif data == 'wikipedia':
        return readFolder(path + '/wikipedia')
    elif 'prompt' in data:
        try:
            return jsonReader(path + '/prompts/' + data + '.json')
        except:
            return jsonReader(path + '/prompts/' + data)
    elif 'wiki' in data:
        try:
            return jsonReader(path + '/wikipedia/' + data + '.json')
        except:
            return jsonReader(path + '/wikipedia/' + data) 
    else:
        path += '/' + data + '.csv'
        return csvReader(path)


#            BUG
#-----------------------------------------------

def BUGHandler(data: str = '') -> pd.DataFrame:
    path = obtainPath('BUG')
    if data.lower() in ('', 'h', 'help'):
        print('List of available datasets:')
        for dataset in os.listdir(path):
            print(dataset)
        print('all')
        return
    elif data.lower() == 'all':
        return readFolder(path)
    elif 'csv' in data:
        return csvReader(path)
    elif 'BUG' not in data:
        path += '/' + data + '_BUG.csv'
        return csvReader(path)
    else:
        path += '/' + data + '.csv'
        return csvReader(path)

#            CrowS-Pairs
#-----------------------------------------------

def CrowSPairsHandler(data: str = ''):
    return csvReader(obtainPath('CrowS-pairs') + '/crows_pairs_anonymized.csv')

#            TO DO 
#===============================================

#            Equity-Evaluation-Corpus
#-----------------------------------------------

def EquityEvaluationCorpusHandler():
    return

#            GAP
#-------------------------------------------

def GAPHandler() -> pd.DataFrame:
    return readFolder(obtainPath('GAP'))

#            TO DO 
#===============================================

#            Grep-BiasIR
#-------------------------------------------

def GrepBiasIRHandler():
    return


#            HolisticBias
#-------------------------------------------

def HolisticBiasHandler(data: str = '') -> pd.DataFrame:
    path = obtainPath('HolisticBias') + '/files'
    if data == 'all':
        return readFolder(path)
    elif 'sentences' in data:
        return csvReader(path + '/sentences.csv')
    elif 'phrases' in data:
        return csvReader(path + '/noun_phrases.csv')
    else:
        print('Allowed data sets: \n noun_phrases.csv \n sentences.csv \n All')
        return
    

#            TO DO 
#===============================================

#            HONEST
#-------------------------------------------

def HONESTHandler():
    return

#            TO DO 
#===============================================

#            PANDA
#--------------------------------------------

def PANDAHandler():
    archive = zipfile.ZipFile('images.zip', 'r')
    return


#            Real Toxicity Prompts
#--------------------------------------------

def RealToxicityPromptsHandler(data: str = ''):
    # Se puede meter una api que lo descargue pero eso ya se va viendo
    print('We do not include the data set due to its size. See https://allenai.org/data/real-toxicity-prompts.')
    return


#            TO DO 
#===============================================


#            RedditBias
#--------------------------------------------

def RedditBiasHandler():
    return

#            SetereoSet
#-----------------------------------------------

def StereoSetHandler(data: str = '') -> pd.DataFrame:

    dataframes = dict()
    path = obtainPath('StereoSet') + '/'
    rowDict = {0: 'sentence', 1: 'word'}

    if data == 'word':
        rows = [1]
    elif data == 'sentence':
        rows = [0]
    elif data == 'all':
        rows = [0,1]
    else:
        print('Available options: \n word \n sentence \n all')
        return

    for dataset in ['test', 'dev']:
        raw_data = jsonReader(path+dataset + '.json')
        for row in rows:
            target = []
            bias_type = []
            context = []
            labels = []
            options = []
            item = raw_data.iloc[row, :]['data']
            for item2 in item:
                sentences = []
                label = []
                for item3 in item2['sentences']:
                    label += [item3['gold_label']]
                    sentences += [item3['sentence']]

                labels += ['[' + '/'.join(label) + ']']
                options += ['[' + '/'.join(sentences) + ']']
                target += [item2['target']]
                bias_type += [item2['bias_type']]
                context += [item2['context']]

            dataframes[dataset + '_' + rowDict[row]] = pd.DataFrame(dict(
                options = options,
                context = context,
                target = target,
                bias_type = bias_type,
                labels = labels
                ))

    return dataframes


#            TO DO 
#===============================================

#            TrustGPT
#--------------------------------------------

def TrustGPTHandler():
    return

#            TO DO 
#===============================================

#            UnQover
#--------------------------------------------

def UnQoverHandler():
    return


#            WinoBias
#--------------------------------------------

def WinoBiasHandler(data: str = '') -> dict[pd.DataFrame]:
    
    files = downloadData('WinoBias', trace = False)

    if data in ('h', 'help'):
        print('Available datasets:\n pairs \n gender words \n WinoBias')
    
    if data == 'pairs':
        sets = [files['generalized_swaps.txt'], files['extra_gendered_words.txt']]
        pairs = [tuple(word.strip() for word in pair.split('\t')) for file in sets for pair in file.split('\n')[:-1]]
        return pairs
    
    if 'gender' in data:

        path_gendered_words = file_path + '/GenderSwaps/gendered_words_unidirectional.txt'
        gendered_words = txtReader(path_gendered_words)
        lists = dict()

        lists['male'] = files['male_occupations.txt'].split('\n')
        lists['female'] = files['female_occupations.txt'].split('\n')

        lists['male'] += [pair.split('\t')[0].strip() for pair in gendered_words.split('\n')[:-1]]
        lists['female'] += [pair.split('\t')[1].strip() for pair in gendered_words.split('\n')[:-1]]

        return lists

    else:
        names = [prefix + '_stereotyped_type' + number +'.txt.' + settype
                for prefix in ['anti', 'pro'] for number in ['1', '2'] for settype in ['dev', 'test']]
        dataframes = dict()

        for name in names:
            sentences = [(' '.join(word.split()[1:])) for word in files[name].split('\n')][:-1]
            entities = []
            pronouns = []

            for i in range(len(sentences)):
                instance = sentences[i]
                match = re.findall(r'\[(.*?)\]', instance)
                entities += [match[0]]
                pronouns += [match[1]]
                sentences[i] = instance.replace('[', '').replace(']', '')

            dataframes[name] = pd.DataFrame(dict(
                sentence = sentences,
                entity = entities,
                pronoun = pronouns
            ))

        return dataframes




#            WinoBias+
#--------------------------------------------

def WinoBiasPlusHandler(data: str = ''):
    # Quizás y esto es una idea, se puede meter simplemente el dataframea de WinoBias y meter una opción que sea gender neutral pero es un cristo
    raw = readFolder(obtainPath('WinoBias+'))
    genderSplit = raw['WinoBias+.preprocessed'].split('\n')
    neutralSplit = raw['WinoBias+.references'].split('\n')
    df = pd.DataFrame(dict(
        gendered = genderSplit,
        neutral = neutralSplit
        ))
    return df

#            TO DO 
#===============================================

#            Winogender
#--------------------------------------------

def WinogenderHandler():
    # REQUIERE MÁS PROCESAMIENTO
    path = 'Fair-LLM-Benchmark/Winogender/data/all_sentences.tsv'
    return tsvReader(path)

#            TO DO 
#===============================================

#            WinoQueer
#--------------------------------------------

def WinoQueerHandler():
    return



#=======================================================================
#                               DATA LOADER 
#=======================================================================


class CustomDataset(pt_Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        features = row[1:]
        label = row[0]
        return features, label

    def __len__(self):
        return len(self.dataframe)




def BiasDataLoader(
        dataset: str = None,
        config = None,
        format: str = 'hf'
    ) -> dict[str, Union[pd.DataFrame, list[str], pt_Dataset, hf_Dataset]]:
    """
    Function that loads the specified data set

    Args:
        dataset (str):  name of the dataset. Use getDatasets() to get a list of available datasets
        config (str):   If the data set has different sets, it should be specified in this parameter
        format (str):   either raw (unprocessed), hf (hugging face) or pt (pytorch)
    
    Returns:
        A dictionary with the corresponding data sets in the appropriate format

    """

    _handlers = {
        name: getattr(this_module, re.sub('[^a-zA-Z]', '', name) + 'Handler')
        for name in getDatasets()
    }
    _too_big = ['RealToxicityPrompts']
    _not_implemented = ['Bias-NLI', 'Equity-Evaluation-Corpus', 'Grep-BiasIR', 'HONEST', 'PANDA', 'RealToxicityPrompts', 'RedditBias', 'TrustGPT', 'UnQover' 'WinoGender', 'WinoQueer']
    _need_config = ['BBQ', 'BEC', 'BOLD', 'BUG', 'HolisticBias', 'StereoSet', 'WinoBias']
    _configs = {
        'BBQ': ['Age', 'Disability_Status', 'Gender_indentity', 'Nationality', 'Physical_appearance', 'Race_ethnicity', 'Race_x_gender', 'Race_x_SES', 'Religion', 'SES', 'Sexual_orientation', 'all'],
        'BEC': ['english', 'german', 'all'],
        'BOLD': ['prompts', 'wikipedia', 'all'],
        'BUG': ['balanced', 'full', 'gold', 'all'],
        'CrowS-pairs': None,
        'GAP': None,
        'HolisticBias': ['noun_phrases', 'sentences'],
        'StereoSet': ['word', 'sentence', 'all'],
        'WinoBias': ['pairs', 'gender_words', 'WinoBias'],
        'WinoBias+': None
    }


    if not dataset:
        print('Available datasets:\n' + '='*10 + '\n')
        for name in getDatasets():
            if name not in _not_implemented:
                print(name)
        return

    
    if dataset in _not_implemented:
        raise NotImplementedError('Dataset not implemented')
    
    if dataset in _too_big:
        raise NotImplementedError('Dataset too big')
    
    if dataset in _need_config:
        if config is None:
            print('Available configurations:\n' + '='*10 + '\n')
            for name in _configs[dataset]:
                print(name)
            return
    
    dataRaw = _handlers[dataset](config)
    dataDict = dict()

    if format == 'hf':
        if isinstance(dataRaw, dict):
            for key in dataRaw.keys():
                if not isinstance(dataRaw[key], pd.DataFrame):
                    raise TypeError('Data must be a pandas dataframe')
                dataDict[key] = hf_Dataset.from_pandas(dataRaw[key])
        elif isinstance(dataRaw, pd.DataFrame): 
            dataDict['data'] = hf_Dataset.from_pandas(dataRaw)
        else:
            raise TypeError

    elif format == 'pt':
        if isinstance(dataRaw, dict):
            for key in dataRaw.keys():
                if not isinstance(dataRaw[key], pd.DataFrame):
                    raise TypeError('Data must be a pandas dataframe')
                dataDict[key] = CustomDataset(dataRaw[key])
        elif isinstance(dataRaw, pd.DataFrame): 
            dataDict['data'] = CustomDataset(dataRaw)
        else:
            raise TypeError
    
    elif format == 'raw':
        dataDict = dataRaw

    else:
        raise AttributeError(r'Formats implemented: "hf", "pt", "raw"')

    return dataDict










#=======================================================================
#                               TO DO???? NOT SURE
#=======================================================================




# TO DO: ORGANIZE IT SO I CAN RUN AN ARBITRARY PROCESS

def RunProcessAndDownload(name: str, *kwargs):
    return 