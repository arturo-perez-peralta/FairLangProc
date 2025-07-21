"""
Fair-LLM-Benchmark Data Loader

A comprehensive data loading system for bias evaluation datasets in LLM fairness research.
Supports multiple output formats (raw, HuggingFace, PyTorch) and various file types.
"""

import subprocess, re, logging
from pathlib import Path
import tarfile
from typing import Union, List, Dict, Any, Optional, Tuple

# Data handling libraries
import numpy as np
import pandas as pd
from torch.utils.data import Dataset as PtDataset
from datasets import Dataset as HfDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FairLLMBenchmarkLoader:
    """Main class for loading Fair-LLM-Benchmark datasets"""
    
    def __init__(self, benchmark_path: Optional[str] = None):
        """Initialize the loader with benchmark path"""
        self.file_path = Path(__file__).parent.absolute()
        self.benchmark_path = Path(benchmark_path) if benchmark_path else self.file_path / 'Fair-LLM-Benchmark'
        
        # Supported file extensions
        self.extensions = {
            'tsv': self._read_tsv,
            'csv': self._read_csv,
            'json': self._read_json,
            'jsonl': self._read_jsonl,
            'txt': self._read_txt,
            'tgz': self._read_tgz,
            'zip': self._read_zip,
            'py': self._run_python,
            'default': self._read_default
        }
        
        # Dataset categories
        self.unavailable_datasets = {'Equity-Evaluation-Corpus', 'RealToxicityPrompts'}
        self.python_datasets = {'HolisticBias', 'Bias-NLI', 'TrustGPT'}
        self.config_required = {
            'BBQ', 'BEC-Pro', 'BOLD', 'BUG', 'HolisticBias', 'StereoSet', 'WinoBias'
        }
        
        # Available configurations
        self.configs = {
            'BBQ': ['Age', 'Disability_Status', 'Gender_identity', 'Nationality', 
                   'Physical_appearance', 'Race_ethnicity', 'Race_x_gender', 
                   'Race_x_SES', 'Religion', 'SES', 'Sexual_orientation', 'all'],
            'BEC-Pro': ['english', 'german', 'all'],
            'BOLD': ['prompts', 'wikipedia', 'all'],
            'BUG': ['balanced', 'full', 'gold', 'all'],
            'HolisticBias': ['noun_phrases', 'sentences', 'all'],
            'StereoSet': ['word', 'sentence', 'all'],
            'WinoBias': ['pairs', 'gender_words', 'WinoBias']
        }
        
        # Initialize handlers
        self._init_handlers()
    
    def _init_handlers(self):
        """Initialize dataset-specific handlers"""
        self.handlers = {
            'BBQ': self._handle_bbq,
            'BEC-Pro': self._handle_bec_pro,
            'Bias-NLI': self._handle_bias_nli,
            'BOLD': self._handle_bold,
            'BUG': self._handle_bug,
            'CrowS-Pairs': self._handle_crows_pairs,
            'GAP': self._handle_gap,
            'Grep-BiasIR': self._handle_grep_biasir,
            'HolisticBias': self._handle_holistic_bias,
            'HONEST': self._handle_honest,
            'PANDA': self._handle_panda,
            'RealToxicityPrompts': self._handle_real_toxicity_prompts,
            'RedditBias': self._handle_reddit_bias,
            'StereoSet': self._handle_stereoset,
            'TrustGPT': self._handle_trustgpt,
            'UnQover': self._handle_unqover,
            'WinoBias': self._handle_winobias,
            'WinoBias+': self._handle_winobias_plus,
            'Winogender': self._handle_winogender,
            'WinoQueer': self._handle_winoqueer
        }
    
    # File readers
    def _read_csv(self, path: Path) -> pd.DataFrame:
        """Read CSV file"""
        return pd.read_csv(path)
    
    def _read_tsv(self, path: Path) -> pd.DataFrame:
        """Read TSV file"""
        return pd.read_csv(path, sep='\t')
    
    def _read_json(self, path: Path) -> pd.DataFrame:
        """Read JSON file"""
        return pd.read_json(path)
    
    def _read_jsonl(self, path: Path) -> pd.DataFrame:
        """Read JSONL file"""
        return pd.read_json(path, lines=True)
    
    def _read_txt(self, path: Path) -> str:
        """Read text file"""
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _read_zip(self, path: Path) -> pd.DataFrame:
        """Read ZIP compressed CSV"""
        return pd.read_csv(path, compression='zip')
    
    def _read_tgz(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read TGZ compressed file - Complete implementation"""
        try:
            data_dict = {}
            
            with tarfile.open(path, 'r:gz') as tar:
                # Extract to temporary directory
                temp_dir = path.parent / f'temp_{path.stem}'
                temp_dir.mkdir(exist_ok=True)
                
                try:
                    tar.extractall(temp_dir)
                    
                    # Read extracted files
                    for item in temp_dir.rglob('*'):
                        if item.is_file():
                            # Get relative path for key
                            rel_path = item.relative_to(temp_dir)
                            key = str(rel_path).replace('/', '_').replace('\\', '_')
                            
                            # Read the file
                            data_dict[key] = self._read_file(item)
                    
                    # Clean up temp directory
                    import shutil
                    shutil.rmtree(temp_dir)
                    
                except Exception as e:
                    # Clean up temp directory on error
                    import shutil
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                    raise e
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error reading TGZ file {path}: {e}")
            return None
    
    def _run_python(self, path: Path, *args) -> None:
        """Execute Python file with arguments"""
        program = path.name
        program_folder = path.parent
        
        # Create files folder if it doesn't exist
        files_dir = program_folder / 'files'
        files_dir.mkdir(exist_ok=True)
        
        # Run the command
        process = ['python', program] + list(args)
        try:
            subprocess.run(process, cwd=program_folder, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running {program}: {e}")
    
    def _read_default(self, path: Path) -> str:
        """Generic data reader"""
        with open(path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _read_file(self, path: Path, *args) -> Union[pd.DataFrame, dict, str, None]:
        """Use appropriate reader based on file extension"""
        extension = path.suffix.lstrip('.')
        
        if extension not in self.extensions:
            extension = 'default'
        
        try:
            return self.extensions[extension](path, *args)
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return None
    
    def _read_folder(self, folder_path: Path) -> Dict[str, Any]:
        """Recursively read folder contents"""
        files = {}
        
        if not folder_path.exists():
            logger.warning(f"Folder not found: {folder_path}")
            return files
        
        for item in folder_path.iterdir():
            if item.is_dir():
                files[item.name] = self._read_folder(item)
            else:
                files[item.name] = self._read_file(item)
        
        return files
    
    def get_datasets(self) -> List[str]:
        """Get list of available datasets"""
        if not self.benchmark_path.exists():
            logger.warning(f"Benchmark path not found: {self.benchmark_path}")
            return []
        
        datasets = [
            item.name for item in self.benchmark_path.iterdir()
            if item.is_dir() and 'git' not in item.name.lower()
        ]
        return sorted(datasets)
    
    def _get_dataset_path(self, name: str) -> Path:
        """Get dataset data path"""
        return self.benchmark_path / name / 'data'
    
    # Dataset handlers
    def _handle_bbq(self, config: str = '') -> Union[pd.DataFrame, Dict[str, Any], None]:
        """Handle BBQ dataset"""
        path = self._get_dataset_path('BBQ')
        
        if config.lower() in ('', 'h', 'help'):
            print('Available BBQ datasets:')
            if path.exists():
                for item in path.iterdir():
                    if item.suffix == '.jsonl':
                        print(item.stem)
                    else:
                        print(item.name)
            print('all')
            return None
        
        if config.lower() == 'all':
            return self._read_folder(path)
        
        if 'template' in config:
            file_path = path / 'templates' / f'{config}.csv'
            return self._read_csv(file_path) if file_path.exists() else None
        
        file_path = path / f'{config}.jsonl'
        return self._read_jsonl(file_path) if file_path.exists() else None
    
    def _handle_bec_pro(self, config: str = '') -> Union[pd.DataFrame, Dict[str, pd.DataFrame], None]:
        """Handle BEC-Pro dataset"""
        path = self._get_dataset_path('BEC-Pro')
        
        files = {
            'english': path / 'BEC-Pro_EN.tsv',
            'german': path / 'BEC-Pro_DE.tsv'
        }
        
        if config == 'all':
            return {lang: self._read_tsv(file_path) 
                   for lang, file_path in files.items() 
                   if file_path.exists()}
        
        if config in files and files[config].exists():
            return self._read_tsv(files[config])
        
        print('Available options: english, german, all')
        return None
    
    def _handle_bias_nli(self, config: str = '') -> Optional[Dict[str, Any]]:
        """Handle Bias-NLI dataset - requires Python execution"""
        path = self._get_dataset_path('Bias-NLI')
        
        if config.lower() in ('', 'h', 'help'):
            print('Bias-NLI dataset configurations:')
            print('  process - Run the processing script')
            print('  load - Load processed data')
            print('  all - Process and load data')
            return None
        
        # Check if we need to run the Python processing script
        if config in ('process', 'all'):
            python_files = list(path.glob('*.py'))
            if python_files:
                try:
                    logger.info("Running Bias-NLI processing script...")
                    self._run_python(python_files[0])
                    logger.info("Bias-NLI processing completed")
                except Exception as e:
                    logger.error(f"Error running Bias-NLI processing: {e}")
                    return None
            else:
                logger.warning("No Python processing script found for Bias-NLI")
        
        # Load processed data
        if config in ('load', 'all', ''):
            processed_path = path / 'processed'
            if processed_path.exists():
                return self._read_folder(processed_path)
            else:
                # Try to load from main data folder
                return self._read_folder(path)
        
        return None
    
    def _handle_bold(self, config: str = '') -> Union[pd.DataFrame, Dict[str, Any], None]:
        """Handle BOLD dataset"""
        path = self._get_dataset_path('BOLD')
        
        if config.lower() in ('', 'h', 'help'):
            print('Available BOLD datasets:')
            prompts_path = path / 'prompts'
            if prompts_path.exists():
                for item in prompts_path.iterdir():
                    if item.suffix == '.json':
                        print(item.stem)
            print('all, prompts, wikipedia')
            return None
        
        if config.lower() == 'all':
            return self._read_folder(path)
        
        if config == 'prompts':
            return self._read_folder(path / 'prompts')
        
        if config == 'wikipedia':
            return self._read_folder(path / 'wikipedia')
        
        # Try prompts folder first
        for folder in ['prompts', 'wikipedia']:
            file_path = path / folder / f'{config}.json'
            if file_path.exists():
                return self._read_json(file_path)
        
        # Try root folder
        file_path = path / f'{config}.csv'
        return self._read_csv(file_path) if file_path.exists() else None
    
    def _handle_bug(self, config: str = '') -> Union[pd.DataFrame, Dict[str, Any], None]:
        """Handle BUG dataset"""
        path = self._get_dataset_path('BUG')
        
        if config.lower() in ('', 'h', 'help'):
            print('Available BUG datasets:')
            if path.exists():
                for item in path.iterdir():
                    print(item.name)
            print('all')
            return None
        
        if config.lower() == 'all':
            return self._read_folder(path)
        
        # Try different file patterns
        patterns = [f'{config}_BUG.csv', f'{config}.csv', f'BUG_{config}.csv']
        
        for pattern in patterns:
            file_path = path / pattern
            if file_path.exists():
                return self._read_csv(file_path)
        
        return None
    
    def _handle_crows_pairs(self, config: str = '') -> Optional[pd.DataFrame]:
        """Handle CrowS-Pairs dataset"""
        file_path = self._get_dataset_path('CrowS-Pairs') / 'crows_pairs_anonymized.csv'
        return self._read_csv(file_path) if file_path.exists() else None
    
    def _handle_gap(self, config: str = '') -> Optional[Dict[str, Any]]:
        """Handle GAP dataset"""
        return self._read_folder(self._get_dataset_path('GAP'))
    
    def _handle_holistic_bias(self, config: str = '') -> Union[pd.DataFrame, Dict[str, Any], None]:
        """Handle HolisticBias dataset"""
        path = self._get_dataset_path('HolisticBias') / 'files'
        
        if config == 'all':
            return self._read_folder(path)
        
        file_mapping = {
            'sentences': 'sentences.csv',
            'phrases': 'noun_phrases.csv',
            'noun_phrases': 'noun_phrases.csv'
        }
        
        if config in file_mapping:
            file_path = path / file_mapping[config]
            return self._read_csv(file_path) if file_path.exists() else None
        
        print('Available datasets: noun_phrases, sentences, all')
        return None
    
    def _handle_stereoset(self, config: str = '') -> Union[Dict[str, pd.DataFrame], None]:
        """Handle StereoSet dataset"""
        path = self._get_dataset_path('StereoSet')
        
        config_mapping = {
            'word': [1],
            'sentence': [0],
            'all': [0, 1]
        }
        
        if config not in config_mapping:
            print('Available options: word, sentence, all')
            return None
        
        rows = config_mapping[config]
        dataframes = {}
        row_dict = {0: 'sentence', 1: 'word'}
        
        for dataset in ['test', 'dev']:
            file_path = path / f'{dataset}.json'
            if not file_path.exists():
                continue
            
            raw_data = self._read_json(file_path)
            
            for row in rows:
                if 'data' not in raw_data.columns or len(raw_data) <= row:
                    continue
                
                target, bias_type, context, labels, options = [], [], [], [], []
                
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
                
                dataframes[f'{dataset}_{row_dict[row]}'] = pd.DataFrame({
                    'options': options,
                    'context': context,
                    'target': target,
                    'bias_type': bias_type,
                    'labels': labels
                })
        
        return dataframes
    
    def _handle_winobias(self, config: str = '') -> Union[List, Dict, None]:
        """Handle WinoBias dataset"""
        files = self._read_folder(self._get_dataset_path('WinoBias'))
        
        if not files:
            return None
        
        if config in ('h', 'help'):
            print('Available datasets: pairs, gender_words, WinoBias')
            return None
        
        if config == 'pairs':
            pairs = []
            for file_name in ['generalized_swaps.txt', 'extra_gendered_words.txt']:
                if file_name in files and files[file_name]:
                    pairs.extend([
                        tuple(word.strip() for word in pair.split('\t'))
                        for pair in files[file_name].split('\n')
                        if pair.strip() and '\t' in pair
                    ])
            return pairs
        
        if 'gender' in config:
            lists = {'male': [], 'female': []}
            
            # Load occupation lists
            if 'male_occupations.txt' in files:
                lists['male'] = [line.strip() for line in files['male_occupations.txt'].split('\n') if line.strip()]
            if 'female_occupations.txt' in files:
                lists['female'] = [line.strip() for line in files['female_occupations.txt'].split('\n') if line.strip()]
            
            # Load gendered words
            gendered_words_path = self.file_path / 'GenderSwaps' / 'gendered_words_unidirectional.txt'
            if gendered_words_path.exists():
                gendered_words = self._read_txt(gendered_words_path)
                for pair in gendered_words.split('\n'):
                    if '\t' in pair:
                        male_word, female_word = pair.split('\t')
                        lists['male'].append(male_word.strip())
                        lists['female'].append(female_word.strip())
            
            return lists
        
        # Process WinoBias files
        prefixes = ['anti', 'pro']
        numbers = ['1', '2']
        set_types = ['dev', 'test']
        
        dataframes = {}
        
        for prefix in prefixes:
            for number in numbers:
                for set_type in set_types:
                    file_name = f'{prefix}_stereotyped_type{number}.txt.{set_type}'
                    
                    if file_name not in files or not files[file_name]:
                        continue
                    
                    sentences = []
                    entities = []
                    pronouns = []
                    
                    for line in files[file_name].split('\n'):
                        if not line.strip():
                            continue
                        
                        # Remove line numbers
                        sentence = ' '.join(line.split()[1:])
                        
                        # Extract entities and pronouns in brackets
                        matches = re.findall(r'\[(.*?)\]', sentence)
                        
                        if len(matches) >= 2:
                            entities.append(matches[0])
                            pronouns.append(matches[1])
                        else:
                            entities.append('')
                            pronouns.append('')
                        
                        # Clean sentence
                        clean_sentence = sentence.replace('[', '').replace(']', '')
                        sentences.append(clean_sentence)
                    
                    dataframes[file_name] = pd.DataFrame({
                        'sentence': sentences,
                        'entity': entities,
                        'pronoun': pronouns
                    })
        
        return dataframes
    
    def _handle_winobias_plus(self, config: str = '') -> Optional[pd.DataFrame]:
        """Handle WinoBias+ dataset"""
        files = self._read_folder(self._get_dataset_path('WinoBias+'))
        
        if not files:
            return None
        
        gendered_data = files.get('WinoBias+.preprocessed', '')
        neutral_data = files.get('WinoBias+.references', '')
        
        gendered_lines = [line.strip() for line in gendered_data.split('\n') if line.strip()] if gendered_data else []
        neutral_lines = [line.strip() for line in neutral_data.split('\n') if line.strip()] if neutral_data else []
        
        # Ensure both lists have the same length
        max_len = max(len(gendered_lines), len(neutral_lines))
        gendered_lines.extend([''] * (max_len - len(gendered_lines)))
        neutral_lines.extend([''] * (max_len - len(neutral_lines)))
        
        return pd.DataFrame({
            'gendered': gendered_lines,
            'neutral': neutral_lines
        })
    
    def _handle_winogender(self, config: str = '') -> Optional[pd.DataFrame]:
        """Handle Winogender dataset"""
        file_path = self._get_dataset_path('Winogender') / 'all_sentences.tsv'
        return self._read_tsv(file_path) if file_path.exists() else None
    
    def _handle_grep_biasir(self, config: str = '') -> Optional[Dict[str, Any]]:
        """Handle Grep-BiasIR dataset"""
        path = self._get_dataset_path('Grep-BiasIR')
        
        if config.lower() in ('', 'h', 'help'):
            print('Grep-BiasIR dataset configurations:')
            print('  queries - Load search queries')
            print('  documents - Load document collection')
            print('  relevance - Load relevance judgments')
            print('  all - Load all data')
            return None
        
        if not path.exists():
            logger.error(f"Grep-BiasIR dataset path not found: {path}")
            return None
        
        data_dict = {}
        
        if config == 'queries' or config == 'all':
            queries_file = path / 'queries.tsv'
            if queries_file.exists():
                data_dict['queries'] = self._read_tsv(queries_file)
            else:
                # Try alternative file names
                for alt_name in ['queries.csv', 'query.tsv', 'query.csv']:
                    alt_file = path / alt_name
                    if alt_file.exists():
                        data_dict['queries'] = self._read_file(alt_file)
                        break
        
        if config == 'documents' or config == 'all':
            docs_file = path / 'documents.tsv'
            if docs_file.exists():
                data_dict['documents'] = self._read_tsv(docs_file)
            else:
                # Try alternative file names
                for alt_name in ['documents.csv', 'docs.tsv', 'docs.csv', 'collection.tsv']:
                    alt_file = path / alt_name
                    if alt_file.exists():
                        data_dict['documents'] = self._read_file(alt_file)
                        break
        
        if config == 'relevance' or config == 'all':
            rel_file = path / 'relevance.tsv'
            if rel_file.exists():
                data_dict['relevance'] = self._read_tsv(rel_file)
            else:
                # Try alternative file names
                for alt_name in ['relevance.csv', 'qrels.tsv', 'qrels.csv', 'judgments.tsv']:
                    alt_file = path / alt_name
                    if alt_file.exists():
                        data_dict['relevance'] = self._read_file(alt_file)
                        break
        
        if config == 'all':
            # Also load any other files in the directory
            for item in path.iterdir():
                if item.is_file() and item.name not in [f['name'] for f in data_dict.values() if isinstance(f, dict) and 'name' in f]:
                    data_dict[item.stem] = self._read_file(item)
        
        return data_dict if data_dict else None

    def _handle_honest(self, config: str = '') -> Optional[Dict[str, Any]]:
        """Handle HONEST dataset"""
        path = self._get_dataset_path('HONEST')
        
        if config.lower() in ('', 'h', 'help'):
            print('HONEST dataset configurations:')
            print('  templates - Load template sentences')
            print('  completions - Load model completions')
            print('  annotations - Load human annotations')
            print('  all - Load all data')
            return None
        
        if not path.exists():
            logger.error(f"HONEST dataset path not found: {path}")
            return None
        
        data_dict = {}
        
        # Load templates
        if config == 'templates' or config == 'all':
            templates_file = path / 'templates.csv'
            if templates_file.exists():
                data_dict['templates'] = self._read_csv(templates_file)
            else:
                # Try finding template files
                template_files = list(path.glob('*template*'))
                if template_files:
                    data_dict['templates'] = self._read_file(template_files[0])
        
        # Load completions
        if config == 'completions' or config == 'all':
            completions_file = path / 'completions.jsonl'
            if completions_file.exists():
                data_dict['completions'] = self._read_jsonl(completions_file)
            else:
                # Try alternative names
                for alt_name in ['completions.json', 'responses.jsonl', 'responses.json']:
                    alt_file = path / alt_name
                    if alt_file.exists():
                        data_dict['completions'] = self._read_file(alt_file)
                        break
        
        # Load annotations
        if config == 'annotations' or config == 'all':
            annotations_file = path / 'annotations.csv'
            if annotations_file.exists():
                data_dict['annotations'] = self._read_csv(annotations_file)
            else:
                # Try alternative names
                for alt_name in ['annotations.tsv', 'labels.csv', 'labels.tsv']:
                    alt_file = path / alt_name
                    if alt_file.exists():
                        data_dict['annotations'] = self._read_file(alt_file)
                        break
        
        # Load all other files if 'all' is specified
        if config == 'all':
            for item in path.iterdir():
                if item.is_file() and item.stem not in data_dict:
                    data_dict[item.stem] = self._read_file(item)
        
        return data_dict if data_dict else None

    def _handle_panda(self, config: str = '') -> Optional[Dict[str, Any]]:
        """Handle PANDA dataset"""
        path = self._get_dataset_path('PANDA')
        
        if config.lower() in ('', 'h', 'help'):
            print('PANDA dataset configurations:')
            print('  train - Load training data')
            print('  test - Load test data')
            print('  dev - Load development/validation data')
            print('  all - Load all splits')
            return None
        
        if not path.exists():
            logger.error(f"PANDA dataset path not found: {path}")
            return None
        
        data_dict = {}
        splits = ['train', 'test', 'dev', 'validation']
        
        if config == 'all':
            target_splits = splits
        elif config in splits:
            target_splits = [config]
        else:
            target_splits = splits
        
        for split in target_splits:
            # Try different file formats and naming conventions
            possible_files = [
                f'{split}.jsonl',
                f'{split}.json',
                f'{split}.csv',
                f'{split}.tsv',
                f'panda_{split}.jsonl',
                f'panda_{split}.json',
                f'panda_{split}.csv',
                f'panda_{split}.tsv'
            ]
            
            for filename in possible_files:
                file_path = path / filename
                if file_path.exists():
                    data_dict[split] = self._read_file(file_path)
                    break
        
        # If no specific splits found, try to load all files
        if not data_dict:
            for item in path.iterdir():
                if item.is_file():
                    data_dict[item.stem] = self._read_file(item)
        
        return data_dict if data_dict else None

    def _handle_reddit_bias(self, config: str = '') -> Optional[Dict[str, Any]]:
        """Handle RedditBias dataset"""
        path = self._get_dataset_path('RedditBias')
        
        if config.lower() in ('', 'h', 'help'):
            print('RedditBias dataset configurations:')
            print('  posts - Load Reddit posts')
            print('  comments - Load Reddit comments')
            print('  annotations - Load bias annotations')
            print('  all - Load all data')
            return None
        
        if not path.exists():
            logger.error(f"RedditBias dataset path not found: {path}")
            return None
        
        data_dict = {}
        
        # Load posts
        if config == 'posts' or config == 'all':
            posts_files = list(path.glob('*post*')) + list(path.glob('*submission*'))
            if posts_files:
                data_dict['posts'] = self._read_file(posts_files[0])
        
        # Load comments
        if config == 'comments' or config == 'all':
            comments_files = list(path.glob('*comment*'))
            if comments_files:
                data_dict['comments'] = self._read_file(comments_files[0])
        
        # Load annotations
        if config == 'annotations' or config == 'all':
            annotation_files = list(path.glob('*annotation*')) + list(path.glob('*label*'))
            if annotation_files:
                data_dict['annotations'] = self._read_file(annotation_files[0])
        
        # Load all files if 'all' specified or no specific files found
        if config == 'all' or not data_dict:
            for item in path.iterdir():
                if item.is_file() and item.stem not in data_dict:
                    data_dict[item.stem] = self._read_file(item)
        
        return data_dict if data_dict else None

    def _handle_trustgpt(self, config: str = '') -> Optional[Dict[str, Any]]:
        """Handle TrustGPT dataset - requires Python execution"""
        path = self._get_dataset_path('TrustGPT')
        
        if config.lower() in ('', 'h', 'help'):
            print('TrustGPT dataset configurations:')
            print('  process - Run the processing script')
            print('  load - Load processed data')
            print('  all - Process and load data')
            print('  benchmarks - Load specific benchmark data')
            return None
        
        # Check if we need to run the Python processing script
        if config in ('process', 'all'):
            python_files = list(path.glob('*.py'))
            if python_files:
                try:
                    logger.info("Running TrustGPT processing script...")
                    self._run_python(python_files[0])
                    logger.info("TrustGPT processing completed")
                except Exception as e:
                    logger.error(f"Error running TrustGPT processing: {e}")
                    return None
            else:
                logger.warning("No Python processing script found for TrustGPT")
        
        # Load processed data
        if config in ('load', 'all', 'benchmarks', ''):
            data_dict = {}
            
            # Look for processed data folder
            processed_path = path / 'processed'
            if processed_path.exists():
                data_dict.update(self._read_folder(processed_path))
            
            # Look for benchmark data
            benchmarks_path = path / 'benchmarks'
            if benchmarks_path.exists():
                data_dict['benchmarks'] = self._read_folder(benchmarks_path)
            
            # Load any other data files
            for item in path.iterdir():
                if item.is_file() and item.suffix in ['.json', '.jsonl', '.csv', '.tsv']:
                    data_dict[item.stem] = self._read_file(item)
            
            return data_dict if data_dict else None
        
        return None

    def _handle_unqover(self, config: str = '') -> Optional[Dict[str, Any]]:
        """Handle UnQover dataset"""
        path = self._get_dataset_path('UnQover')
        
        if config.lower() in ('', 'h', 'help'):
            print('UnQover dataset configurations:')
            print('  questions - Load questions')
            print('  answers - Load answers')
            print('  annotations - Load coverage annotations')
            print('  all - Load all data')
            return None
        
        if not path.exists():
            logger.error(f"UnQover dataset path not found: {path}")
            return None
        
        data_dict = {}
        
        # Load questions
        if config == 'questions' or config == 'all':
            question_files = list(path.glob('*question*')) + list(path.glob('*queries*'))
            if question_files:
                data_dict['questions'] = self._read_file(question_files[0])
        
        # Load answers
        if config == 'answers' or config == 'all':
            answer_files = list(path.glob('*answer*')) + list(path.glob('*response*'))
            if answer_files:
                data_dict['answers'] = self._read_file(answer_files[0])
        
        # Load annotations
        if config == 'annotations' or config == 'all':
            annotation_files = (list(path.glob('*annotation*')) + 
                            list(path.glob('*coverage*')) + 
                            list(path.glob('*label*')))
            if annotation_files:
                data_dict['annotations'] = self._read_file(annotation_files[0])
        
        # Load all files if 'all' specified or no specific files found
        if config == 'all' or not data_dict:
            for item in path.iterdir():
                if item.is_file() and item.stem not in data_dict:
                    data_dict[item.stem] = self._read_file(item)
        
        return data_dict if data_dict else None

    def _handle_winoqueer(self, config: str = '') -> Optional[Dict[str, Any]]:
        """Handle WinoQueer dataset"""
        path = self._get_dataset_path('WinoQueer')
        
        if config.lower() in ('', 'h', 'help'):
            print('WinoQueer dataset configurations:')
            print('  sentences - Load WinoQueer sentences')
            print('  templates - Load sentence templates')
            print('  annotations - Load annotations')
            print('  all - Load all data')
            return None
        
        if not path.exists():
            logger.error(f"WinoQueer dataset path not found: {path}")
            return None
        
        data_dict = {}
        
        # Load sentences
        if config == 'sentences' or config == 'all':
            sentence_files = (list(path.glob('*sentence*')) + 
                            list(path.glob('*winoqueer*')) +
                            list(path.glob('*wq*')))
            if sentence_files:
                data_dict['sentences'] = self._read_file(sentence_files[0])
        
        # Load templates
        if config == 'templates' or config == 'all':
            template_files = list(path.glob('*template*'))
            if template_files:
                data_dict['templates'] = self._read_file(template_files[0])
        
        # Load annotations
        if config == 'annotations' or config == 'all':
            annotation_files = list(path.glob('*annotation*')) + list(path.glob('*label*'))
            if annotation_files:
                data_dict['annotations'] = self._read_file(annotation_files[0])
        
        # Load all files if 'all' specified or no specific files found
        if config == 'all' or not data_dict:
            for item in path.iterdir():
                if item.is_file() and item.stem not in data_dict:
                    data_dict[item.stem] = self._read_file(item)
        
        return data_dict if data_dict else None
    
    def _handle_real_toxicity_prompts(self, config: str = '') -> Optional[Dict[str, Any]]:
        """Handle RealToxicityPrompts dataset"""
        path = self._get_dataset_path('RealToxicityPrompts')
        
        if config.lower() in ('', 'h', 'help'):
            print('RealToxicityPrompts dataset configurations:')
            print('  prompts - Load toxic prompts')
            print('  generations - Load model generations')
            print('  annotations - Load toxicity annotations')
            print('  train - Load training split')
            print('  test - Load test split')
            print('  dev - Load development split')
            print('  all - Load all available data')
            print('\nNote: This dataset is not included by default due to size.')
            print('Download from: https://allenai.org/data/real-toxicity-prompts')
            return None
        
        if not path.exists():
            print('RealToxicityPrompts dataset not found locally.')
            print('This dataset is not included due to size constraints.')
            print('Please download from: https://allenai.org/data/real-toxicity-prompts')
            print(f'Expected location: {path}')
            return None
        
        data_dict = {}
        
        try:
            # Load prompts
            if config == 'prompts' or config == 'all':
                prompts_files = (list(path.glob('*prompt*')) + 
                            list(path.glob('*input*')) +
                            list(path.glob('prompts.*')))
                
                for prompts_file in prompts_files:
                    if prompts_file.exists():
                        key = f'prompts_{prompts_file.stem}' if len(prompts_files) > 1 else 'prompts'
                        data_dict[key] = self._read_file(prompts_file)
                        logger.info(f"Loaded prompts from {prompts_file.name}")
                        break
            
            # Load generations
            if config == 'generations' or config == 'all':
                generation_files = (list(path.glob('*generation*')) + 
                                list(path.glob('*output*')) +
                                list(path.glob('*completion*')) +
                                list(path.glob('generations.*')))
                
                for gen_file in generation_files:
                    if gen_file.exists():
                        key = f'generations_{gen_file.stem}' if len(generation_files) > 1 else 'generations'
                        data_dict[key] = self._read_file(gen_file)
                        logger.info(f"Loaded generations from {gen_file.name}")
                        break
            
            # Load annotations/toxicity scores
            if config == 'annotations' or config == 'all':
                annotation_files = (list(path.glob('*annotation*')) + 
                                list(path.glob('*toxicity*')) +
                                list(path.glob('*score*')) +
                                list(path.glob('*label*')) +
                                list(path.glob('annotations.*')))
                
                for ann_file in annotation_files:
                    if ann_file.exists():
                        key = f'annotations_{ann_file.stem}' if len(annotation_files) > 1 else 'annotations'
                        data_dict[key] = self._read_file(ann_file)
                        logger.info(f"Loaded annotations from {ann_file.name}")
                        break
            
            # Load specific splits
            splits = ['train', 'test', 'dev', 'validation']
            for split in splits:
                if config == split or config == 'all':
                    split_files = []
                    
                    # Look for various naming patterns
                    patterns = [
                        f'{split}.*',
                        f'*{split}*',
                        f'realtoxicityprompts_{split}.*',
                        f'rtp_{split}.*'
                    ]
                    
                    for pattern in patterns:
                        split_files.extend(list(path.glob(pattern)))
                    
                    # Remove duplicates and filter for data files
                    split_files = list(set([f for f in split_files 
                                        if f.suffix in ['.jsonl', '.json', '.csv', '.tsv', '.txt']]))
                    
                    if split_files:
                        # Use the first file found for this split
                        split_file = split_files[0]
                        data_dict[split] = self._read_file(split_file)
                        logger.info(f"Loaded {split} split from {split_file.name}")
            
            # Load perspective API scores if available
            if config == 'perspective' or config == 'all':
                perspective_files = (list(path.glob('*perspective*')) + 
                                list(path.glob('*api*')))
                
                for persp_file in perspective_files:
                    if persp_file.exists():
                        key = f'perspective_{persp_file.stem}' if len(perspective_files) > 1 else 'perspective'
                        data_dict[key] = self._read_file(persp_file)
                        logger.info(f"Loaded Perspective API scores from {persp_file.name}")
                        break
            
            # Load model-specific generations if available
            if config == 'models' or config == 'all':
                models_dir = path / 'models'
                if models_dir.exists():
                    data_dict['models'] = self._read_folder(models_dir)
                    logger.info("Loaded model-specific generations")
            
            # Load all files if 'all' specified or no specific files found
            if config == 'all' or not data_dict:
                logger.info("Loading all available files...")
                
                for item in path.iterdir():
                    if item.is_file() and item.stem not in data_dict:
                        # Skip very large files unless specifically requested
                        if item.stat().st_size > 100 * 1024 * 1024:  # 100MB threshold
                            logger.warning(f"Skipping large file {item.name} (size: {item.stat().st_size / (1024*1024):.1f}MB)")
                            continue
                        
                        try:
                            data_dict[item.stem] = self._read_file(item)
                            logger.info(f"Loaded {item.name}")
                        except Exception as e:
                            logger.warning(f"Could not load {item.name}: {e}")
                    
                    elif item.is_dir() and item.name not in data_dict:
                        try:
                            data_dict[item.name] = self._read_folder(item)
                            logger.info(f"Loaded directory {item.name}")
                        except Exception as e:
                            logger.warning(f"Could not load directory {item.name}: {e}")
            
            # Post-process data if needed
            if data_dict:
                # Combine prompts and generations if both are available
                if 'prompts' in data_dict and 'generations' in data_dict:
                    try:
                        prompts_df = data_dict['prompts']
                        generations_df = data_dict['generations']
                        
                        if isinstance(prompts_df, pd.DataFrame) and isinstance(generations_df, pd.DataFrame):
                            # Try to merge on common columns
                            common_cols = set(prompts_df.columns) & set(generations_df.columns)
                            if common_cols:
                                merged = prompts_df.merge(generations_df, on=list(common_cols), how='outer')
                                data_dict['combined'] = merged
                                logger.info("Created combined prompts+generations dataset")
                    except Exception as e:
                        logger.warning(f"Could not combine prompts and generations: {e}")
                
                # Add metadata
                data_dict['_metadata'] = {
                    'dataset': 'RealToxicityPrompts',
                    'description': 'Dataset for evaluating neural toxic degeneration in language models',
                    'source': 'https://allenai.org/data/real-toxicity-prompts',
                    'paper': 'RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models',
                    'loaded_components': list(data_dict.keys()),
                    'load_time': pd.Timestamp.now().isoformat()
                }
                
                logger.info(f"Successfully loaded RealToxicityPrompts with components: {list(data_dict.keys())}")
                return data_dict
            
            else:
                logger.warning("No data files found in RealToxicityPrompts directory")
                return None
        
        except Exception as e:
            logger.error(f"Error loading RealToxicityPrompts dataset: {e}")
            return None
    
    def run_process_and_download(self, name: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Run process for Python datasets and download data"""
        if name in self.python_datasets:
            args = [str(v) for v in kwargs.values()]
            self._run_python_dataset(name, *args)
        
        return self.load_dataset(name)
    
    def _run_python_dataset(self, name: str, *args) -> None:
        """Run Python script for dataset processing"""
        path = self._get_dataset_path(name)
        
        if not path.exists():
            logger.error(f"Dataset path not found: {path}")
            return
        
        python_files = list(path.glob('*.py'))
        
        if not python_files:
            logger.error(f"No Python files found in {path}")
            return
        
        # Use the first Python file found
        self._run_python(python_files[0], *args)
    
    def load_dataset(self, dataset: str, config: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load dataset with optional configuration"""
        if dataset.lower() in ('', 'h', 'help'):
            print('Available datasets:')
            print('=' * 50)
            for ds in self.get_datasets():
                if ds not in {'Equity-Evaluation-Corpus', 'RealToxicityPrompts'}:
                    print(f"  {ds}")
            return None
        
        if dataset in self.unavailable_datasets:
            logger.error(f'Dataset {dataset} is not available')
            return None
        
        if dataset not in self.handlers:
            logger.error(f'Dataset {dataset} not found')
            return None
        
        if dataset in self.config_required and config is None:
            print(f'Available configurations for {dataset}:')
            print('=' * 50)
            for conf in self.configs.get(dataset, []):
                print(f"  {conf}")
            return None
        
        try:
            return self.handlers[dataset](config or '')
        except Exception as e:
            logger.error(f"Error loading dataset {dataset}: {e}")
            return None


class CustomDataset(PtDataset):
    """Custom PyTorch Dataset wrapper"""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
    
    def __getitem__(self, index: int) -> Tuple[np.ndarray, Any]:
        row = self.dataframe.iloc[index]
        features = row.iloc[1:].to_numpy()
        label = row.iloc[0]
        return features, label
    
    def __len__(self) -> int:
        return len(self.dataframe)


def BiasDataLoader(
    dataset: Optional[str] = None,
    config: Optional[str] = None,
    format: str = 'hf',
    benchmark_path: Optional[str] = None
) -> Optional[Dict[str, Union[pd.DataFrame, List[str], PtDataset, HfDataset]]]:
    """
    Load specified bias evaluation dataset
    
    Args:
        dataset: Name of the dataset to load
        config: Dataset configuration if applicable
        format: Output format - 'raw', 'hf' (HuggingFace), or 'pt' (PyTorch)
        benchmark_path: Path to Fair-LLM-Benchmark directory
    
    Returns:
        Dictionary with datasets in the appropriate format
    """
    
    loader = FairLLMBenchmarkLoader(benchmark_path)
    
    if not dataset:
        return loader.load_dataset('')
    
    # Load raw data
    raw_data = loader.load_dataset(dataset, config)
    
    if raw_data is None:
        return None
    
    # Convert to requested format
    if format == 'raw':
        return raw_data if isinstance(raw_data, dict) else {'data': raw_data}
    
    data_dict = {}
    
    if format == 'hf':
        if isinstance(raw_data, dict):
            for key, value in raw_data.items():
                if isinstance(value, pd.DataFrame):
                    data_dict[key] = HfDataset.from_pandas(value)
                else:
                    logger.warning(f"Skipping {key}: not a DataFrame")
        elif isinstance(raw_data, pd.DataFrame):
            data_dict['data'] = HfDataset.from_pandas(raw_data)
        else:
            raise TypeError("Data must be a pandas DataFrame or dict of DataFrames for HF format")
    
    elif format == 'pt':
        if isinstance(raw_data, dict):
            for key, value in raw_data.items():
                if isinstance(value, pd.DataFrame):
                    data_dict[key] = CustomDataset(value)
                else:
                    logger.warning(f"Skipping {key}: not a DataFrame")
        elif isinstance(raw_data, pd.DataFrame):
            data_dict['data'] = CustomDataset(raw_data)
        else:
            raise TypeError("Data must be a pandas DataFrame or dict of DataFrames for PyTorch format")
    
    else:
        raise ValueError('Supported formats: "hf", "pt", "raw"')
    
    return data_dict