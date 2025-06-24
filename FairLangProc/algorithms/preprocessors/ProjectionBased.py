# Standard libraries
import logging, pickle, json, torch
from pathlib import Path
from typing import TypeVar, Optional, Union, Dict, Any, List, Tuple
from abc import abstractmethod, ABC
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, 
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    AutoConfig
)

# THIS CODE NEEDS COMPLETION

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type hints
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizer)

@dataclass
class BiasEvaluationResult:
    """Structured result for bias evaluation."""
    mean_bias_score: float
    std_bias_score: float
    max_bias_score: float
    min_bias_score: float
    median_bias_score: float
    percentile_95: float
    explained_variance_ratio: Optional[List[float]] = None
    individual_scores: Optional[List[float]] = None
    sentences: Optional[List[str]] = None


@dataclass
class DebiasConfig:
    """Configuration for debiasing parameters."""
    word_pairs: Optional[List[Tuple[str, str]]] = None
    n_components: int = 1
    pooling_strategy: str = 'mean'
    bias_subspace_path: Optional[str] = None
    neutralize_strength: float = 1.0
    use_regularization: bool = False
    regularization_weight: float = 0.01
    batch_size: int = 32
    max_length: int = 512
    custom_word_pairs_path: Optional[str] = None
    bias_direction_method: str = 'pca'  # 'pca', 'two_means', 'classification'
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.pooling_strategy not in ['mean', 'cls', 'max', 'weighted']:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")
        if self.bias_direction_method not in ['pca', 'two_means', 'classification']:
            raise ValueError(f"Invalid bias direction method: {self.bias_direction_method}")
        if not 0 < self.neutralize_strength <= 1:
            raise ValueError("neutralize_strength must be between 0 and 1")


class BiasSubspaceComputer:
    """Separate class for computing bias subspaces with different methods."""
    
    def __init__(self, config: DebiasConfig, device: torch.device):
        self.config = config
        self.device = device
    
    def compute_pca_subspace(self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """Compute bias subspace using PCA on embedding differences."""
        diffs = (embeddings_1 - embeddings_2).cpu().numpy()
        
        # Handle case where we have fewer samples than components
        n_components = min(self.config.n_components, diffs.shape[0], diffs.shape[1])
        if n_components != self.config.n_components:
            logger.warning(f"Adjusting n_components from {self.config.n_components} to {n_components}")
        
        pca = PCA(n_components=n_components)
        pca.fit(diffs)
        
        bias_subspace = torch.tensor(pca.components_.T, dtype=torch.float32).to(self.device)
        return bias_subspace, pca.explained_variance_ratio_
    
    def compute_two_means_subspace(self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """Compute bias direction as difference of means."""
        mean_1 = embeddings_1.mean(dim=0)
        mean_2 = embeddings_2.mean(dim=0)
        bias_direction = (mean_1 - mean_2).unsqueeze(1)  # Shape: [embedding_dim, 1]
        
        # Normalize
        bias_direction = F.normalize(bias_direction, dim=0)
        
        return bias_direction, np.array([1.0])  # Single direction explains all variance
    
    def compute_classification_subspace(self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """Compute bias subspace using a linear classifier approach."""
        # Combine embeddings and create labels
        all_embeddings = torch.cat([embeddings_1, embeddings_2], dim=0)
        labels = torch.cat([
            torch.zeros(embeddings_1.size(0)),
            torch.ones(embeddings_2.size(0))
        ]).to(self.device)
        
        # Train a simple linear classifier
        classifier = nn.Linear(all_embeddings.size(1), 2).to(self.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
        
        for _ in range(100):  # Quick training
            logits = classifier(all_embeddings)
            loss = F.cross_entropy(logits, labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Use classifier weights as bias direction
        bias_direction = classifier.weight[1] - classifier.weight[0]  # Difference between class weights
        bias_direction = bias_direction.unsqueeze(1)  # Shape: [embedding_dim, 1]
        bias_direction = F.normalize(bias_direction, dim=0)
        
        return bias_direction, np.array([1.0])


class EmbeddingPooler:
    """Enhanced embedding pooling with multiple strategies."""
    
    @staticmethod
    def pool_embeddings(hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor], 
                       strategy: str = 'mean') -> torch.Tensor:
        """Pool token embeddings using specified strategy."""
        if strategy == 'cls':
            return hidden_states[:, 0]
        
        elif strategy == 'mean':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return hidden_states.mean(dim=1)
        
        elif strategy == 'max':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                hidden_states_masked = hidden_states.clone()
                hidden_states_masked[mask_expanded == 0] = -1e9
                return torch.max(hidden_states_masked, 1)[0]
            else:
                return torch.max(hidden_states, 1)[0]
        
        elif strategy == 'weighted':
            # Attention-weighted pooling
            if attention_mask is not None:
                # Simple attention mechanism
                attention_weights = F.softmax(attention_mask.float(), dim=1)
                weighted_sum = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
                return weighted_sum
            else:
                return hidden_states.mean(dim=1)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")


class SentDebiasModel(nn.Module, ABC):
    """
    Enhanced base class for sentence debiasing models using projection-based bias removal.
    
    This implementation provides multiple bias computation methods, regularization options,
    and comprehensive bias evaluation capabilities.
    """

    def __init__(
        self,
        model: Union[nn.Module, str],
        config: Optional[Union[DebiasConfig, Dict[str, Any]]] = None,
        tokenizer: Optional[TokenizerType] = None,
        device: Optional[str] = None,
        model_config: Optional[str] = None
    ):
        super().__init__()
        
        # Handle config
        if config is None:
            config = DebiasConfig()
        elif isinstance(config, dict):
            config = DebiasConfig(**config)
        self.config = config
        
        # Initialize model and tokenizer
        self._setup_model_and_tokenizer(model, model_config, tokenizer)
        
        # Device setup
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        
        # Initialize components
        self.pooler = EmbeddingPooler()
        self.bias_computer = BiasSubspaceComputer(self.config, self.device)
        
        # Load word pairs
        self.word_pairs = self._load_word_pairs()
        
        # Check if model has classification head
        self.has_head = self._check_classification_head()
        
        # Initialize bias subspace and other components
        self.bias_subspace = None
        self.explained_variance_ratio = None
        self._load_or_compute_bias_subspace()
        
        # Initialize loss function
        self._setup_loss()
        
        logger.info(f"Initialized {self.__class__.__name__} with device: {self.device}")

    def _setup_model_and_tokenizer(self, model, model_config, tokenizer):
        """Initialize model and tokenizer with enhanced error handling."""
        if isinstance(model, nn.Module):
            self.model = model
            self.model_name = getattr(model, 'name_or_path', 'custom_model')
            if tokenizer is None:
                raise ValueError("Tokenizer is required when using a custom model.")
            self.tokenizer = tokenizer
        elif isinstance(model, str):
            self.model_name = model
            try:
                # Load model config first if provided
                config_obj = None
                if model_config:
                    config_obj = AutoConfig.from_pretrained(model_config)
                
                self.model = self._load_model(model, config_obj)
                
                if tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(model)
                    # Ensure pad token exists
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer = tokenizer
            except Exception as e:
                logger.error(f"Failed to load model {model}: {e}")
                raise
        else:
            raise TypeError(f"Model must be nn.Module or str, got {type(model)}")

    def _load_word_pairs(self) -> List[Tuple[str, str]]:
        """Load word pairs from config or file."""
        if self.config.word_pairs:
            return self.config.word_pairs
        elif self.config.custom_word_pairs_path:
            try:
                return self._load_word_pairs_from_file(self.config.custom_word_pairs_path)
            except Exception as e:
                logger.warning(f"Failed to load custom word pairs: {e}. Using defaults.")
                return self._get_default_word_pairs()
        else:
            return self._get_default_word_pairs()

    def _load_word_pairs_from_file(self, file_path: str) -> List[Tuple[str, str]]:
        """Load word pairs from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            pairs = json.load(f)
        return [(pair[0], pair[1]) for pair in pairs]

    def _get_default_word_pairs(self) -> List[Tuple[str, str]]:
        """Extended default gender word pairs for bias computation."""
        return [
            # Basic pronouns and family
            ("man", "woman"), ("boy", "girl"), ("he", "she"), ("him", "her"),
            ("his", "hers"), ("son", "daughter"), ("father", "mother"),
            ("uncle", "aunt"), ("brother", "sister"), ("grandfather", "grandmother"),
            
            # Titles and roles
            ("king", "queen"), ("prince", "princess"), ("duke", "duchess"), 
            ("lord", "lady"), ("sir", "madam"), ("gentleman", "lady"),
            ("emperor", "empress"), ("baron", "baroness"),
            
            # Occupations
            ("actor", "actress"), ("waiter", "waitress"), ("hero", "heroine"),
            ("god", "goddess"), ("host", "hostess"), ("steward", "stewardess"),
            ("chairman", "chairwoman"), ("spokesman", "spokeswoman"),
            ("businessman", "businesswoman"), ("policeman", "policewoman"),
            
            # Relationships
            ("husband", "wife"), ("boyfriend", "girlfriend"), ("fiance", "fiancee"),
            ("bachelor", "bachelorette"), ("monk", "nun"),
            
            # Adjectives and descriptors
            ("male", "female"), ("masculine", "feminine"), ("manly", "womanly"),
            ("handsome", "beautiful"), ("strong", "pretty")
        ]

    def _check_classification_head(self) -> bool:
        """Check if model has a classification head."""
        return (hasattr(self.model, 'classifier') or 
                hasattr(self.model, 'head') or 
                hasattr(self.model, 'score'))

    @abstractmethod
    def _get_embedding(self, **inputs) -> torch.Tensor:
        """Extract embeddings from model inputs."""
        pass
    
    @abstractmethod
    def _setup_loss(self):
        """Initialize loss function."""
        pass
    
    @abstractmethod
    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for the specific task."""
        pass
    
    @abstractmethod
    def _load_model(self, model_name: str, config: Optional[Any]) -> PreTrainedModel:
        """Load the appropriate model for the task."""
        pass

    def _load_or_compute_bias_subspace(self):
        """Load pre-computed bias subspace or compute it."""
        if self.config.bias_subspace_path and self._load_bias_subspace():
            logger.info(f"Loaded bias subspace from {self.config.bias_subspace_path}")
            return
        
        logger.info("Computing bias subspace...")
        self._compute_bias_subspace()
        
        if self.config.bias_subspace_path:
            self._save_bias_subspace()

    def _compute_bias_subspace(self):
        """Compute bias subspace using the specified method."""
        if not self.word_pairs:
            raise ValueError("Word pairs are required for bias subspace computation")

        try:
            # Tokenize word pairs
            male_words = [pair[0] for pair in self.word_pairs]
            female_words = [pair[1] for pair in self.word_pairs]
            
            # Process in batches to handle memory constraints
            batch_size = min(self.config.batch_size, len(male_words))
            all_male_embeddings = []
            all_female_embeddings = []
            
            for i in range(0, len(male_words), batch_size):
                batch_male = male_words[i:i + batch_size]
                batch_female = female_words[i:i + batch_size]
                
                male_tokens = self.tokenizer(
                    batch_male, return_tensors="pt", padding=True, 
                    truncation=True, max_length=self.config.max_length
                ).to(self.device)
                female_tokens = self.tokenizer(
                    batch_female, return_tensors="pt", padding=True, 
                    truncation=True, max_length=self.config.max_length
                ).to(self.device)

                # Get embeddings
                with torch.no_grad():
                    male_embeddings = self._get_embedding(**male_tokens)
                    female_embeddings = self._get_embedding(**female_tokens)
                
                all_male_embeddings.append(male_embeddings)
                all_female_embeddings.append(female_embeddings)
            
            # Concatenate all embeddings
            male_embeddings = torch.cat(all_male_embeddings, dim=0)
            female_embeddings = torch.cat(all_female_embeddings, dim=0)
            
            # Compute bias subspace using the specified method
            if self.config.bias_direction_method == 'pca':
                bias_subspace, explained_var = self.bias_computer.compute_pca_subspace(
                    male_embeddings, female_embeddings
                )
            elif self.config.bias_direction_method == 'two_means':
                bias_subspace, explained_var = self.bias_computer.compute_two_means_subspace(
                    male_embeddings, female_embeddings
                )
            elif self.config.bias_direction_method == 'classification':
                bias_subspace, explained_var = self.bias_computer.compute_classification_subspace(
                    male_embeddings, female_embeddings
                )
            else:
                raise ValueError(f"Unknown bias direction method: {self.config.bias_direction_method}")
            
            self.bias_subspace = bias_subspace
            self.explained_variance_ratio = explained_var
            
            logger.info(f"Computed bias subspace: {bias_subspace.shape}")
            logger.info(f"Method: {self.config.bias_direction_method}")
            logger.info(f"Explained variance ratio: {explained_var}")
            
        except Exception as e:
            logger.error(f"Error computing bias subspace: {e}")
            raise

    def _save_bias_subspace(self):
        """Save bias subspace to disk."""
        if not self.config.bias_subspace_path:
            return
            
        try:
            # Ensure directory exists
            Path(self.config.bias_subspace_path).parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'bias_subspace': self.bias_subspace.cpu().numpy(),
                'explained_variance_ratio': self.explained_variance_ratio,
                'word_pairs': self.word_pairs,
                'config': {
                    'n_components': self.config.n_components,
                    'bias_direction_method': self.config.bias_direction_method,
                    'pooling_strategy': self.config.pooling_strategy
                },
                'model_name': self.model_name
            }
            with open(self.config.bias_subspace_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved bias subspace to {self.config.bias_subspace_path}")
        except Exception as e:
            logger.error(f"Failed to save bias subspace: {e}")

    def _load_bias_subspace(self) -> bool:
        """Load bias subspace from disk."""
        try:
            with open(self.config.bias_subspace_path, 'rb') as f:
                data = pickle.load(f)
            
            self.bias_subspace = torch.tensor(data['bias_subspace'], dtype=torch.float32).to(self.device)
            self.explained_variance_ratio = data.get('explained_variance_ratio')
            
            # Validate compatibility
            if data.get('model_name') != self.model_name:
                logger.warning(f"Loaded bias subspace was computed for {data.get('model_name')}, "
                             f"but current model is {self.model_name}")
            
            return True
        except FileNotFoundError:
            logger.info(f"Bias subspace file not found: {self.config.bias_subspace_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load bias subspace: {e}")
            return False

    def _neutralize(self, embeddings: torch.Tensor, strength: Optional[float] = None) -> torch.Tensor:
        """Project embeddings onto bias-free subspace with adjustable strength."""
        if self.bias_subspace is None:
            logger.warning("Bias subspace not initialized, returning original embeddings")
            return embeddings
        
        strength = strength or self.config.neutralize_strength
        
        # Compute projection coefficients
        proj_coeff = torch.matmul(embeddings, self.bias_subspace)
        
        # Compute projection onto bias subspace
        proj = torch.matmul(proj_coeff, self.bias_subspace.T)
        
        # Remove bias component with adjustable strength
        debiased = embeddings - strength * proj
        
        return debiased

    def get_bias_projection(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get the bias component of embeddings (for analysis)."""
        if self.bias_subspace is None:
            return torch.zeros_like(embeddings)
        
        proj_coeff = torch.matmul(embeddings, self.bias_subspace)
        proj = torch.matmul(proj_coeff, self.bias_subspace.T)
        return proj

    def compute_bias_score(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute bias score as magnitude of projection onto bias subspace."""
        bias_proj = self.get_bias_projection(embeddings)
        return torch.norm(bias_proj, dim=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                token_type_ids: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None,
                return_bias_info: bool = False, neutralize_strength: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with debiasing and optional regularization."""
        # Get embeddings
        embeddings = self._get_embedding(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        
        # Debias embeddings
        debiased_embeddings = self._neutralize(embeddings, neutralize_strength)
        
        outputs = {'debiased_embeddings': debiased_embeddings}
        
        # Add bias information if requested
        if return_bias_info:
            bias_scores = self.compute_bias_score(embeddings)
            bias_projection = self.get_bias_projection(embeddings)
            outputs.update({
                'bias_scores': bias_scores,
                'bias_projection': bias_projection,
                'original_embeddings': embeddings
            })
        
        # Classification head
        if self.has_head:
            logits = self._get_logits(debiased_embeddings)
            outputs['logits'] = logits
            
            if labels is not None:
                loss = self._loss(logits, labels)
                
                # Add regularization if enabled
                if self.config.use_regularization:
                    reg_loss = self._compute_regularization_loss(embeddings, debiased_embeddings)
                    loss = loss + self.config.regularization_weight * reg_loss
                    outputs['regularization_loss'] = reg_loss
                
                outputs['loss'] = loss
        
        return outputs

    def _get_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get logits from embeddings using the appropriate head."""
        if hasattr(self.model, "classifier"):
            return self.model.classifier(embeddings)
        elif hasattr(self.model, "head"):
            return self.model.head(embeddings)
        elif hasattr(self.model, "score"):
            return self.model.score(embeddings)
        else:
            raise AttributeError("Model has no classifier, head, or score layer")

    def _compute_regularization_loss(self, original_embeddings: torch.Tensor, 
                                   debiased_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss to preserve semantic information."""
        # L2 distance between original and debiased embeddings
        return F.mse_loss(debiased_embeddings, original_embeddings)

    def evaluate_bias(self, sentences: List[str], 
                     batch_size: Optional[int] = None,
                     return_individual_scores: bool = False) -> BiasEvaluationResult:
        """Enhanced bias evaluation with comprehensive metrics."""
        batch_size = batch_size or self.config.batch_size
        all_bias_scores = []
        
        self.eval()
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_sentences, return_tensors="pt", padding=True, 
                    truncation=True, max_length=self.config.max_length
                ).to(self.device)
                
                outputs = self.forward(**inputs, return_bias_info=True)
                bias_scores = outputs['bias_scores'].cpu().numpy()
                all_bias_scores.extend(bias_scores.tolist())
        
        bias_scores_array = np.array(all_bias_scores)
        
        result = BiasEvaluationResult(
            mean_bias_score=float(np.mean(bias_scores_array)),
            std_bias_score=float(np.std(bias_scores_array)),
            max_bias_score=float(np.max(bias_scores_array)),
            min_bias_score=float(np.min(bias_scores_array)),
            median_bias_score=float(np.median(bias_scores_array)),
            percentile_95=float(np.percentile(bias_scores_array, 95)),
            explained_variance_ratio=self.explained_variance_ratio.tolist() if self.explained_variance_ratio is not None else None
        )
        
        if return_individual_scores:
            result.individual_scores = all_bias_scores
            result.sentences = sentences
        
        return result

    def save_model(self, save_path: str, save_tokenizer: bool = True):
        """Save the complete model with enhanced metadata."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_data = {
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'config': self.config,
            'bias_subspace': self.bias_subspace.cpu() if self.bias_subspace is not None else None,
            'explained_variance_ratio': self.explained_variance_ratio,
            'word_pairs': self.word_pairs
        }
        
        torch.save(model_data, save_path / 'model.pt')
        
        # Save tokenizer
        if save_tokenizer:
            self.tokenizer.save_pretrained(save_path / 'tokenizer')
        
        # Save config as JSON for human readability
        with open(save_path / 'config.json', 'w') as f:
            config_dict = {
                'model_name': self.model_name,
                'n_components': self.config.n_components,
                'pooling_strategy': self.config.pooling_strategy,
                'bias_direction_method': self.config.bias_direction_method,
                'neutralize_strength': self.config.neutralize_strength,
                'use_regularization': self.config.use_regularization,
                'regularization_weight': self.config.regularization_weight
            }
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        """Load the complete model with validation."""
        load_path = Path(load_path)
        
        # Load model data
        model_data = torch.load(load_path / 'model.pt', map_location=self.device)
        
        # Load state dict
        self.load_state_dict(model_data['model_state_dict'])
        
        # Restore configuration
        if isinstance(model_data['config'], DebiasConfig):
            self.config = model_data['config']
        else:
            # Handle legacy configs
            self.config.word_pairs = model_data.get('word_pairs', self.config.word_pairs)
            self.config.n_components = model_data.get('n_components', self.config.n_components)
            self.config.pooling_strategy = model_data.get('pooling_strategy', self.config.pooling_strategy)
        
        # Restore bias subspace
        if model_data['bias_subspace'] is not None:
            self.bias_subspace = model_data['bias_subspace'].to(self.device)
        
        self.explained_variance_ratio = model_data.get('explained_variance_ratio')
        self.word_pairs = model_data.get('word_pairs', self.word_pairs)
        
        logger.info(f"Model loaded from {load_path}")


class SentDebiasForSequenceClassification(SentDebiasModel):
    """
    Enhanced SentDebias implementation for sequence classification tasks.
    """

    def _setup_loss(self):
        """Initialize loss function with label smoothing option."""
        self.loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1 if self.config.use_regularization else 0.0)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute classification loss."""
        return self.loss_fct(logits, labels)

    def _load_model(self, model_name: str, config: Optional[Any]) -> PreTrainedModel:
        """Load AutoModelForSequenceClassification."""
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config if config else None
        )

    def _get_embedding(self, **inputs) -> torch.Tensor:
        """Extract pooled embeddings for classification."""
        # Handle different model architectures
        if hasattr(self.model, 'bert'):
            outputs = self.model.bert(**inputs)
        elif hasattr(self.model, 'roberta'):
            outputs = self.model.roberta(**inputs)
        elif hasattr(self.model, 'distilbert'):
            outputs = self.model.distilbert(**inputs)
        elif hasattr(self.model, 'electra'):
            outputs = self.model.electra(**inputs)
        else:
            # Fallback: try to get hidden states from base model
            base_model = getattr(self.model, self.model.base_model_prefix, self.model)
            outputs = base_model(**inputs)
        
        hidden_states = outputs.last_hidden_state
        
        # Pool embeddings
        attention_mask = inputs.get('attention_mask')
        pooled = self.pooler.pool_embeddings(hidden_states, attention_mask, self.config.pooling_strategy)
            
        return pooled

    def predict(self, sentences: List[str], batch_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Make predictions on sentences with bias scores."""
        batch_size = batch_size or self.config.batch_size
        all_predictions = []
        all_probabilities = []
        all_bias_scores = []
        
        self.eval()
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_sentences, return_tensors="pt", padding=True, 
                    truncation=True, max_length=self.config.max_length
                ).to(self.device)
                
                outputs = self.forward(**inputs, return_bias_info=True)
                
                logits = outputs['logits']
                probabilities = F.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                bias_scores = outputs['bias_scores']
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())
                all_bias_scores.extend(bias_scores.cpu().numpy())
        
        return {
            'predictions': np.array(all_predictions),
            'probabilities': np.vstack(all_probabilities),
            'bias_scores': np.array(all_bias_scores)
        }

    def evaluate_classification(self, sentences: List[str], labels: List[int], 
                              batch_size: Optional[int] = None) -> Dict[str, float]:
        """Evaluate classification performance."""
        predictions = self.predict(sentences, batch_size)
        
        accuracy = accuracy_score(labels, predictions['predictions'])
        f1 = f1_score(labels, predictions['predictions'], average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'mean_bias_score': float(np.mean(predictions['bias_scores'])),
            'std_bias_score': float(np.std(predictions['bias_scores']))
        }


class SentDebiasForEmbedding(SentDebiasModel):
    """
    Enhanced SentDebias implementation for general embedding extraction.
    """

    def _setup_loss(self):
        """Initialize dummy loss (not used for embedding models)."""
        self.loss_fct = nn.MSELoss()

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Dummy loss implementation."""
        return torch.tensor(0.0, device=logits.device)

    def _load_model(self, model_name: str, config: Optional[Any]) -> PreTrainedModel:
        """Load AutoModel for embedding extraction."""
        return AutoModel.from_pretrained(
            model_name, config=config if config else None
        )

    def _get_embedding(self, **inputs) -> torch.Tensor:
        """Extract embeddings from the model."""
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state
        
        # Pool embeddings
        attention_mask = inputs.get('attention_mask')
        pooled = self.pooler.pool_embeddings(hidden_states, attention_mask, self.config.pooling_strategy)
            
        return pooled

    def encode(self, sentences: List[str], batch_size: Optional[int] = None, 
               normalize_embeddings: bool = False, return_bias_info: bool = False) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Encode sentences to debiased embeddings with additional options."""
        batch_size = batch_size or self.config.batch_size
        all_embeddings = []
        all_bias_scores = []
        all_original_embeddings = []
        
        self.eval()
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_sentences, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=self.config.max_length
                ).to(self.device)
                
                outputs = self.forward(**inputs, return_bias_info=return_bias_info)
                embeddings = outputs['debiased_embeddings']
                
                if normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
                
                if return_bias_info:
                    all_bias_scores.append(outputs['bias_scores'].cpu().numpy())
                    all_original_embeddings.append(outputs['original_embeddings'].cpu().numpy())
        
        debiased_embeddings = np.vstack(all_embeddings)
        
        if return_bias_info:
            result = {
                'embeddings': debiased_embeddings,
                'bias_scores': np.concatenate(all_bias_scores),
                'original_embeddings': np.vstack(all_original_embeddings)
            }
            return result
        
        return debiased_embeddings

    def similarity(self, sentences1: List[str], sentences2: List[str], 
                   batch_size: Optional[int] = None) -> np.ndarray:
        """Compute cosine similarity between sentence pairs."""
        emb1 = self.encode(sentences1, batch_size, normalize_embeddings=True)
        emb2 = self.encode(sentences2, batch_size, normalize_embeddings=True)
        
        # Compute cosine similarity
        similarities = np.sum(emb1 * emb2, axis=1)
        return similarities

    def find_similar(self, query: str, candidates: List[str], 
                     top_k: int = 5, batch_size: Optional[int] = None) -> List[Tuple[str, float]]:
        """Find most similar sentences to a query."""
        query_emb = self.encode([query], batch_size, normalize_embeddings=True)[0]
        candidate_embs = self.encode(candidates, batch_size, normalize_embeddings=True)
        
        # Compute similarities
        similarities = np.dot(candidate_embs, query_emb)
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = [(candidates[i], float(similarities[i])) for i in top_indices]
        return results


class SentDebiasForRegression(SentDebiasModel):
    """
    SentDebias implementation for regression tasks.
    """

    def _setup_loss(self):
        """Initialize MSE loss for regression."""
        self.loss_fct = nn.MSELoss()

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute regression loss."""
        return self.loss_fct(logits.squeeze(), labels.float())

    def _load_model(self, model_name: str, config: Optional[Any]) -> PreTrainedModel:
        """Load model for regression (modify final layer)."""
        model = AutoModel.from_pretrained(model_name, config=config if config else None)
        
        # Add regression head
        if hasattr(model, 'config'):
            hidden_size = model.config.hidden_size
        else:
            # Fallback: try to infer from model
            hidden_size = 768  # Common default
            
        model.regressor = nn.Linear(hidden_size, 1)
        return model

    def _get_embedding(self, **inputs) -> torch.Tensor:
        """Extract embeddings for regression."""
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state
        
        # Pool embeddings
        attention_mask = inputs.get('attention_mask')
        pooled = self.pooler.pool_embeddings(hidden_states, attention_mask, self.config.pooling_strategy)
        
        return pooled

    def _get_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get regression output."""
        return self.model.regressor(embeddings)

    def predict(self, sentences: List[str], batch_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Make regression predictions."""
        batch_size = batch_size or self.config.batch_size
        all_predictions = []
        all_bias_scores = []
        
        self.eval()
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch_sentences = sentences[i:i + batch_size]
                inputs = self.tokenizer(
                    batch_sentences, return_tensors="pt", padding=True, 
                    truncation=True, max_length=self.config.max_length
                ).to(self.device)
                
                outputs = self.forward(**inputs, return_bias_info=True)
                
                predictions = outputs['logits'].squeeze()
                bias_scores = outputs['bias_scores']
                
                all_predictions.extend(predictions.cpu().numpy())
                all_bias_scores.extend(bias_scores.cpu().numpy())
        
        return {
            'predictions': np.array(all_predictions),
            'bias_scores': np.array(all_bias_scores)
        }


# Utility classes and functions

class BiasEvaluator:
    """Comprehensive bias evaluation toolkit."""
    
    @staticmethod
    def evaluate_word_embedding_association_test(model: SentDebiasModel, 
                                                target_words: List[str],
                                                attribute_words_1: List[str],
                                                attribute_words_2: List[str]) -> float:
        """Perform Word Embedding Association Test (WEAT)."""
        # Get embeddings for all words
        all_words = target_words + attribute_words_1 + attribute_words_2
        embeddings_dict = {}
        
        for word in all_words:
            emb = model.encode([word])[0]
            embeddings_dict[word] = emb
        
        # Compute association scores
        target_embs = [embeddings_dict[word] for word in target_words]
        attr1_embs = [embeddings_dict[word] for word in attribute_words_1]
        attr2_embs = [embeddings_dict[word] for word in attribute_words_2]
        
        # Calculate mean embeddings for attribute sets
        attr1_mean = np.mean(attr1_embs, axis=0)
        attr2_mean = np.mean(attr2_embs, axis=0)
        
        # Calculate association scores
        association_scores = []
        for target_emb in target_embs:
            sim1 = np.dot(target_emb, attr1_mean) / (np.linalg.norm(target_emb) * np.linalg.norm(attr1_mean))
            sim2 = np.dot(target_emb, attr2_mean) / (np.linalg.norm(target_emb) * np.linalg.norm(attr2_mean))
            association_scores.append(sim1 - sim2)
        
        # Return effect size
        return float(np.mean(association_scores) / np.std(association_scores))
    
    @staticmethod
    def evaluate_sentence_templates(model: SentDebiasModel, 
                                   templates: List[str],
                                   male_words: List[str],
                                   female_words: List[str]) -> Dict[str, float]:
        """Evaluate bias using sentence templates."""
        male_sentences = []
        female_sentences = []
        
        for template in templates:
            for male_word in male_words:
                male_sentences.append(template.format(male_word))
            for female_word in female_words:
                female_sentences.append(template.format(female_word))
        
        # Get embeddings
        male_embs = model.encode(male_sentences)
        female_embs = model.encode(female_sentences)
        
        # Calculate bias metrics
        male_mean = np.mean(male_embs, axis=0)
        female_mean = np.mean(female_embs, axis=0)
        
        bias_direction = male_mean - female_mean
        bias_magnitude = np.linalg.norm(bias_direction)
        
        # Calculate individual biases
        male_biases = [np.dot(emb, bias_direction) for emb in male_embs]
        female_biases = [np.dot(emb, bias_direction) for emb in female_embs]
        
        return {
            'bias_magnitude': float(bias_magnitude),
            'male_bias_mean': float(np.mean(male_biases)),
            'female_bias_mean': float(np.mean(female_biases)),
            'bias_difference': float(np.mean(male_biases) - np.mean(female_biases))
        }


def create_debias_model(model_name: str, 
                       task_type: str = 'embedding',
                       config: Optional[Union[Dict[str, Any], DebiasConfig]] = None,
                       **kwargs) -> SentDebiasModel:
    """Factory function to create appropriate debiasing model."""
    
    if isinstance(config, dict):
        config = DebiasConfig(**config)
    elif config is None:
        config = DebiasConfig()
    
    task_type = task_type.lower()
    
    if task_type in ['embedding', 'encode']:
        return SentDebiasForEmbedding(model_name, config=config, **kwargs)
    elif task_type in ['classification', 'classify']:
        return SentDebiasForSequenceClassification(model_name, config=config, **kwargs)
    elif task_type in ['regression', 'regress']:
        return SentDebiasForRegression(model_name, config=config, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}. "
                        f"Supported types: 'embedding', 'classification', 'regression'")


def load_word_pairs_from_file(file_path: str) -> List[Tuple[str, str]]:
    """Load word pairs from JSON file with enhanced error handling."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            if all(isinstance(pair, list) and len(pair) == 2 for pair in data):
                return [(pair[0], pair[1]) for pair in data]
            else:
                raise ValueError("Word pairs must be lists of length 2")
        elif isinstance(data, dict):
            if 'word_pairs' in data:
                pairs = data['word_pairs']
                return [(pair[0], pair[1]) for pair in pairs]
            else:
                # Assume dict keys are male words, values are female words
                return [(k, v) for k, v in data.items()]
        else:
            raise ValueError("Invalid file format")
    
    except Exception as e:
        logger.error(f"Failed to load word pairs from {file_path}: {e}")
        raise


def evaluate_model_bias(model: SentDebiasModel, 
                       test_sentences: List[str],
                       evaluation_type: str = 'comprehensive',
                       save_results: bool = False, 
                       results_path: Optional[str] = None) -> Dict[str, Any]:
    """Comprehensive bias evaluation with multiple metrics."""
    
    results = {}
    
    # Basic bias evaluation
    bias_result = model.evaluate_bias(test_sentences, return_individual_scores=True)
    results['basic_metrics'] = {
        'mean_bias_score': bias_result.mean_bias_score,
        'std_bias_score': bias_result.std_bias_score,
        'max_bias_score': bias_result.max_bias_score,
        'min_bias_score': bias_result.min_bias_score,
        'median_bias_score': bias_result.median_bias_score,
        'percentile_95': bias_result.percentile_95
    }
    
    if evaluation_type == 'comprehensive':
        # Template-based evaluation
        evaluator = BiasEvaluator()
        
        # Example profession templates
        profession_templates = [
            "The {} is very professional.",
            "I met a {} yesterday.",
            "The {} was working hard.",
            "Every {} deserves respect."
        ]
        
        male_words = ["man", "he", "guy", "male"]
        female_words = ["woman", "she", "girl", "female"]
        
        template_results = evaluator.evaluate_sentence_templates(
            model, profession_templates, male_words, female_words
        )
        results['template_evaluation'] = template_results
        
        # WEAT-style evaluation
        try:
            career_words = ["executive", "management", "professional", "corporation", "salary", "office"]
            family_words = ["home", "parents", "children", "family", "cousins", "marriage"]
            
            weat_score = evaluator.evaluate_word_embedding_association_test(
                model, male_words + female_words, career_words, family_words
            )
            results['weat_score'] = weat_score
        except Exception as e:
            logger.warning(f"WEAT evaluation failed: {e}")
            results['weat_score'] = None
    
    # Add model information
    results['model_info'] = {
        'model_name': model.model_name,
        'n_components': model.config.n_components,
        'pooling_strategy': model.config.pooling_strategy,
        'bias_direction_method': model.config.bias_direction_method,
        'neutralize_strength': model.config.neutralize_strength
    }
    
    if save_results and results_path:
        # Ensure directory exists
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Bias evaluation results saved to {results_path}")
    
    return results


def compare_models(models: List[SentDebiasModel], 
                  test_sentences: List[str],
                  model_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """Compare bias metrics across multiple models."""
    
    if model_names is None:
        model_names = [f"model_{i}" for i in range(len(models))]
    
    comparison_results = {}
    
    for model, name in zip(models, model_names):
        try:
            bias_result = model.evaluate_bias(test_sentences)
            comparison_results[name] = {
                'mean_bias_score': bias_result.mean_bias_score,
                'std_bias_score': bias_result.std_bias_score,
                'max_bias_score': bias_result.max_bias_score,
                'median_bias_score': bias_result.median_bias_score
            }
        except Exception as e:
            logger.error(f"Failed to evaluate model {name}: {e}")
            comparison_results[name] = {'error': str(e)}
    
    return comparison_results