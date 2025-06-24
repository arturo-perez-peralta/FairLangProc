import logging
from typing import Optional, Dict, Union, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, PreTrainedModel, AutoConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BLINDModel(nn.Module, ABC):
    """
    Abstract class for implementing BLIND (Bias-aware Learning with Instance-level Noise Detection) debiasing.
    
    BLIND is a debiasing technique that identifies and reweights biased examples during training
    by learning instance-level bias patterns and adjusting the loss accordingly.
    
    Args:
        model (Union[nn.Module, str]): Language model to be debiased or model name/path
        config (Optional[str]): Configuration name or path (used with AutoModel)
        gamma (float): Focal loss gamma parameter controlling reweighting strength
        temperature (float): Temperature parameter for BLIND logits softmax
        hidden_dim (int): Hidden dimension of the language model
        dropout_rate (float): Dropout rate for BLIND head
        bias_head_layers (int): Number of layers in the bias detection head
        freeze_backbone (bool): Whether to freeze the backbone model during training
        device (str): Device to run the model on
    """

    def __init__(
        self,
        model: Union[nn.Module, str],
        config: Optional[str] = None,
        gamma: float = 2.0,
        temperature: float = 1.0,
        hidden_dim: int = 768,
        dropout_rate: float = 0.1,
        bias_head_layers: int = 1,
        freeze_backbone: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.gamma = gamma
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.bias_head_layers = bias_head_layers
        self.freeze_backbone = freeze_backbone
        self.device = device
        
        # Load or set the model
        self.model = self._setup_model(model, config)
        
        # Validate model has classification head
        self._validate_model()
        
        # Create BLIND bias detection head
        self.bias_head = self._create_bias_head()
        
        # Initialize loss function
        self._setup_loss()
        
        # Move to device
        self.to(device)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
            
        logger.info(f"BLIND model initialized with gamma={gamma}, temperature={temperature}")

    def _setup_model(self, model: Union[nn.Module, str], config: Optional[str] = None) -> nn.Module:
        """Setup the base model."""
        if isinstance(model, nn.Module):
            return model
        elif isinstance(model, str):
            return self._load_model(model, config)
        else:
            raise TypeError(f"Model must be nn.Module or str, got {type(model)}")

    def _validate_model(self):
        """Validate that the model has a classification head."""
        self.has_classifier = hasattr(self.model, 'classifier')
        self.has_head = hasattr(self.model, 'head')
        
        if not (self.has_classifier or self.has_head):
            raise AttributeError("Model must have either 'classifier' or 'head' attribute")

    def _create_bias_head(self) -> nn.Module:
        """Create the bias detection head."""
        layers = []
        
        input_dim = self.hidden_dim
        for i in range(self.bias_head_layers):
            if i > 0:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout_rate))
            
            # Last layer outputs 2 classes (biased/unbiased)
            output_dim = self.hidden_dim // 2 if i < self.bias_head_layers - 1 else 2
            layers.append(nn.Linear(input_dim, output_dim))
            input_dim = output_dim
        
        return nn.Sequential(*layers)

    def _freeze_backbone(self):
        """Freeze the backbone model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Backbone model frozen")

    def _get_classifier_logits(self, embedding: torch.Tensor) -> torch.Tensor:
        """Get logits from the classification head."""
        if self.has_classifier:
            return self.model.classifier(embedding)
        elif self.has_head:
            return self.model.head(embedding)
        else:
            raise AttributeError("No classification head found")

    def _compute_blind_weights(self, embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute BLIND reweighting factors."""
        # Get bias detection logits
        bias_logits = self.bias_head(embedding)
        
        # Apply temperature scaling and softmax
        bias_probs = F.softmax(bias_logits / self.temperature, dim=-1)
        
        # Get probability of being biased (class 1)
        bias_prob = bias_probs[:, 1]
        
        # Compute focal loss style weights: (1 - p_bias)^gamma
        weights = torch.pow(1 - bias_prob, self.gamma)
        
        return weights

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        token_type_ids: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the BLIND model.
        
        Returns:
            Dictionary containing:
            - loss: Combined weighted loss (if labels provided)
            - logits: Classification logits
            - bias_logits: Bias detection logits
            - weights: BLIND reweighting factors (if labels provided)
        """
        # Get embeddings
        embedding = self._get_embedding(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        
        # Get classification logits
        logits = self._get_classifier_logits(embedding)
        
        # Get bias detection logits
        bias_logits = self.bias_head(embedding)
        
        outputs = {
            'logits': logits,
            'bias_logits': bias_logits
        }
        
        if labels is not None:
            # Compute main task loss (per-example, no reduction)
            main_loss = self._compute_loss(logits, labels)
            
            # Compute BLIND weights
            weights = self._compute_blind_weights(embedding, labels)
            
            # Apply weights to main loss
            weighted_loss = main_loss * weights
            total_loss = weighted_loss.mean()
            
            outputs.update({
                'loss': total_loss,
                'main_loss': main_loss.mean(),
                'weights': weights,
                'weighted_loss': weighted_loss
            })
        
        if return_dict:
            return outputs
        else:
            return tuple(outputs.values())

    def get_bias_predictions(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get bias predictions for input examples."""
        with torch.no_grad():
            embedding = self._get_embedding(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids
            )
            bias_logits = self.bias_head(embedding)
            bias_probs = F.softmax(bias_logits, dim=-1)
            return bias_probs[:, 1]  # Return probability of being biased

    def compute_metrics(self, dataloader: DataLoader) -> Dict[str, float]:
        """Compute evaluation metrics on a dataset."""
        self.eval()
        total_loss = 0.0
        total_main_loss = 0.0
        total_samples = 0
        all_weights = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.forward(**batch)
                
                if 'loss' in outputs:
                    batch_size = batch['input_ids'].size(0)
                    total_loss += outputs['loss'].item() * batch_size
                    total_main_loss += outputs['main_loss'].item() * batch_size
                    total_samples += batch_size
                    all_weights.extend(outputs['weights'].cpu().numpy())
        
        metrics = {
            'avg_loss': total_loss / total_samples if total_samples > 0 else 0.0,
            'avg_main_loss': total_main_loss / total_samples if total_samples > 0 else 0.0,
            'avg_weight': sum(all_weights) / len(all_weights) if all_weights else 0.0,
            'min_weight': min(all_weights) if all_weights else 0.0,
            'max_weight': max(all_weights) if all_weights else 0.0
        }
        
        return metrics

    @abstractmethod
    def _load_model(self, model_name: str, config: Optional[str] = None) -> nn.Module:
        """Load model from name/path."""
        pass

    @abstractmethod
    def _get_embedding(self, **inputs) -> torch.Tensor:
        """Extract embeddings from the model."""
        pass

    @abstractmethod
    def _setup_loss(self):
        """Setup the loss function."""
        pass

    @abstractmethod
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute per-example loss."""
        pass


class BLINDModelForSequenceClassification(BLINDModel):
    """
    BLIND implementation for sequence classification tasks.
    
    This class implements BLIND debiasing for text classification using
    transformer models with sequence classification heads.
    """

    def _load_model(self, model_name: str, config: Optional[str] = None) -> PreTrainedModel:
        """Load a sequence classification model."""
        try:
            if config:
                model_config = AutoConfig.from_pretrained(config)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, config=model_config
                )
            else:
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            logger.info(f"Loaded model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def _get_embedding(self, **inputs) -> torch.Tensor:
        """Extract [CLS] token embedding from the model."""
        # Get all hidden states from the model
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            outputs = self.model.base_model(**inputs, output_hidden_states=True)
            
        # Use the [CLS] token embedding (first token) from the last hidden layer
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]  # Shape: (batch_size, hidden_dim)
        
        return cls_embedding

    def _setup_loss(self):
        """Setup cross-entropy loss with no reduction."""
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute per-example cross-entropy loss."""
        return self.loss_fct(logits, labels)


class BLINDModelForRegression(BLINDModel):
    """
    BLIND implementation for regression tasks.
    """

    def _load_model(self, model_name: str, config: Optional[str] = None) -> PreTrainedModel:
        """Load a model for regression (using sequence classification with num_labels=1)."""
        try:
            if config:
                model_config = AutoConfig.from_pretrained(config)
                model_config.num_labels = 1
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, config=model_config
                )
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=1
                )
            
            logger.info(f"Loaded regression model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load regression model {model_name}: {e}")
            raise

    def _get_embedding(self, **inputs) -> torch.Tensor:
        """Extract [CLS] token embedding from the model."""
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            outputs = self.model.base_model(**inputs, output_hidden_states=True)
            
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]
        
        return cls_embedding

    def _setup_loss(self):
        """Setup MSE loss with no reduction."""
        self.loss_fct = nn.MSELoss(reduction='none')

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute per-example MSE loss."""
        # Squeeze logits to remove extra dimension for regression
        logits = logits.squeeze(-1)
        return self.loss_fct(logits, labels.float())


class BLINDModelForTokenClassification(BLINDModel):
    """
    BLIND implementation for token classification tasks (NER, POS tagging, etc.).
    """

    def _load_model(self, model_name: str, config: Optional[str] = None) -> PreTrainedModel:
        """Load a token classification model."""
        try:
            from transformers import AutoModelForTokenClassification
            
            if config:
                model_config = AutoConfig.from_pretrained(config)
                model = AutoModelForTokenClassification.from_pretrained(
                    model_name, config=model_config
                )
            else:
                model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            logger.info(f"Loaded token classification model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load token classification model {model_name}: {e}")
            raise

    def _get_embedding(self, **inputs) -> torch.Tensor:
        """Extract mean pooled embeddings from the model."""
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            outputs = self.model.base_model(**inputs, output_hidden_states=True)
            
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs.get('attention_mask')
        
        if attention_mask is not None:
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
        else:
            # Simple mean pooling
            mean_embeddings = torch.mean(last_hidden_state, dim=1)
        
        return mean_embeddings

    def _setup_loss(self):
        """Setup cross-entropy loss with no reduction."""
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute per-example cross-entropy loss for token classification."""
        # Flatten for token-level loss computation
        active_loss = labels.view(-1) != -100  # Ignore padding tokens
        active_logits = logits.view(-1, logits.size(-1))[active_loss]
        active_labels = labels.view(-1)[active_loss]
        
        if active_logits.size(0) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        token_losses = self.loss_fct(active_logits, active_labels)
        
        # Aggregate token losses back to sequence level
        # This is a simplified aggregation - you might want to implement more sophisticated methods
        batch_size = labels.size(0)
        seq_len = labels.size(1)
        
        # Reshape losses back to (batch, seq_len) format
        loss_matrix = torch.zeros_like(labels, dtype=torch.float)
        loss_matrix.view(-1)[active_loss] = token_losses
        
        # Mean loss per sequence
        sequence_losses = loss_matrix.sum(dim=1) / (labels != -100).sum(dim=1).float()
        
        return sequence_losses


# Utility functions
def create_blind_model(
    model_name_or_path: str,
    task_type: str = 'classification',
    **kwargs
) -> BLINDModel:
    """
    Factory function to create appropriate BLIND model based on task type.
    
    Args:
        model_name_or_path: HuggingFace model name or local path
        task_type: Type of task ('classification', 'regression', 'token_classification')
        **kwargs: Additional arguments passed to the BLIND model
    
    Returns:
        Initialized BLIND model
    """
    task_type = task_type.lower()
    
    if task_type == 'classification':
        return BLINDModelForSequenceClassification(model_name_or_path, **kwargs)
    elif task_type == 'regression':
        return BLINDModelForRegression(model_name_or_path, **kwargs)
    elif task_type == 'token_classification':
        return BLINDModelForTokenClassification(model_name_or_path, **kwargs)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def load_blind_model(checkpoint_path: str, task_type: str = 'classification') -> BLINDModel:
    """
    Load a saved BLIND model from checkpoint.
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        task_type: Type of task the model was trained for
    
    Returns:
        Loaded BLIND model
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model configuration from checkpoint
    model_config = checkpoint.get('model_config', {})
    
    # Create model instance
    model = create_blind_model(task_type=task_type, **model_config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded BLIND model from {checkpoint_path}")
    return model