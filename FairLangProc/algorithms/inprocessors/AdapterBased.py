import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union
import logging

try:
    import adapters
    from adapters import AdapterConfig, LoRAConfig, PrefixTuningConfig
    ADAPTERS_AVAILABLE = True
except ImportError:
    ADAPTERS_AVAILABLE = False
    logging.warning("adapters library not found. Install with: pip install adapters")

class DebiasAdapter(nn.Module):
    """
    Wrapper for adding adapters to pretrained models for debiasing.
    
    Args:
        model: Pretrained model (e.g., BERT, RoBERTa)
        adapter_config: Adapter configuration name or dict
        adapter_name: Name for the adapter
        freeze_base: Whether to freeze base model parameters
    """
    
    def __init__(self, model: nn.Module, adapter_config: Union[str, Dict] = 'lora',
                 adapter_name: str = 'debias_adapter', freeze_base: bool = True):
        super().__init__()
        
        if not ADAPTERS_AVAILABLE:
            raise ImportError("adapters library required. Install: pip install adapters")
            
        self.model = model
        self.adapter_name = adapter_name
        self.freeze_base = freeze_base
        
        # Initialize adapters
        adapters.init(self.model)
        
        # Configure adapter
        if isinstance(adapter_config, str):
            config = self._get_config(adapter_config)
        else:
            config = adapter_config
            
        # Add and activate adapter
        self.model.add_adapter(adapter_name, config=config)
        self.model.train_adapter(adapter_name)
        
        if freeze_base:
            self._freeze_base_model()
            
        self.config = config
        
    def _get_config(self, config_name: str) -> AdapterConfig:
        """Get predefined adapter configurations"""
        configs = {
            'lora': LoRAConfig(r=16, alpha=32, dropout=0.1),
            'lora_small': LoRAConfig(r=8, alpha=16, dropout=0.1),
            'lora_large': LoRAConfig(r=32, alpha=64, dropout=0.1),
            'bottleneck': AdapterConfig(reduction_factor=16, non_linearity='gelu'),
            'bottleneck_small': AdapterConfig(reduction_factor=32, non_linearity='gelu'),
            'prefix_tuning': PrefixTuningConfig(flat=False, prefix_length=30),
        }
        
        if config_name not in configs:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
            
        return configs[config_name]
        
    def _freeze_base_model(self):
        """Freeze all non-adapter parameters"""
        for name, param in self.model.named_parameters():
            if 'adapter' not in name.lower():
                param.requires_grad = False
                
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through model with adapter"""
        
        # Ensure adapter is active
        self.model.set_active_adapters(self.adapter_name)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            **kwargs
        )
        
        return outputs
        
    def get_adapter_params(self) -> Dict[str, torch.Tensor]:
        """Get adapter parameters for analysis"""
        adapter_params = {}
        for name, param in self.model.named_parameters():
            if 'adapter' in name.lower() and param.requires_grad:
                adapter_params[name] = param
        return adapter_params
        
    def get_param_count(self) -> Dict[str, int]:
        """Get parameter counts"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'adapter_ratio': trainable_params / total_params
        }
        
    def save_adapter(self, path: str):
        """Save only adapter weights"""
        self.model.save_adapter(path, self.adapter_name)
        
    def load_adapter(self, path: str):
        """Load adapter weights"""
        self.model.load_adapter(path, self.adapter_name)


class MultiAdapterModel(nn.Module):
    """Model with multiple adapters for different debiasing tasks"""
    
    def __init__(self, model: nn.Module, adapter_configs: Dict[str, Union[str, Dict]]):
        super().__init__()
        
        if not ADAPTERS_AVAILABLE:
            raise ImportError("adapters library required")
            
        self.model = model
        self.adapter_names = list(adapter_configs.keys())
        
        # Initialize adapters
        adapters.init(self.model)
        
        # Add all adapters
        for name, config in adapter_configs.items():
            if isinstance(config, str):
                config = self._get_config(config)
            self.model.add_adapter(name, config=config)
            
        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False
            
    def _get_config(self, config_name: str) -> AdapterConfig:
        """Get adapter configuration"""
        return DebiasAdapter._get_config(None, config_name)
        
    def forward(self, input_ids: torch.Tensor, adapter_name: str = None,
                attention_mask: Optional[torch.Tensor] = None, **kwargs):
        """Forward pass with specified adapter"""
        
        if adapter_name:
            if adapter_name not in self.adapter_names:
                raise ValueError(f"Unknown adapter: {adapter_name}")
            self.model.set_active_adapters(adapter_name)
        
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
    def train_adapter(self, adapter_name: str):
        """Set specific adapter to training mode"""
        if adapter_name not in self.adapter_names:
            raise ValueError(f"Unknown adapter: {adapter_name}")
        self.model.train_adapter(adapter_name)


class ParameterEfficientDebiaser:
    """Utility class for parameter-efficient debiasing"""
    
    @staticmethod
    def create_lora_model(model: nn.Module, rank: int = 16, alpha: float = 32,
                         dropout: float = 0.1) -> DebiasAdapter:
        """Create LoRA adapter model"""
        config = LoRAConfig(r=rank, alpha=alpha, dropout=dropout)
        return DebiasAdapter(model, config, 'lora_debias')
        
    @staticmethod  
    def create_bottleneck_model(model: nn.Module, reduction_factor: int = 16) -> DebiasAdapter:
        """Create bottleneck adapter model"""
        config = AdapterConfig(reduction_factor=reduction_factor, non_linearity='gelu')
        return DebiasAdapter(model, config, 'bottleneck_debias')
        
    @staticmethod
    def compare_adapters(model: nn.Module, test_input: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Compare different adapter configurations"""
        
        configs = {
            'lora_small': 'lora_small',
            'lora_medium': 'lora',  
            'lora_large': 'lora_large',
            'bottleneck': 'bottleneck'
        }
        
        results = {}
        
        for name, config in configs.items():
            adapter_model = DebiasAdapter(model, config, f'{name}_adapter')
            param_info = adapter_model.get_param_count()
            
            # Test forward pass
            with torch.no_grad():
                outputs = adapter_model(**test_input)
                
            results[name] = {
                'trainable_params': param_info['trainable'],
                'adapter_ratio': param_info['adapter_ratio'],
                'output_shape': outputs.last_hidden_state.shape if hasattr(outputs, 'last_hidden_state') else None
            }
            
        return results


# Training utilities
def get_adapter_optimizer(adapter_model: DebiasAdapter, lr: float = 1e-4) -> torch.optim.Optimizer:
    """Get optimizer for adapter parameters only"""
    adapter_params = [p for p in adapter_model.parameters() if p.requires_grad]
    return torch.optim.AdamW(adapter_params, lr=lr, weight_decay=0.01)


def freeze_base_except_adapters(model: nn.Module):
    """Freeze all parameters except adapters"""
    for name, param in model.named_parameters():
        if 'adapter' not in name.lower():
            param.requires_grad = False