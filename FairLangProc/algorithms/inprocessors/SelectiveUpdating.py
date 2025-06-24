import torch.nn as nn
from typing import List, Dict
import re

def freeze_all_parameters(model: nn.Module) -> None:
    """
    Freeze all parameters of the model.
    
    Args:
        model: Model whose parameters will be frozen
    """
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_all_parameters(model: nn.Module) -> None:
    """
    Unfreeze all parameters of the model.
    
    Args:
        model: Model whose parameters will be unfrozen
    """
    for param in model.parameters():
        param.requires_grad = True

def freeze_by_name(model: nn.Module, parameter_patterns: List[str]) -> None:
    """
    Freeze parameters whose names match any of the specified patterns.
    
    Args:
        model: The model whose parameters will be adjusted
        parameter_patterns: List of patterns to match parameter names
    """
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in parameter_patterns):
            param.requires_grad = False

def unfreeze_by_name(model: nn.Module, parameter_patterns: List[str]) -> None:
    """
    Unfreeze parameters whose names match any of the specified patterns.
    
    Args:
        model: The model whose parameters will be adjusted
        parameter_patterns: List of patterns to match parameter names
    """
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in parameter_patterns):
            param.requires_grad = True

def freeze_by_regex(model: nn.Module, regex_patterns: List[str]) -> None:
    """
    Freeze parameters whose names match any of the regex patterns.
    
    Args:
        model: The model whose parameters will be adjusted
        regex_patterns: List of regex patterns to match parameter names
    """
    compiled_patterns = [re.compile(pattern) for pattern in regex_patterns]
    
    for name, param in model.named_parameters():
        if any(pattern.search(name) for pattern in compiled_patterns):
            param.requires_grad = False

def unfreeze_by_regex(model: nn.Module, regex_patterns: List[str]) -> None:
    """
    Unfreeze parameters whose names match any of the regex patterns.
    
    Args:
        model: The model whose parameters will be adjusted
        regex_patterns: List of regex patterns to match parameter names
    """
    compiled_patterns = [re.compile(pattern) for pattern in regex_patterns]
    
    for name, param in model.named_parameters():
        if any(pattern.search(name) for pattern in compiled_patterns):
            param.requires_grad = True

def selective_unfreezing(model: nn.Module, parameter_patterns: List[str]) -> None:
    """
    Freeze all model parameters and selectively unfreeze those matching patterns.
    
    Args:
        model: The model whose parameters will be adjusted
        parameter_patterns: List of patterns to match parameter names for unfreezing
    """
    freeze_all_parameters(model)
    unfreeze_by_name(model, parameter_patterns)

def selective_freezing(model: nn.Module, parameter_patterns: List[str]) -> None:
    """
    Unfreeze all model parameters and selectively freeze those matching patterns.
    
    Args:
        model: The model whose parameters will be adjusted
        parameter_patterns: List of patterns to match parameter names for freezing
    """
    unfreeze_all_parameters(model)
    freeze_by_name(model, parameter_patterns)

def freeze_by_layer_range(model: nn.Module, start_layer: int, end_layer: int, 
                         layer_pattern: str = "layer") -> None:
    """
    Freeze parameters in a range of layers.
    
    Args:
        model: The model whose parameters will be adjusted
        start_layer: Starting layer index (inclusive)
        end_layer: Ending layer index (inclusive)
        layer_pattern: Pattern to identify layers in parameter names
    """
    for name, param in model.named_parameters():
        if layer_pattern in name:
            # Extract layer number from parameter name
            layer_match = re.search(rf'{layer_pattern}\.(\d+)', name)
            if layer_match:
                layer_num = int(layer_match.group(1))
                if start_layer <= layer_num <= end_layer:
                    param.requires_grad = False

def unfreeze_by_layer_range(model: nn.Module, start_layer: int, end_layer: int,
                           layer_pattern: str = "layer") -> None:
    """
    Unfreeze parameters in a range of layers.
    
    Args:
        model: The model whose parameters will be adjusted
        start_layer: Starting layer index (inclusive)
        end_layer: Ending layer index (inclusive)
        layer_pattern: Pattern to identify layers in parameter names
    """
    for name, param in model.named_parameters():
        if layer_pattern in name:
            layer_match = re.search(rf'{layer_pattern}\.(\d+)', name)
            if layer_match:
                layer_num = int(layer_match.group(1))
                if start_layer <= layer_num <= end_layer:
                    param.requires_grad = True

def freeze_by_module_type(model: nn.Module, module_types: List[type]) -> None:
    """
    Freeze parameters of specific module types.
    
    Args:
        model: The model whose parameters will be adjusted
        module_types: List of module types to freeze
    """
    for module in model.modules():
        if type(module) in module_types:
            for param in module.parameters():
                param.requires_grad = False

def unfreeze_by_module_type(model: nn.Module, module_types: List[type]) -> None:
    """
    Unfreeze parameters of specific module types.
    
    Args:
        model: The model whose parameters will be adjusted
        module_types: List of module types to unfreeze
    """
    for module in model.modules():
        if type(module) in module_types:
            for param in module.parameters():
                param.requires_grad = True

def get_parameter_info(model: nn.Module) -> Dict[str, Dict]:
    """
    Get detailed information about model parameters.
    
    Args:
        model: The model to analyze
        
    Returns:
        Dictionary containing parameter information
    """
    info = {
        'total_params': 0,
        'trainable_params': 0,
        'frozen_params': 0,
        'parameter_details': []
    }
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        is_trainable = param.requires_grad
        
        info['total_params'] += param_count
        if is_trainable:
            info['trainable_params'] += param_count
        else:
            info['frozen_params'] += param_count
            
        info['parameter_details'].append({
            'name': name,
            'shape': list(param.shape),
            'param_count': param_count,
            'trainable': is_trainable,
            'dtype': str(param.dtype)
        })
    
    info['trainable_ratio'] = info['trainable_params'] / info['total_params'] if info['total_params'] > 0 else 0
    
    return info

def print_parameter_summary(model: nn.Module) -> None:
    """
    Print a summary of model parameters.
    
    Args:
        model: The model to summarize
    """
    info = get_parameter_info(model)
    
    print(f"Parameter Summary:")
    print(f"  Total parameters: {info['total_params']:,}")
    print(f"  Trainable parameters: {info['trainable_params']:,}")
    print(f"  Frozen parameters: {info['frozen_params']:,}")
    print(f"  Trainable ratio: {info['trainable_ratio']:.4f}")

def get_parameters_by_pattern(model: nn.Module, patterns: List[str]) -> List[str]:
    """
    Get parameter names that match any of the given patterns.
    
    Args:
        model: The model to search
        patterns: List of patterns to match
        
    Returns:
        List of matching parameter names
    """
    matching_params = []
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            matching_params.append(name)
    return matching_params

def create_parameter_groups(model: nn.Module, group_configs: Dict[str, Dict]) -> List[Dict]:
    """
    Create parameter groups for optimizers with different learning rates.
    
    Args:
        model: The model whose parameters to group
        group_configs: Dictionary mapping group names to config dicts
                      Each config should have 'patterns' and optionally 'lr', 'weight_decay', etc.
    
    Returns:
        List of parameter groups for optimizer
    """
    parameter_groups = []
    used_params = set()
    
    for group_name, config in group_configs.items():
        patterns = config.get('patterns', [])
        group_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad and name not in used_params:
                if any(pattern in name for pattern in patterns):
                    group_params.append(param)
                    used_params.add(name)
        
        if group_params:
            group_dict = {'params': group_params, 'name': group_name}
            # Add other config options (lr, weight_decay, etc.)
            for key, value in config.items():
                if key != 'patterns':
                    group_dict[key] = value
            parameter_groups.append(group_dict)
    
    # Add remaining parameters to default group
    remaining_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and name not in used_params:
            remaining_params.append(param)
    
    if remaining_params:
        parameter_groups.append({'params': remaining_params, 'name': 'default'})
    
    return parameter_groups

class ParameterController:
    """
    A controller class for managing parameter freezing/unfreezing strategies.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_states = {}
        self._save_original_states()
    
    def _save_original_states(self) -> None:
        """Save the original requires_grad states"""
        for name, param in self.model.named_parameters():
            self.original_states[name] = param.requires_grad
    
    def restore_original_states(self) -> None:
        """Restore parameters to their original requires_grad states"""
        for name, param in self.model.named_parameters():
            if name in self.original_states:
                param.requires_grad = self.original_states[name]
    
    def apply_strategy(self, strategy: str, **kwargs) -> None:
        """
        Apply a parameter updating strategy.
        
        Args:
            strategy: Name of the strategy to apply
            **kwargs: Additional arguments for the strategy
        """
        strategies = {
            'freeze_all': lambda: freeze_all_parameters(self.model),
            'unfreeze_all': lambda: unfreeze_all_parameters(self.model),
            'selective_unfreezing': lambda: selective_unfreezing(
                self.model, kwargs.get('patterns', [])
            ),
            'selective_freezing': lambda: selective_freezing(
                self.model, kwargs.get('patterns', [])
            ),
            'freeze_layers': lambda: freeze_by_layer_range(
                self.model, 
                kwargs.get('start_layer', 0),
                kwargs.get('end_layer', 0),
                kwargs.get('layer_pattern', 'layer')
            ),
            'unfreeze_layers': lambda: unfreeze_by_layer_range(
                self.model,
                kwargs.get('start_layer', 0), 
                kwargs.get('end_layer', 0),
                kwargs.get('layer_pattern', 'layer')
            ),
            'freeze_module_types': lambda: freeze_by_module_type(
                self.model, kwargs.get('module_types', [])
            ),
            'unfreeze_module_types': lambda: unfreeze_by_module_type(
                self.model, kwargs.get('module_types', [])
            )
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        strategies[strategy]()
    
    def get_summary(self) -> Dict:
        """Get current parameter summary"""
        return get_parameter_info(self.model)
    
    def print_summary(self) -> None:
        """Print current parameter summary"""
        print_parameter_summary(self.model)

# Predefined strategies for common use cases
class CommonStrategies:
    """Common parameter updating strategies for different model types"""
    
    @staticmethod
    def bert_fine_tune_classifier_only(model: nn.Module) -> None:
        """Fine-tune only the classifier head of BERT"""
        selective_unfreezing(model, ['classifier'])
    
    @staticmethod
    def bert_fine_tune_last_layers(model: nn.Module, num_layers: int = 2) -> None:
        """Fine-tune last N layers of BERT"""
        # Unfreeze classifier and last N encoder layers
        patterns = ['classifier']
        if hasattr(model, 'bert') and hasattr(model.bert, 'encoder'):
            total_layers = len(model.bert.encoder.layer)
            start_layer = max(0, total_layers - num_layers)
            patterns.extend([f'encoder.layer.{i}' for i in range(start_layer, total_layers)])
        
        selective_unfreezing(model, patterns)
    
    @staticmethod
    def bert_fine_tune_attention_only(model: nn.Module) -> None:
        """Fine-tune only attention layers"""
        selective_unfreezing(model, ['attention', 'classifier'])
    
    @staticmethod
    def gradual_unfreezing_schedule(model: nn.Module, current_epoch: int, 
                                   unfreeze_schedule: Dict[int, List[str]]) -> None:
        """
        Gradually unfreeze parameters based on training epoch.
        
        Args:
            model: The model to update
            current_epoch: Current training epoch
            unfreeze_schedule: Dict mapping epochs to lists of patterns to unfreeze
        """
        for epoch, patterns in unfreeze_schedule.items():
            if current_epoch >= epoch:
                unfreeze_by_name(model, patterns)