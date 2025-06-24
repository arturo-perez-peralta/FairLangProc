import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
import logging
from contextlib import contextmanager
import weakref
import numpy as np
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionHookManager:
    """
    Advanced hook manager for modifying attention mechanisms in transformer models.
    Supports multiple modification strategies, dynamic parameter adjustment, and
    comprehensive monitoring capabilities.
    """
    
    def __init__(self):
        self.hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self.active_modifications: Dict[str, Dict] = {}
        self.attention_stats: Dict[str, List] = defaultdict(list)
        self.model_ref: Optional[weakref.ref] = None
        
    def add_EAT_hook(
        self, 
        model: nn.Module, 
        beta: float = 1.1,
        layer_indices: Optional[List[int]] = None,
        hook_name: str = "default_eat",
        collect_stats: bool = False,
        adaptive_beta: bool = False,
        min_beta: float = 0.5,
        max_beta: float = 2.0
    ) -> Dict[str, Any]:
        """
        Enhanced EAT (Enhanced Attention Temperature) hook with advanced features.
        
        Args:
            model (nn.Module): Target model (supports BERT, RoBERTa, GPT-2, T5, etc.)
            beta (float): Temperature parameter for attention scaling
            layer_indices (List[int], optional): Specific layers to modify. If None, modifies all layers
            hook_name (str): Unique identifier for this hook configuration
            collect_stats (bool): Whether to collect attention statistics
            adaptive_beta (bool): Whether to dynamically adjust beta based on attention entropy
            min_beta (float): Minimum beta value for adaptive mode
            max_beta (float): Maximum beta value for adaptive mode
            
        Returns:
            Dict containing hook information and statistics
        """
        
        if hook_name in self.hooks:
            logger.warning(f"Hook '{hook_name}' already exists. Removing previous hook.")
            self.remove_hook(hook_name)
        
        self.model_ref = weakref.ref(model)
        attention_layers = self._get_attention_layers(model)
        
        if not attention_layers:
            raise ValueError("No attention layers found in the model")
        
        # Filter layers if specific indices are provided
        if layer_indices is not None:
            attention_layers = {k: v for i, (k, v) in enumerate(attention_layers.items()) 
                             if i in layer_indices}
        
        stats = {
            'hook_name': hook_name,
            'beta': beta,
            'layers_modified': len(attention_layers),
            'layer_names': list(attention_layers.keys()),
            'adaptive_beta': adaptive_beta
        }
        
        def create_attention_hook(layer_name: str):
            def attention_hook(module, input, output):
                return self._process_attention_output(
                    output, beta, layer_name, collect_stats, 
                    adaptive_beta, min_beta, max_beta
                )
            return attention_hook
        
        # Register hooks on attention layers
        handles = []
        for layer_name, layer in attention_layers.items():
            hook_fn = create_attention_hook(layer_name)
            
            # Try different common attention module patterns
            attention_module = self._get_attention_module(layer)
            if attention_module:
                handle = attention_module.register_forward_hook(hook_fn)
                handles.append(handle)
                logger.info(f"Registered EAT hook on layer: {layer_name}")
            else:
                logger.warning(f"Could not find attention module in layer: {layer_name}")
        
        if handles:
            self.hooks[hook_name] = handles
            self.active_modifications[hook_name] = {
                'type': 'EAT',
                'beta': beta,
                'adaptive_beta': adaptive_beta,
                'layer_count': len(handles)
            }
            logger.info(f"Successfully registered {len(handles)} EAT hooks under name '{hook_name}'")
        else:
            raise RuntimeError("Failed to register any hooks")
        
        return stats
    
    def add_attention_dropout_hook(
        self,
        model: nn.Module,
        dropout_rate: float = 0.1,
        layer_indices: Optional[List[int]] = None,
        hook_name: str = "attention_dropout"
    ) -> Dict[str, Any]:
        """
        Add hooks to modify attention dropout dynamically.
        """
        if hook_name in self.hooks:
            self.remove_hook(hook_name)
        
        attention_layers = self._get_attention_layers(model)
        
        if layer_indices is not None:
            attention_layers = {k: v for i, (k, v) in enumerate(attention_layers.items()) 
                             if i in layer_indices}
        
        def create_dropout_hook(layer_name: str):
            def dropout_hook(module, input, output):
                if self.model_ref and self.model_ref().training:
                    attention_scores = output[0] if isinstance(output, tuple) else output
                    # Apply custom dropout
                    dropout_mask = torch.rand_like(attention_scores) > dropout_rate
                    modified_scores = attention_scores * dropout_mask.float()
                    
                    if isinstance(output, tuple):
                        return (modified_scores,) + output[1:]
                    return modified_scores
                return output
            return dropout_hook
        
        handles = []
        for layer_name, layer in attention_layers.items():
            attention_module = self._get_attention_module(layer)
            if attention_module:
                hook_fn = create_dropout_hook(layer_name)
                handle = attention_module.register_forward_hook(hook_fn)
                handles.append(handle)
        
        if handles:
            self.hooks[hook_name] = handles
            self.active_modifications[hook_name] = {
                'type': 'dropout',
                'dropout_rate': dropout_rate,
                'layer_count': len(handles)
            }
        
        return {
            'hook_name': hook_name,
            'dropout_rate': dropout_rate,
            'layers_modified': len(handles)
        }
    
    def add_attention_head_masking_hook(
        self,
        model: nn.Module,
        head_mask_prob: float = 0.1,
        layer_indices: Optional[List[int]] = None,
        hook_name: str = "head_masking"
    ) -> Dict[str, Any]:
        """
        Add hooks to randomly mask attention heads during training.
        """
        if hook_name in self.hooks:
            self.remove_hook(hook_name)
        
        attention_layers = self._get_attention_layers(model)
        
        if layer_indices is not None:
            attention_layers = {k: v for i, (k, v) in enumerate(attention_layers.items()) 
                             if i in layer_indices}
        
        def create_head_mask_hook(layer_name: str):
            def head_mask_hook(module, input, output):
                if self.model_ref and self.model_ref().training:
                    attention_scores = output[0] if isinstance(output, tuple) else output
                    
                    # Get number of heads from attention scores shape
                    # Typical shape: [batch, heads, seq_len, seq_len]
                    if len(attention_scores.shape) >= 4:
                        num_heads = attention_scores.shape[1]
                        batch_size = attention_scores.shape[0]
                        
                        # Create head mask
                        head_mask = torch.rand(batch_size, num_heads, 1, 1, 
                                             device=attention_scores.device) > head_mask_prob
                        masked_scores = attention_scores * head_mask.float()
                        
                        if isinstance(output, tuple):
                            return (masked_scores,) + output[1:]
                        return masked_scores
                return output
            return head_mask_hook
        
        handles = []
        for layer_name, layer in attention_layers.items():
            attention_module = self._get_attention_module(layer)
            if attention_module:
                hook_fn = create_head_mask_hook(layer_name)
                handle = attention_module.register_forward_hook(hook_fn)
                handles.append(handle)
        
        if handles:
            self.hooks[hook_name] = handles
            self.active_modifications[hook_name] = {
                'type': 'head_masking',
                'head_mask_prob': head_mask_prob,
                'layer_count': len(handles)
            }
        
        return {
            'hook_name': hook_name,
            'head_mask_prob': head_mask_prob,
            'layers_modified': len(handles)
        }
    
    def _process_attention_output(
        self, 
        output: Union[torch.Tensor, Tuple], 
        beta: float, 
        layer_name: str,
        collect_stats: bool,
        adaptive_beta: bool,
        min_beta: float,
        max_beta: float
    ) -> Union[torch.Tensor, Tuple]:
        """Process and modify attention output with optional statistics collection."""
        
        if isinstance(output, tuple):
            attention_scores = output[0]
            rest = output[1:]
        else:
            attention_scores = output
            rest = ()
        
        # Collect statistics if requested
        if collect_stats:
            with torch.no_grad():
                entropy = self._calculate_attention_entropy(attention_scores)
                self.attention_stats[layer_name].append({
                    'entropy': entropy.mean().item(),
                    'max_attention': attention_scores.max().item(),
                    'min_attention': attention_scores.min().item(),
                    'beta_used': beta
                })
        
        # Adaptive beta adjustment based on attention entropy
        if adaptive_beta:
            with torch.no_grad():
                entropy = self._calculate_attention_entropy(attention_scores)
                # Higher entropy -> lower beta (less sharpening needed)
                # Lower entropy -> higher beta (more sharpening needed)
                avg_entropy = entropy.mean()
                # Normalize entropy to [0, 1] range and map to beta range
                normalized_entropy = torch.sigmoid(avg_entropy - 2.0)  # Adjust threshold as needed
                adaptive_beta_val = min_beta + (max_beta - min_beta) * (1 - normalized_entropy)
                beta = adaptive_beta_val.item()
        
        # Apply temperature scaling
        modified_scores = attention_scores * beta
        
        if rest:
            return (modified_scores,) + rest
        return modified_scores
    
    def _calculate_attention_entropy(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """Calculate entropy of attention distributions."""
        # Apply softmax to get probabilities
        attention_probs = torch.softmax(attention_scores, dim=-1)
        # Calculate entropy: -sum(p * log(p))
        log_probs = torch.log(attention_probs + 1e-12)  # Add small epsilon for numerical stability
        entropy = -torch.sum(attention_probs * log_probs, dim=-1)
        return entropy
    
    def _get_attention_layers(self, model: nn.Module) -> Dict[str, nn.Module]:
        """
        Automatically detect attention layers in various transformer architectures.
        """
        attention_layers = {}
        
        # Common patterns for different model architectures
        patterns = [
            # BERT, RoBERTa, DistilBERT
            ('encoder.layer', lambda m: hasattr(m, 'attention')),
            ('bert.encoder.layer', lambda m: hasattr(m, 'attention')),
            ('roberta.encoder.layer', lambda m: hasattr(m, 'attention')),
            ('distilbert.transformer.layer', lambda m: hasattr(m, 'attention')),
            
            # GPT-2, GPT
            ('transformer.h', lambda m: hasattr(m, 'attn')),
            ('h', lambda m: hasattr(m, 'attn')),
            
            # T5
            ('encoder.block', lambda m: hasattr(m, 'layer') and len(m.layer) > 0),
            ('decoder.block', lambda m: hasattr(m, 'layer') and len(m.layer) > 0),
            
            # Generic transformer layer
            ('layers', lambda m: hasattr(m, 'self_attn') or hasattr(m, 'attention')),
        ]
        
        for pattern, check_fn in patterns:
            try:
                layers = self._get_nested_attr(model, pattern)
                if layers is not None:
                    if hasattr(layers, '__iter__') and not isinstance(layers, nn.Module):
                        # It's a list/ModuleList
                        for i, layer in enumerate(layers):
                            if check_fn(layer):
                                attention_layers[f"{pattern}.{i}"] = layer
                    elif check_fn(layers):
                        attention_layers[pattern] = layers
            except AttributeError:
                continue
        
        return attention_layers
    
    def _get_attention_module(self, layer: nn.Module) -> Optional[nn.Module]:
        """Extract the specific attention module from a transformer layer."""
        
        # Common attention module patterns
        attention_paths = [
            'attention.self',  # BERT-style
            'attention',       # Generic
            'attn',           # GPT-2 style
            'self_attn',      # Some transformer implementations
            'layer.0',        # T5 encoder self-attention
        ]
        
        for path in attention_paths:
            try:
                attention_module = self._get_nested_attr(layer, path)
                if attention_module is not None:
                    return attention_module
            except AttributeError:
                continue
        
        return None
    
    def _get_nested_attr(self, obj: nn.Module, attr_path: str) -> Optional[nn.Module]:
        """Get nested attribute from object using dot notation."""
        try:
            attrs = attr_path.split('.')
            for attr in attrs:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            return None
    
    def update_beta(self, hook_name: str, new_beta: float) -> bool:
        """
        Dynamically update beta parameter for an existing EAT hook.
        
        Args:
            hook_name (str): Name of the hook to update
            new_beta (float): New beta value
            
        Returns:
            bool: True if successful, False otherwise
        """
        if hook_name not in self.active_modifications:
            logger.error(f"Hook '{hook_name}' not found")
            return False
        
        if self.active_modifications[hook_name]['type'] != 'EAT':
            logger.error(f"Hook '{hook_name}' is not an EAT hook")
            return False
        
        self.active_modifications[hook_name]['beta'] = new_beta
        logger.info(f"Updated beta for hook '{hook_name}' to {new_beta}")
        return True
    
    def get_attention_stats(self, hook_name: Optional[str] = None) -> Dict:
        """
        Get collected attention statistics.
        
        Args:
            hook_name (str, optional): Specific hook name. If None, returns all stats.
            
        Returns:
            Dict: Statistics dictionary
        """
        if hook_name:
            return {hook_name: dict(self.attention_stats.get(hook_name, {}))}
        return dict(self.attention_stats)
    
    def remove_hook(self, hook_name: str) -> bool:
        """
        Remove a specific hook.
        
        Args:
            hook_name (str): Name of the hook to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        if hook_name not in self.hooks:
            logger.warning(f"Hook '{hook_name}' not found")
            return False
        
        handles = self.hooks[hook_name]
        if isinstance(handles, list):
            for handle in handles:
                handle.remove()
        else:
            handles.remove()
        
        del self.hooks[hook_name]
        if hook_name in self.active_modifications:
            del self.active_modifications[hook_name]
        if hook_name in self.attention_stats:
            del self.attention_stats[hook_name]
        
        logger.info(f"Removed hook '{hook_name}'")
        return True
    
    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for hook_name in list(self.hooks.keys()):
            self.remove_hook(hook_name)
        logger.info("Removed all hooks")
    
    def list_active_hooks(self) -> Dict[str, Dict]:
        """List all currently active hooks and their configurations."""
        return dict(self.active_modifications)
    
    @contextmanager
    def temporary_hook(self, model: nn.Module, **kwargs):
        """
        Context manager for temporary hook application.
        
        Usage:
            with hook_manager.temporary_hook(model, beta=1.5) as hook_info:
                output = model(input_ids)
        """
        temp_name = f"temp_{id(self)}"
        try:
            hook_info = self.add_EAT_hook(model, hook_name=temp_name, **kwargs)
            yield hook_info
        finally:
            self.remove_hook(temp_name)
    
    def __del__(self):
        """Cleanup hooks when the manager is destroyed."""
        try:
            self.remove_all_hooks()
        except:
            pass  # Ignore errors during cleanup


# Convenience functions for backward compatibility and ease of use
def add_EAT_hook(model: nn.Module, beta: float = 1.1, **kwargs) -> AttentionHookManager:
    """
    Convenience function to quickly add EAT hooks to a model.
    
    Args:
        model (nn.Module): Target model
        beta (float): Temperature parameter
        **kwargs: Additional arguments passed to AttentionHookManager.add_EAT_hook
        
    Returns:
        AttentionHookManager: Hook manager instance for further control
    """
    manager = AttentionHookManager()
    manager.add_EAT_hook(model, beta=beta, **kwargs)
    return manager


def create_attention_analysis_suite(model: nn.Module) -> AttentionHookManager:
    """
    Create a comprehensive attention analysis setup with multiple hook types.
    
    Args:
        model (nn.Module): Target model
        
    Returns:
        AttentionHookManager: Configured hook manager
    """
    manager = AttentionHookManager()
    
    # Add EAT hook with statistics collection
    manager.add_EAT_hook(
        model, 
        beta=1.2, 
        hook_name="analysis_eat",
        collect_stats=True,
        adaptive_beta=True
    )
    
    return manager


# Example usage and testing
if __name__ == "__main__":
    # Example with a simple model structure (you would use your actual model)
    class MockTransformerLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = nn.Module()
            self.base_model.encoder = nn.Module()
            self.base_model.encoder.layer = nn.ModuleList([
                MockTransformerLayer() for _ in range(12)
            ])
    
    # Create mock model and hook manager
    model = MockModel()
    hook_manager = AttentionHookManager()
    
    # Add EAT hooks with different configurations
    stats = hook_manager.add_EAT_hook(
        model, 
        beta=1.5, 
        hook_name="test_eat",
        collect_stats=True,
        adaptive_beta=True
    )
    
    print("Hook registration stats:", stats)
    print("Active hooks:", hook_manager.list_active_hooks())
    
    # Example of temporary hook usage
    with hook_manager.temporary_hook(model, beta=2.0, hook_name="temp_test") as temp_stats:
        print("Temporary hook stats:", temp_stats)
        # Your model inference would go here
    
    # Cleanup
    hook_manager.remove_all_hooks()
    print("All hooks removed")