# Standard imports
from typing import Union

# Pytorch
import torch
import torch.nn as nn

# Adapters
import adapters


class DebiasAdapter(nn.Module):
    """
    Implements ADELE debiasing based on bottleneck adapter.
    
    Args:
        model (nn.Module):                  Pretrained model (e.g., BERT, GPT-2)
        adapter_name (str):                 Tensor with ids of text with demographic information of group A
        adapter_config (Union[str, dict]):  Name or dictionary of the desired configuration for the adapter (bottleneck by default)
    """
    def __init__(
        self,
        model: nn.Module,
        adapter_name: str = "debias_adapter",
        adapter_config: Union[str, dict] = "seq_bn",
    ):
        
        super().__init__()
        self.adapter_name = adapter_name
        adapters.init(model)
        self.model = model

        # Verify support
        if not hasattr(self.model, "add_adapter"):
            raise ValueError("Model does not support adapters.")

        # Load adapter config
        if isinstance(adapter_config, str):
            config = adapters.AdapterConfig.load(adapter_config)
        elif isinstance(adapter_config, dict):
            config = adapters.AdapterConfig(**adapter_config)
        else:
            config = adapter_config

        # Add adapter and set it up
        self.model.add_adapter(adapter_name, config=config)
        self.model.set_active_adapters(adapter_name)
        self.model.train_adapter(self.adapter_name)

    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def get_model(self):
        return self.model

    def save_adapter(self, save_path: str):
        self.model.save_adapter(save_path, self.adapter_name)

    def load_adapter(self, path: str):
        self.model.load_adapter(path, load_as=self.adapter_name)
        self.model.set_active_adapters(self.adapter_name)