# Standard imports
from typing import Union

# Pytorch
import torch
import torch.nn as nn

# Adapters
import adapters


class DebiasAdapter(nn.Module):
    """Implements ADELE debiasing based on bottleneck adapter.
    
    Example
    -------
    >>> from adapters import AdapterTrainer
    >>> from FairLangProc.algorithms.inprocessors import DebiasAdapter
    >>> 
    >>> DebiasAdapter = DebiasAdapter(
    >>>     model = AutoModel.from_pretrained('bert-base-uncased'),
    >>>     adapter_config = "seq_bn"
    >>>     )
    >>> AdeleModel = DebiasAdapter.get_model()
    >>> 
    >>> trainer = AdapterTrainer(
    >>>     model=AdeleModel,
    >>>     args=training_args,
    >>>     train_dataset=train_CDA,
    >>>     eval_dataset=val_dataset,
    >>>     optimizers=(
    >>>         AdamW(AdeleModel.parameters(),lr=1e-5, weight_decay=0.1),
    >>>         None
    >>>         )
    >>> )
    >>> trainer.train()
    >>> results = trainer.evaluate()
    >>> print(results)
    """
    def __init__(
        self,
        model: nn.Module,
        adapter_name: str = "debias_adapter",
        adapter_config: Union[str, dict] = "seq_bn",
    ) -> None:
        r"""Constructor of the DebiasAdapter class.
        
        Parameters
        ----------
        model : nn.Module 
            Pretrained model (e.g., BERT, GPT-2)
        adapter_name : str        
            Tensor with ids of text with demographic information of group A
        adapter_config : Union[str, dict]
            Name or dictionary of the desired configuration for the adapter (bottleneck by default)
        """
        
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