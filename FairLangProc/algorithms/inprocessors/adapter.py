import torch
import torch.nn as nn
import adapters

class DebiasAdapter(nn.Module):
    """
    Initialize a model and add an adapter.
    
    Args:
        model (nn.Module):  Pretrained model
        config (str):       Name of the adapter's configuration 
    """

    def __init__(
            self,
            model: nn.Module,
            config = 'lora'
            ):
        super().__init__()
        
        self.model = model
        # Add an adapter
        adapters.init(self.model)
        
        # Freeze the original model parameters to only train the adapter:
        self.model.add_adapter('debias_adapter', config = config)
        self.model.train_adapter('debias_adapter')

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        forward pass
        """
        # Forward pass is as usual; the adapters will be automatically applied
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        return outputs
