from dataclasses import dataclass
from typing import Optional

from transformers.utils import ModelOutput
import torch


@dataclass
class CustomOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None