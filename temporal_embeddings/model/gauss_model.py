from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, ModelOutput

from temporal_embeddings.parameters.parameters import POSITIONAL_ENCODING_DIM
from temporal_embeddings.model.attention_pooling import AttentionPooling

@dataclass
class GaussOutput(ModelOutput):
    mu: torch.FloatTensor = None
    std: torch.FloatTensor = None


class GaussModel(nn.Module):
    def __init__(self, model_name: str, gradient_checkpointing: bool = False) -> None:
        super().__init__()

        self.backbone: PreTrainedModel = AutoModel.from_pretrained(model_name, local_files_only=True)

        # Freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self.hidden_size: int = self.backbone.config.hidden_size

        num_dimensions = self.hidden_size + POSITIONAL_ENCODING_DIM

        self.attention_pooling = AttentionPooling(self.hidden_size)

        # Multi-layer projection for better capacity
        self.temporal_projection = nn.Sequential(
            nn.Linear(num_dimensions, num_dimensions * 2),
            nn.LayerNorm(num_dimensions * 2),
            nn.ReLU()
        )
        
        # Separate heads for mu and log_var
        self.mu_head = nn.Sequential(
            nn.Linear(num_dimensions * 2, self.hidden_size),
            nn.Tanh()
        )
        
        self.log_var_head = nn.Linear(num_dimensions * 2, self.hidden_size)
    
    def forward(self, input_ids, attention_mask, dates, **_) -> GaussOutput:
        with torch.no_grad():
            outputs: BaseModelOutput = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        emb = self.attention_pooling(outputs.last_hidden_state, attention_mask)
        emb_dates = torch.cat((emb, dates), dim=-1)

        # Shared temporal projection
        temporal_features = self.temporal_projection(emb_dates)
        
        # Separate mu and variance computation
        mu = self.mu_head(temporal_features)
        
        log_var = self.log_var_head(temporal_features)
        std = torch.sqrt(log_var.exp())

        return GaussOutput(mu=mu, std=std)