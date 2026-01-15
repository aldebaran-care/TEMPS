from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, ModelOutput

from temporal_embeddings.parameters.parameters import POSITIONAL_ENCODING_DIM

@dataclass
class GaussOutput(ModelOutput):
    mu: torch.FloatTensor = None
    std: torch.FloatTensor = None


class GaussModel(nn.Module):
    def __init__(self, model_name: str, gradient_checkpointing: bool = False) -> None:
        super().__init__()

        self.backbone: PreTrainedModel = AutoModel.from_pretrained(model_name)

        # Freeze backbone weights
        for param in self.backbone.parameters():
            param.requires_grad = False

        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self.hidden_size: int = self.backbone.config.hidden_size

        # Multi-layer projection for better capacity
        self.temporal_projection = nn.Sequential(
            nn.Linear(self.hidden_size + POSITIONAL_ENCODING_DIM, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
        
        # Separate heads for mu and log_var
        self.mu_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        )
        
        self.log_var_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Softplus()  # Ensures positive variance
        )

    def forward(self, input_ids, attention_mask, dates, **_) -> GaussOutput:
        with torch.no_grad():
            outputs: BaseModelOutput = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        emb = self.mean_pooling(outputs, attention_mask)
        emb_dates = torch.cat((emb, dates), dim=-1)

        # Shared temporal projection
        temporal_features = self.temporal_projection(emb_dates)
        
        # Separate mu and variance computation
        mu = self.mu_head(temporal_features)
        log_var = self.log_var_head(temporal_features)
        
        # Clamp log_var to prevent numerical instability
        log_var = torch.clamp(log_var, min=-10, max=10)
        std = torch.exp(0.5 * log_var)

        return GaussOutput(mu=mu, std=std)

    def mean_pooling(self, model_output, attention_mask):
        # token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        token_embeddings = model_output.last_hidden_state
        attention_mask_copy = attention_mask.clone()
        input_mask_expanded = attention_mask_copy.unsqueeze(-1).repeat(1, 1, token_embeddings.size(-1))
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)