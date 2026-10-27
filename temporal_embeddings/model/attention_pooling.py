import torch
from torch import nn

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        # This linear layer acts as the "trainable eyes" looking at each token
        self.attention_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, last_hidden_state, attention_mask):
        # 1. Calculate "importance scores" for each token
        scores = self.attention_scorer(last_hidden_state)
        
        # 2. Mask out padding tokens so they don't influence the softmax
        mask_value = -1e9
        extended_mask = (1.0 - attention_mask.unsqueeze(-1)) * mask_value
        scores = scores + extended_mask
        
        # 3. Calculate weights (probabilities)
        weights = torch.softmax(scores, dim=1)
        
        # 4. Compute weighted average
        embeddings = torch.sum(last_hidden_state * weights, dim=1)
        
        return embeddings