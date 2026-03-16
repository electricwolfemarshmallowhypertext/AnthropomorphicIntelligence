import torch.nn as nn
from transformers.activations import ACT2FN


class ResponseHead(nn.Module):
    def __init__(self, hidden_size, down_size, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = int(hidden_size / 4)
        self.down_size =down_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.down_size, bias=bias)
        self.act_fn = ACT2FN['gelu']

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))