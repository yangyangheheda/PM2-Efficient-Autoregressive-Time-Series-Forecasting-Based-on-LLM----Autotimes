import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from typing import Optional

class SegmentEmbedding(nn.Module):
    def __init__(self, seg_dim: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(seg_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

    def forward(self, x):
        return self.mlp(x)

class SegmentProjection(nn.Module):
    def __init__(self, seg_dim: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, seg_dim)
        )

    def forward(self, h):
        return self.mlp(h)

class AutoTimesModel(nn.Module):
    def __init__(self, llm_name: str = 'gpt2', seg_len: int = 96, num_vars: int = 1):
        super().__init__()
        self.seg_len = seg_len
        self.num_vars = num_vars
        self.seg_dim = seg_len * num_vars
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        for p in self.llm.parameters():
            p.requires_grad = False
        hidden = self.llm.config.hidden_size
        self.seg_embed = SegmentEmbedding(self.seg_dim, hidden)
        self.seg_proj = SegmentProjection(self.seg_dim, hidden)
        self.hidden = hidden

    def forward(self, seg_tokens, ts_emb):
        B, N, _ = seg_tokens.shape
        inp = self.seg_embed(seg_tokens.view(-1, self.seg_dim)) + ts_emb.view(-1, self.hidden)
        inp = inp.view(B, N, self.hidden)
        out = self.llm(inputs_embeds=inp, output_hidden_states=True).last_hidden_state
        next_h = out[:, -1, :]
        pred_seg = self.seg_proj(next_h)
        return pred_seg
