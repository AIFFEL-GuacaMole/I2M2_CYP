# fusions.py

import torch
import torch.nn as nn

class ConcatFusion(nn.Module):
    def __init__(self, input_dims, out_dim):
        super(ConcatFusion, self).__init__()
        total_in = sum(input_dims)
        self.linear = nn.Linear(total_in, out_dim)

    def forward(self, *features):
        cat_feat = torch.cat(features, dim=1)
        out = self.linear(cat_feat)
        return out

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, out_dim=256):
        super(CrossAttentionFusion, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear = nn.Linear(embed_dim, out_dim)

    def forward(self, f1, f2, f3):
        B, D = f1.shape
        q = f1.unsqueeze(1)
        kv= torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
        attn_out, _ = self.attn(q, kv, kv)
        out = self.linear(attn_out.squeeze(1))
        return out