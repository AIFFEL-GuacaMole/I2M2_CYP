# unimol_3d.py

import torch
import torch.nn as nn

class UniMol_3D(nn.Module):
    def __init__(self, hidden_dim=256, dropout_rate=0.4, task_type="classification", max_atomic_len=None):
        super(UniMol_3D, self).__init__()
        self.task_type = task_type
        self.cls_embedding_dim = 512
        self.atomic_embedding_dim = 512
        self.dropout_rate = dropout_rate
        self.max_atomic_len = max_atomic_len
        self.attn = nn.MultiheadAttention(self.atomic_embedding_dim, num_heads=4, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(self.cls_embedding_dim + self.atomic_embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        if self.task_type=="classification":
            self.classifier = nn.Linear(hidden_dim, 2)
        else:
            self.classifier = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward_middle(self, cls_embeddings, atomic_embeddings):
        device = cls_embeddings.device
        if cls_embeddings.dim()>2:
            cls_embeddings = cls_embeddings.squeeze(1)

        if self.max_atomic_len is not None:
            B, n_atoms, D = atomic_embeddings.size()
            if n_atoms < self.max_atomic_len:
                pad_size = self.max_atomic_len - n_atoms
                pad = torch.zeros(B, pad_size, D, device=device)
                atomic_embeddings = torch.cat([atomic_embeddings,pad],dim=1)
            elif n_atoms>self.max_atomic_len:
                atomic_embeddings = atomic_embeddings[:, :self.max_atomic_len, :]

        attn_out, _ = self.attn(atomic_embeddings, atomic_embeddings, atomic_embeddings)
        atomic_summary = attn_out.mean(dim=1)
        combined = torch.cat([cls_embeddings, atomic_summary], dim=1)
        x = self.dropout(combined)
        hidden = self.mlp(x)
        return hidden

    def forward(self, cls_embeddings, atomic_embeddings):
        h = self.forward_middle(cls_embeddings, atomic_embeddings)
        h = self.dropout(h)
        out = self.classifier(h)
        return out