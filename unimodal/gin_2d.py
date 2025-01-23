# gin_2d.py

import torch
import torch.nn as nn

class GIN(nn.Module):
    def __init__(self, hidden_dim=256, dropout_rate=0.4, task_type="classification"):
        super(GIN, self).__init__()
        self.task_type = task_type
        self.embedding_dim = 300
        self.dropout_rate = dropout_rate
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        if self.task_type == "classification":
            self.classifier = nn.Linear(hidden_dim, 2)
        else:
            self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward_middle(self, embeddings):
        if embeddings.dim() > 2:
            embeddings = embeddings.squeeze(1)
        x = self.dropout(embeddings)
        hidden = self.mlp(x)
        return hidden

    def forward(self, embeddings):
        h = self.forward_middle(embeddings)
        h = self.dropout(h)
        out = self.classifier(h)
        return out