# loss_fn.py

import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, task_type="classification"):
        super(CustomLoss, self).__init__()
        self.task_type = task_type
        if self.task_type=="classification":
            self.criterion = nn.CrossEntropyLoss()
        elif self.task_type=="regression":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task_type {self.task_type}")
    def forward(self, logits, targets):
        return self.criterion(logits, targets)