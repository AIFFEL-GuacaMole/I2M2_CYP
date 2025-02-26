# earlystop.py 

import torch

class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose  = verbose
        self.delta    = delta
        self.counter  = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.val_loss_min = val_loss