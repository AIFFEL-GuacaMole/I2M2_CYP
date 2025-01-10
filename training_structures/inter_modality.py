"""
training_structures/inter_modality.py

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryAveragePrecision



class FocalBCELoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = nn.CrossEntropyLoss()(logits, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss


class InterModalModel(nn.Module):
    def __init__(self, encoders, projectors, fusion, hidden_dim=128, dropout_prob=0.5, use_batch_norm=True):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.projectors = nn.ModuleList(projectors)
        self.fusion = fusion

        # MLP head with BatchNorm and Dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 2)  # Output layer for binary classification
        )

    def forward(self, inputs_list):
        feats = []
        for i, encoder in enumerate(self.encoders):
            inp = inputs_list[i]
            if isinstance(inp, tuple):  # For BERT-like encoders
                input_ids, attention_mask = inp
                features = encoder(input_ids, attention_mask)
            else:  # For other encoders (e.g., CNN-GRU)
                features = encoder(inp)

            # Project to unified dimension
            proj_feat = self.projectors[i](features)
            feats.append(proj_feat)

        # Fusion of features
        fused = self.fusion(feats)

        # Pass through MLP head
        logits = self.head(fused)

        return logits


def load_checkpoint(model, checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"[Info] Model loaded from {checkpoint_path}")

def train_inter_modality(model, train_loaders, valid_loaders, epochs=10, lr=1e-4, weight_decay=1e-4, device="cuda", 
                         loss_type="focal_bce", alpha=0.25, gamma=2.0, early_stopping=True, patience=5):
    model.to(device)

    # Loss selection
    if loss_type == "focal_bce":
        criterion = FocalBCELoss(alpha=alpha, gamma=gamma)
    elif loss_type == "bce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss type")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auprc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _train_epoch(model, train_loaders, optimizer, criterion, device)
        val_loss, val_acc, val_auprc = _evaluate_epoch(model, valid_loaders, device, criterion)

        print(f"[Inter] Epoch {epoch}/{epochs} - TrainLoss: {train_loss:.4f}, TrainAcc: {train_acc:.4f}, "
              f"ValLoss: {val_loss:.4f}, ValAcc: {val_acc:.4f}, ValAUPRC: {val_auprc:.4f}")

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if early_stopping and patience_counter >= patience:
            print("[Info] Early stopping triggered.")
            break

        scheduler.step()

    print(f"[Inter] Best Val AUPRC: {best_val_auprc:.4f}")
    return best_state


def _train_epoch(model, train_loaders, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_tuple in zip(*train_loaders):
        inputs_list, label = _process_batch(batch_tuple, device)

        logits = model(inputs_list)
        loss = criterion(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)

    train_loss = total_loss / len(train_loaders[0])
    train_acc = correct / total
    return train_loss, train_acc


def _evaluate_epoch(model, loaders, device, criterion):
    model.to(device)
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    auprc = BinaryAveragePrecision().to(device)
    auprc.reset()

    with torch.no_grad():
        for batch_tuple in zip(*loaders):
            inputs_list, label = _process_batch(batch_tuple, device)

            logits = model(inputs_list)
            preds = torch.argmax(logits, dim=1)

            if criterion:
                loss = criterion(logits, label)
                total_loss += loss.item()

            correct += (preds == label).sum().item()
            total += label.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]
            auprc.update(probs, label)

    accuracy = correct / total
    avg_loss = total_loss / len(loaders[0])
    auprc_score = auprc.compute()
    return avg_loss, accuracy, auprc_score


def _process_batch(batch_tuple, device):
    inputs_list, label = [], None
    for i, (x, y) in enumerate(batch_tuple):
        if i == 0:
            label = y.to(device, dtype=torch.long)
        if isinstance(x, tuple):
            input_ids, attention_mask = x
            inputs_list.append((input_ids.to(device), attention_mask.to(device)))
        else:
            inputs_list.append(x.to(device, dtype=torch.long))
    return inputs_list, label


def test_inter_modality(model, test_loaders, device="cuda"):
    _, acc, auprc_score = _evaluate_epoch(model, test_loaders, device, None)
    print(f"[Test: Inter] Acc: {acc:.4f}, AUPRC: {auprc_score:.4f}")
    return acc, auprc_score