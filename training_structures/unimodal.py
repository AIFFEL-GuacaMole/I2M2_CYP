"""
training_structures/unimodal.py

"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import BinaryAveragePrecision

# Focal + BCE Loss 
class FocalBCELoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce_loss = nn.CrossEntropyLoss()(logits, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss


def train_unimodal(
    model,
    train_loader,
    valid_loader,
    epochs=5,
    lr=1e-4,
    weight_decay=1e-4,
    device="cuda",
    loss_type="focal_bce",
    alpha=0.25,
    gamma=2.0
):
    """
    Train a unimodal model (ChemBERT or CNN+GRU) using CrossEntropyLoss or Focal+BCE Loss.
    """
    model = model.to(device)

    # Loss
    if loss_type == "focal_bce":
        criterion = FocalBCELoss(alpha=alpha, gamma=gamma)
    elif loss_type == "bce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss type")

    # AdamW + weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_val_auprc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):

        # Training
        
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

        scheduler.step()

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples

        # Validation
        val_acc, val_loss, val_auprc = evaluate_unimodal(
            model=model,
            loader=valid_loader,
            device=device,
            criterion=criterion
        )

        print(f"[Unimodal] Epoch {epoch}/{epochs} "
              f"- TrainLoss: {train_loss:.4f}, TrainAcc: {train_acc:.4f} "
              f"- ValLoss: {val_loss:.4f}, ValAcc: {val_acc:.4f}, ValAUPRC: {val_auprc:.4f}")

        # Best checkpoint 갱신
        if val_auprc > best_val_auprc:
            best_val_acc = val_acc
            best_val_auprc = val_auprc
            best_state = model.state_dict()

    print(f"[Unimodal] Best Val Acc: {best_val_acc:.4f}, Best Val AUPRC: {best_val_auprc:.4f}")
    return best_state


def evaluate_unimodal(model, loader, device="cuda", criterion=None):
    """
    Evaluate model on a given DataLoader, returning accuracy, loss, and AUPRC.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    # Initialize AUPRC metric
    auprc_metric = BinaryAveragePrecision().to(device)

    compute_loss = (criterion is not None)

    with torch.no_grad():
        for batch in loader:
            x, y = batch

            # Ensure inputs and labels are on the correct device
            if isinstance(x, tuple):
                input_ids, attention_mask = x
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                x = (input_ids, attention_mask)
            else:
                x = x.to(device)

            y = y.to(device)

            # Forward pass
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            # Update metrics
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

            # Update AUPRC
            probs = torch.softmax(logits, dim=1)[:, 1]  
            auprc_metric.update(probs, y)

            if compute_loss:
                loss = criterion(logits, y)
                total_loss += loss.item()

    # Compute AUPRC
    auprc_score = auprc_metric.compute()
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = (total_loss / len(loader)) if (compute_loss and len(loader) > 0) else 0.0

    return accuracy, avg_loss, auprc_score


def test_unimodal(model, test_loader, device="cuda"):
    """
    Test a unimodal model on the test dataset and output AUPRC.
    """
    print("[Info] Testing the model...")

    # Ensure model is on the correct device
    model.to(device)

    # Evaluate model
    test_acc, _, test_auprc = evaluate_unimodal(
        model=model,
        loader=test_loader,
        device=device,
        criterion=None  
    )
    print(f"[Test: Unimodal] Acc: {test_acc:.4f},  AUPRC: {test_auprc:.4f}")
    return test_acc, test_auprc