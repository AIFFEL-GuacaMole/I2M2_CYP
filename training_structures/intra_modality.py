"""
training_structures/intra_modality.py

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryAveragePrecision
from tqdm import tqdm


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


def train_intra_modality(
    unimodal_models,
    train_loader,
    valid_loader,
    epochs=10,
    lr=1e-4,
    weight_decay=1e-4,
    device="cuda",
    loss_type="focal_bce",
    alpha=0.25,
    gamma=2.0,
    early_stopping=True,
    patience=5,
    gradient_clip=5.0
):
    """
    Train ensemble of unimodal models using learned weights for fusion.
    """
    # Initialize models and set to device
    params = []
    for model in unimodal_models:
        model.to(device)
        params += list(model.parameters())

    # Learnable fusion weights
    fusion_weights = nn.Parameter(torch.ones(len(unimodal_models), device=device), requires_grad=True)
    params.append(fusion_weights)

    # Optimizer and scheduler
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss selection
    if loss_type == "focal_bce":
        criterion = FocalBCELoss(alpha=alpha, gamma=gamma)
    elif loss_type == "bce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss type")

    # Best metrics for early stopping
    best_val_auprc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        
        # Training
        for model in unimodal_models:
            model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - Training"):
            x, y = batch
            y = y.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits_list = []
            for model in unimodal_models:
                x_ = x.to(device)
                logits = model(x_)
                logits_list.append(logits)

            # Weighted ensemble
            weights = torch.softmax(fusion_weights, dim=0)
            ensemble_logits = sum(w * l for w, l in zip(weights, logits_list))

            # Loss and backward pass
            loss = criterion(ensemble_logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(params, gradient_clip)
            optimizer.step()

            # Metrics update
            total_loss += loss.item()
            preds = torch.argmax(ensemble_logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Validation 
        val_loss, val_acc, val_auprc = evaluate_intra_modality(
            unimodal_models,
            valid_loader,
            device=device,
            criterion=criterion,
            fusion_weights=fusion_weights
        )

        print(f"Epoch {epoch}/{epochs} - "
              f"TrainLoss: {train_loss:.4f}, TrainAcc: {train_acc:.4f} "
              f"ValLoss: {val_loss:.4f}, ValAcc: {val_acc:.4f}, ValAUPRC: {val_auprc:.4f}")

        # Early stopping
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = [model.state_dict() for model in unimodal_models] + [fusion_weights.clone()]
            patience_counter = 0
        else:
            patience_counter += 1

        if early_stopping and patience_counter >= patience:
            print("[Info] Early stopping triggered.")
            break

    return best_state


def evaluate_intra_modality(models, loader, device="cuda", criterion=None, fusion_weights=None):
    """
    Evaluate ensemble of models.
    """
    for model in models:
        model.to(device)
        model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    # AUPRC metric
    auprc = BinaryAveragePrecision().to(device)
    auprc.reset()

    if fusion_weights is None:
        fusion_weights = torch.ones(len(models), device=device, requires_grad=False)

    with torch.no_grad():
        for batch in loader:
            x, y = batch
            y = y.to(device)

            # Forward pass
            logits_list = []
            for model in models:
                x_ = x.to(device)
                logits = model(x_)
                logits_list.append(logits)

            # Weighted ensemble
            weights = torch.softmax(fusion_weights, dim=0)
            ensemble_logits = sum(w * l for w, l in zip(weights, logits_list))

            # Loss
            if criterion:
                loss = criterion(ensemble_logits, y)
                total_loss += loss.item()

            # Metrics update
            preds = torch.argmax(ensemble_logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            probs = torch.softmax(ensemble_logits, dim=1)[:, 1]
            auprc.update(probs, y)

    accuracy = correct / total
    avg_loss = total_loss / len(loader)
    auprc_score = auprc.compute()

    return avg_loss, accuracy, auprc_score


def test_intra_modality(models, test_loader, fusion_weights=None, device="cuda"):
    """
    Test intra-modality models and report AUPRC.
    """
    _, acc, auprc_score = evaluate_intra_modality(models, test_loader, device, fusion_weights=fusion_weights)
    print(f"[Test: Intra] Acc: {acc:.4f}, AUPRC: {auprc_score:.4f}")
    return acc, auprc_score