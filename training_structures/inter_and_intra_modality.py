"""
training_structures/inter_and_intra_modality.py

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryAveragePrecision
from tqdm import tqdm

def focal_bce_loss(logits, targets, alpha=1.0, gamma=2.0):
    bce_loss = nn.CrossEntropyLoss()(logits, targets)
    pt = torch.exp(-bce_loss)
    return alpha * (1 - pt) ** gamma * bce_loss

def train_inter_intra_modality(
    inter_model,
    unimodal_models,
    train_loaders,
    valid_loaders,
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
    Train inter- and intra-modality models using weighted ensemble.
    """
    # Initialize models and set to device
    inter_model.to(device)
    params = list(inter_model.parameters())

    for model in unimodal_models:
        model.to(device)
        params += list(model.parameters())

    # Learnable fusion weights
    fusion_weights = nn.Parameter(torch.ones(len(unimodal_models) + 1, device=device), requires_grad=True)
    params.append(fusion_weights)

    # Optimizer and scheduler
    optimizer = optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss selection
    if loss_type == "focal_bce":
        def criterion(logits, targets):
            return focal_bce_loss(logits, targets, alpha=alpha, gamma=gamma)
    elif loss_type == "bce":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss type")

    # Best metrics for early stopping
    best_val_auprc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):

        inter_model.train()
        for model in unimodal_models:
            model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_tuple in tqdm(zip(*train_loaders), desc=f"Epoch {epoch}/{epochs} - Training"):
            inputs_list = []
            label = None

            for i, (x, y) in enumerate(batch_tuple):
                if i == 0:
                    label = y.to(device)
                if isinstance(x, tuple):
                    inputs_list.append((x[0].to(device), x[1].to(device)))
                else:
                    inputs_list.append(x.to(device))

            optimizer.zero_grad()

            # Forward pass for inter-model
            logits_inter = inter_model(inputs_list)

            # Forward pass for unimodal models
            logits_unimodal = []
            for i, model in enumerate(unimodal_models):
                inp = inputs_list[i]
                if isinstance(inp, tuple):
                    logits = model(inp[0], inp[1])
                else:
                    logits = model(inp)
                logits_unimodal.append(logits)

            # Weighted ensemble
            all_logits = [logits_inter] + logits_unimodal
            weights = torch.softmax(fusion_weights, dim=0)
            ensemble_logits = sum(w * l for w, l in zip(weights, all_logits))

            # Loss and backward pass
            loss = criterion(ensemble_logits, label)
            loss.backward()
            nn.utils.clip_grad_norm_(params, gradient_clip)
            optimizer.step()

            # Metrics update
            total_loss += loss.item()
            preds = torch.argmax(ensemble_logits, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

        scheduler.step()
        train_loss = total_loss / len(train_loaders[0])
        train_acc = correct / total

        val_loss, val_acc, val_auprc = evaluate_inter_intra_modality(
            inter_model, unimodal_models, valid_loaders, device=device, criterion=criterion, fusion_weights=fusion_weights
        )

        print(f"Epoch {epoch}/{epochs} - "
              f"TrainLoss: {train_loss:.4f}, TrainAcc: {train_acc:.4f} "
              f"ValLoss: {val_loss:.4f}, ValAcc: {val_acc:.4f}, ValAUPRC: {val_auprc:.4f}")

        # Early stopping
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = {
                "inter_model": inter_model.state_dict(),
                "unimodals": [model.state_dict() for model in unimodal_models],
                "fusion_weights": fusion_weights.clone()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if early_stopping and patience_counter >= patience:
            print("[Info] Early stopping triggered.")
            break

    return best_state

def evaluate_inter_intra_modality(inter_model, unimodal_models, loaders, device="cuda", criterion=None, fusion_weights=None):
    """
    Evaluate ensemble of inter- and intra-modality models.
    """
    inter_model.eval()
    inter_model.to(device)
    for model in unimodal_models:
        model.eval()
        model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    auprc = BinaryAveragePrecision().to(device)
    auprc.reset()

    if fusion_weights is None:
        fusion_weights = torch.ones(len(unimodal_models) + 1, device=device, requires_grad=False)

    with torch.no_grad():
        for batch_tuple in loaders:
            inputs_list = []
            label = None

            for i, (x, y) in enumerate(batch_tuple):
                if i == 0:
                    label = y.to(device)
                if isinstance(x, tuple):
                    inputs_list.append((x[0].to(device), x[1].to(device)))
                else:
                    inputs_list.append(x.to(device))

            # Forward pass for inter-model
            logits_inter = inter_model(inputs_list)

            # Forward pass for unimodal models
            logits_unimodal = []
            for i, model in enumerate(unimodal_models):
                inp = inputs_list[i]
                if isinstance(inp, tuple):
                    logits = model(inp[0], inp[1])
                else:
                    logits = model(inp)
                logits_unimodal.append(logits)

            # Weighted ensemble
            all_logits = [logits_inter] + logits_unimodal
            weights = torch.softmax(fusion_weights, dim=0)
            ensemble_logits = sum(w * l for w, l in zip(weights, all_logits))

            # Loss
            if criterion:
                loss = criterion(ensemble_logits, label)
                total_loss += loss.item()

            # Metrics update
            preds = torch.argmax(ensemble_logits, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
            probs = torch.softmax(ensemble_logits, dim=1)[:, 1]
            auprc.update(probs, label)

    accuracy = correct / total
    avg_loss = total_loss / len(loaders[0])
    auprc_score = auprc.compute()

    return avg_loss, accuracy, auprc_score

def test_inter_intra_modality(inter_model, unimodal_models, test_loaders, fusion_weights=None, device="cuda"):
    """
    Test inter- and intra-modality models and return predictions along with metrics.
    """
    all_preds = []
    all_labels = []
    
    inter_model.eval()
    for model in unimodal_models:
        model.eval()

    with torch.no_grad():
        for loader in test_loaders:
            for inputs, labels in loader:  # Include labels for evaluation
                inputs = [input_data.to(device) for input_data in inputs]
                labels = labels.to(device)
                
                # Get predictions from the inter-model
                outputs = inter_model(*inputs)
                preds = torch.softmax(outputs, dim=1)[:, 1]  # Assuming binary classification, extract probabilities
                
                # Collect predictions and labels
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
    
    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Compute metrics
    _, acc, auprc_score = evaluate_inter_intra_modality(
        inter_model, unimodal_models, test_loaders, device, fusion_weights=fusion_weights
    )
    
    print(f"[Test: Inter+Intra] Acc: {acc:.4f}, AUPRC: {auprc_score:.4f}")
    return all_preds, all_labels, acc, auprc_score