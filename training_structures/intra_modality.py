"""
training_structures/intra_modality.py

loss fuction -> CrossEntropyLoss (It will be changed soon).
"""

import torch
import torch.nn as nn
import torch.optim as optim

def train_intra_modality(models, train_loader, valid_loader, epochs=5, lr=1e-4, device="cuda"):
    """
    models: ChemBERT + CNN+GRU 
    """
    params = []
    for m in models:
        m.to(device)
        params += list(m.parameters())

    optimizer = optim.AdamW(params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_states = None

    for epoch in range(1, epochs + 1):
        for m in models:
            m.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            x, y = batch
            y = y.to(device, dtype=torch.long)
            logits_list = []
            for m in models:
                if isinstance(x, tuple):
                    # BERT
                    input_ids, attention_mask = x
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    out = m(input_ids, attention_mask)
                else:
                    # CNN+GRU
                    x_ = x.to(device, dtype=torch.long)
                    out = m(x_)
                logits_list.append(out)

            # ensemble - mean(simple method)
            ensemble_logits = torch.stack(logits_list, dim=0).mean(dim=0)

            loss = criterion(ensemble_logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(ensemble_logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        val_acc = evaluate_intra_modality(models, valid_loader, device)

        print(f"[Intra] Epoch {epoch}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_states = [m.state_dict() for m in models]

    print(f"[Intra] Best Val Acc: {best_val_acc:.4f}")
    return best_states

def evaluate_intra_modality(models, loader, device="cuda"):
    for m in models:
        m.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            y = y.to(device, dtype=torch.long)
            logits_list = []
            for m in models:
                if isinstance(x, tuple):
                    input_ids, attention_mask = x
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    out = m(input_ids, attention_mask)
                else:
                    x_ = x.to(device, dtype=torch.long)
                    out = m(x_)
                logits_list.append(out)
            ensemble_logits = torch.stack(logits_list, dim=0).mean(dim=0)
            preds = torch.argmax(ensemble_logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def test_intra_modality(models, test_loader, device="cuda"):
    acc = evaluate_intra_modality(models, test_loader, device)
    print(f"[Test: Intra] Acc: {acc:.4f}")
    return acc