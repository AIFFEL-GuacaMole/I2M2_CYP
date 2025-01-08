"""
training_structures/unimodal.py

loss fuction -> CrossEntropyLoss (It will be changed soon).
"""

import torch
import torch.nn as nn
import torch.optim as optim

def train_unimodal(model, train_loader, valid_loader, epochs=5, lr=1e-4, device="cuda"):

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_loader:

            # - ChemBERT -> (input_ids, attention_mask), label
            # - CNN+GRU -> (smiles_idx_tensor), label
            x, y = batch
            y = y.to(device, dtype=torch.long)

            # forward
            if isinstance(x, tuple):
                # BERT input
                input_ids, attention_mask = x
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                # CNN+GRU input
                x = x.to(device, dtype=torch.long)
                logits = model(x)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples

        # validation
        val_acc = evaluate_unimodal(model, valid_loader, device)
        print(f"[Unimodal] Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}")

        # best check point save
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    print(f"[Unimodal] Best Val Acc: {best_val_acc:.4f}")
    return best_state

def evaluate_unimodal(model, loader, device="cuda"):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            y = y.to(device, dtype=torch.long)
            if isinstance(x, tuple):
                # BERT
                input_ids, attention_mask = x
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                logits = model(input_ids, attention_mask)
            else:
                # CNN+GRU
                x = x.to(device, dtype=torch.long)
                logits = model(x)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)
    return total_correct / total_samples

def test_unimodal(model, test_loader, device="cuda"):
    acc = evaluate_unimodal(model, test_loader, device)
    print(f"[Test: Unimodal] Acc: {acc:.4f}")
    return acc