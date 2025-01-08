"""
training_structures/inter_modality.py

loss fuction -> CrossEntropyLoss (It will be changed soon).
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Multi-Modal Model: encoders + fusion + head
class InterModalModel(nn.Module):

    def __init__(self, encoders, fusion, head):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fusion = fusion
        self.head = head

    def forward(self, inputs_list):

        feats = []
        for i, enc in enumerate(self.encoders):
            inp = inputs_list[i]
            if isinstance(inp, tuple):
                input_ids, attention_mask = inp
                feats.append(enc(input_ids, attention_mask))
            else:
                feats.append(enc(inp))
        fused = self.fusion(feats)  # e.g. Concat, LowRankTensorFusion
        logits = self.head(fused)
        return logits

def train_inter_modality(model, train_loaders, valid_loaders, epochs=5, lr=1e-4, device="cuda"):

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_tuple in zip(*train_loaders):
            # batch_tuple: ( (x_bert, y_bert), (x_cnn, y_cnn) )
            inputs_list = []
            label = None
            for i, (x, y) in enumerate(batch_tuple):
                if i == 0:
                    label = y.to(device, dtype=torch.long)
                # x is either (input_ids, attention_mask) or tensor
                if isinstance(x, tuple):
                    input_ids, attention_mask = x
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    inputs_list.append( (input_ids, attention_mask) )
                else:
                    x_ = x.to(device, dtype=torch.long)
                    inputs_list.append(x_)

            logits = model(inputs_list)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

        train_acc = correct / total
        val_acc = evaluate_inter_modality(model, valid_loaders, device)

        print(f"[Inter] Epoch {epoch}/{epochs} - Loss: {total_loss/len(train_loaders[0]):.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    print(f"[Inter] Best Val Acc: {best_val_acc:.4f}")
    return best_state

def evaluate_inter_modality(model, loaders, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_tuple in zip(*loaders):
            inputs_list = []
            label = None
            for i, (x, y) in enumerate(batch_tuple):
                if i == 0:
                    label = y.to(device, dtype=torch.long)
                if isinstance(x, tuple):
                    input_ids, attention_mask = x
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    inputs_list.append((input_ids, attention_mask))
                else:
                    x_ = x.to(device, dtype=torch.long)
                    inputs_list.append(x_)

            logits = model(inputs_list)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
    return correct / total

def test_inter_modality(model, test_loaders, device="cuda"):
    acc = evaluate_inter_modality(model, test_loaders, device)
    print(f"[Test: Inter] Acc: {acc:.4f}")
    return acc