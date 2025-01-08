"""
training_structures/inter_and_intra_modality.py

- loss fuction => CrossEntropyLoss

- Inter- and Intra-Modality
- inference => ensemble: alpha*inter + beta*(ensemble of unimodals)
"""

import torch
import torch.nn as nn
import torch.optim as optim

def train_inter_and_intra_modality(
    inter_model,
    unimodal_models,
    train_loaders,
    valid_loaders,
    epochs=5,
    lr=1e-4,
    device="cuda",
    alpha=0.5,
    beta=0.5
):

    params = list(inter_model.parameters())
    for m in unimodal_models:
        params += list(m.parameters())

    optimizer = optim.AdamW(params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_states = None

    inter_model.to(device)
    for m in unimodal_models:
        m.to(device)

    for epoch in range(1, epochs+1):
        inter_model.train()
        for m in unimodal_models:
            m.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch_tuple in zip(*train_loaders):
            inputs_list = []
            label = None
            unimodal_logits_list = []

            for i, (x, y) in enumerate(batch_tuple):
                if i == 0:
                    label = y.to(device, dtype=torch.long)

                # prepare input
                if isinstance(x, tuple):
                    input_ids, attention_mask = x
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    inputs_list.append( (input_ids, attention_mask) )
                else:
                    x_ = x.to(device, dtype=torch.long)
                    inputs_list.append(x_)

            # 1) inter_model forward
            logits_inter = inter_model(inputs_list)

            # 2) unimodal ensemble
            for i, m in enumerate(unimodal_models):
                # unimodal_models[i] inputs_list[i]
                inp = inputs_list[i]
                if isinstance(inp, tuple):
                    out = m(inp[0], inp[1])
                else:
                    out = m(inp)
                unimodal_logits_list.append(out)
            # unimodal ensemble => mean(simple method)
            logits_unimodal = torch.stack(unimodal_logits_list, dim=0).mean(dim=0)

            # 3) combined loss
            loss_inter = criterion(logits_inter, label)
            loss_uni = criterion(logits_unimodal, label)
            loss = alpha * loss_inter + beta * loss_uni

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 4) prediction
            final_logits = 0.5*logits_inter + 0.5*logits_unimodal
            preds = torch.argmax(final_logits, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

        train_acc = correct / total
        val_acc = evaluate_inter_and_intra_modality(inter_model, unimodal_models, valid_loaders, device)
        print(f"[Inter+Intra] Epoch {epoch}/{epochs}, Loss={total_loss/len(train_loaders[0]):.4f}, TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # save model states
            best_states = {
                "inter_model": inter_model.state_dict(),
                "unimodals": [m.state_dict() for m in unimodal_models]
            }

    print(f"[Inter+Intra] Best Val Acc: {best_val_acc:.4f}")
    return best_states

def evaluate_inter_and_intra_modality(inter_model, unimodal_models, loaders, device="cuda"):
    inter_model.eval()
    for m in unimodal_models:
        m.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_tuple in zip(*loaders):
            inputs_list = []
            label = None
            unimodal_logits_list = []

            for i, (x, y) in enumerate(batch_tuple):
                if i == 0:
                    label = y.to(device, dtype=torch.long)
                if isinstance(x, tuple):
                    input_ids, attention_mask = x
                    inputs_list.append( (input_ids.to(device), attention_mask.to(device)) )
                else:
                    inputs_list.append( x.to(device, dtype=torch.long) )

            # inter
            logits_inter = inter_model(inputs_list)

            # unimodal
            for i, m in enumerate(unimodal_models):
                inp = inputs_list[i]
                if isinstance(inp, tuple):
                    out = m(inp[0], inp[1])
                else:
                    out = m(inp)
                unimodal_logits_list.append(out)
            logits_unimodal = torch.stack(unimodal_logits_list, dim=0).mean(dim=0)

            final_logits = 0.5*logits_inter + 0.5*logits_unimodal
            preds = torch.argmax(final_logits, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

    return correct / total

def test_inter_and_intra_modality(inter_model, unimodal_models, test_loaders, device="cuda"):
    acc = evaluate_inter_and_intra_modality(inter_model, unimodal_models, test_loaders, device)
    print(f"[Test: Inter+Intra] Acc: {acc:.4f}")
    return acc