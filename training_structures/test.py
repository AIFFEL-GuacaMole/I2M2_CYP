# test.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, average_precision_score
from utils.loss_fn import CustomLoss
from training_structures.unimodal import get_unimodal_logits
from training_structures.intra_modality import get_intra_logits
from training_structures.inter_modality import get_inter_logits
from training_structures.inter_and_intra_modality import get_inter_intra_logits

def test_model(model_or_tuple, test_loader, mode, args, device="cuda"):
    if mode=="intra":
        (m1,m2,m3) = model_or_tuple
        m1.to(device).eval()
        m2.to(device).eval()
        m3.to(device).eval()
    else:
        model_or_tuple.to(device).eval()

    crit = CustomLoss(args.task_type)
    total_loss= 0.0
    total_count=0
    all_labels= []
    all_probs=  []
    all_targs=  []

    for batch in tqdm(test_loader, desc=f"[{mode} Test]"):
        if mode=="unimodal":
            logits = get_unimodal_logits(model_or_tuple,batch,device,args.model_type)
        elif mode=="intra":
            (m1,m2,m3) = model_or_tuple
            logits = get_intra_logits(m1,m2,m3,batch,device)
        elif mode=="inter":
            logits = get_inter_logits(model_or_tuple,batch,device)
        else:
            logits = get_inter_intra_logits(model_or_tuple,batch,device)

        y = batch["y"].to(device)
        if args.task_type=="classification":
            y = y.long()
        else:
            y = y.float()

        loss= crit(logits, y)
        bs  = y.size(0)
        total_loss += loss.item()*bs
        total_count+= bs

        if args.task_type=="classification":
            _, pred_idx = torch.max(logits, dim=1)
            all_labels.extend(pred_idx.cpu().numpy())
            soft = F.softmax(logits, dim=1)
            pos_prob = soft[:,1].detach().cpu().numpy()
            all_probs.extend(pos_prob)
            all_targs.extend(y.cpu().numpy())
        else:
            all_labels.extend(logits.detach().cpu().numpy())
            all_targs.extend(y.cpu().numpy())

    avg_loss = total_loss/total_count

    if args.task_type=="classification":
        acc   = accuracy_score(all_targs, all_labels)
        auprc = average_precision_score(all_targs, all_probs)
        print(f"[{mode} Test] Loss={avg_loss:.4f}, Acc={acc:.4f}, AUPRC={auprc:.4f}")
    else:
        mse= mean_squared_error(all_targs, all_labels)
        mae= mean_absolute_error(all_targs, all_labels)
        print(f"[{mode} Test] Loss={avg_loss:.4f}, MSE={mse:.4f}, MAE={mae:.4f}")