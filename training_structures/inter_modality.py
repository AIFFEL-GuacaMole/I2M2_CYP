# inter_modality.py

import torch
import torch.nn as nn

class InterModel(nn.Module):
    def __init__(self, model_1d, model_2d, model_3d, fusion_module, out_dim=256, task_type="classification"):
        super(InterModel, self).__init__()
        self.m1 = model_1d
        self.m2 = model_2d
        self.m3 = model_3d
        self.fusion = fusion_module
        self.task_type = task_type

        self.mlp_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

        if self.task_type == "classification":
            self.final_fc = nn.Linear(out_dim, 2) 
        else:
            self.final_fc = nn.Linear(out_dim, 1) 

    def forward(self, batch, device="cuda"):
        x1 = batch["feat_1d"].to(device)
        x2 = batch["feat_2d"].to(device)
        c3 = batch["feat_3d_cls"].to(device)
        a3 = batch["feat_3d_atom"].to(device)

        f1 = self.m1.forward_middle(x1)
        f2 = self.m2.forward_middle(x2)
        f3 = self.m3.forward_middle(c3, a3)

        fused = self.fusion(f1, f2, f3) 

        post = self.mlp_head(fused)    

        logit = self.final_fc(post)    

        return logit, post, [f1,f2,f3]

def get_inter_logits(inter_model, batch, device="cuda"):
    logit, fused, sub_embs = inter_model(batch, device=device)
    return logit