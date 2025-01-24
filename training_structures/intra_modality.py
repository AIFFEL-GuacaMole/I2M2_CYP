# intra_modality.py

import torch

def get_intra_logits(model_1d, model_2d, model_3d, batch, device="cuda"):
    x1 = batch["feat_1d"].to(device)
    x2 = batch["feat_2d"].to(device)
    c3 = batch["feat_3d_cls"].to(device)
    a3 = batch["feat_3d_atom"].to(device)

    logit_1d = model_1d(x1)
    logit_2d = model_2d(x2)
    logit_3d = model_3d(c3,a3)

    final_logit = logit_1d + logit_2d + logit_3d
    return final_logit