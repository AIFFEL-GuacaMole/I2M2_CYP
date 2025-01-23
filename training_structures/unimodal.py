# unimodal.py

def get_unimodal_logits(model, batch, device="cuda", model_type="1D"):
    if model_type=="1D":
        x = batch["feat_1d"].to(device)
        return model(x)
    elif model_type=="2D":
        x = batch["feat_2d"].to(device)
        return model(x)
    
    elif model_type=="3D":
        cls_3d = batch["feat_3d_cls"].to(device)
        atom_3d= batch["feat_3d_atom"].to(device)
        return model(cls_3d, atom_3d)