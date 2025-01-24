# data_loader.py

import os
import torch
from torch.utils.data import Dataset
from molfeat.trans.pretrained import PretrainedDGLTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
from unimol_tools import UniMolRepr

class FullFeatureDataset(Dataset):
    def __init__(self, data, is_train=False, is_val=False, is_test=False):
        self.data = data.reset_index(drop=True)
        if is_train:
            cache_path = "./data/train_features.pt"
        elif is_val:
            cache_path = "./data/val_features.pt"
        elif is_test:
            cache_path = "./data/test_features.pt"
        else:
            cache_path = None

        if cache_path and os.path.exists(cache_path):
            self.samples = torch.load(cache_path)
        else:
            self.samples = []
            self.trans_1d = PretrainedHFTransformer(kind="ChemBERTa-77M-MTR", notation="smiles", dtype=float)
            self.trans_2d = PretrainedDGLTransformer(kind="gin_supervised_infomax", notation="smiles", dtype=float)
            self.trans_3d = UniMolRepr(data_type='molecule', remove_hs=False, model_name='unimolv1', model_size='84m')

            for i in range(len(self.data)):
                smi = self.data.iloc[i]["SMILES"]
                y   = self.data.iloc[i]["Y"]
                rep_1d = self.trans_1d(smi)
                if not isinstance(rep_1d, torch.Tensor):
                    rep_1d = torch.tensor(rep_1d, dtype=torch.float32)
                rep_2d = self.trans_2d(smi)
                if not isinstance(rep_2d, torch.Tensor):
                    rep_2d = torch.tensor(rep_2d, dtype=torch.float32)

                rep_3d_all = self.trans_3d.get_repr([smi], return_atomic_reprs=True)
                cls_repr_3d = rep_3d_all['cls_repr'][0]
                atomic_reprs= rep_3d_all['atomic_reprs'][0]
                if not isinstance(cls_repr_3d, torch.Tensor):
                    cls_repr_3d = torch.tensor(cls_repr_3d, dtype=torch.float32)
                atomic_tensor = torch.tensor(atomic_reprs, dtype=torch.float32)

                self.samples.append({
                    "feat_1d": rep_1d,
                    "feat_2d": rep_2d,
                    "feat_3d_cls": cls_repr_3d,
                    "feat_3d_atom": atomic_tensor,
                    "y": y
                })

            if cache_path:
                torch.save(self.samples, cache_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def full_feature_collate_fn(batch):
    feat_1d_list = []
    feat_2d_list = []
    cls_3d_list  = []
    atom_3d_list = []
    y_list       = []

    for b in batch:
        feat_1d_list.append(b["feat_1d"].unsqueeze(0))
        feat_2d_list.append(b["feat_2d"].unsqueeze(0))
        cls_3d_list.append(b["feat_3d_cls"].unsqueeze(0))
        atom_3d_list.append(b["feat_3d_atom"])  # (N,512)
        y_list.append(b["y"])

    feat_1d_tensor = torch.cat(feat_1d_list, dim=0)
    feat_2d_tensor = torch.cat(feat_2d_list, dim=0)
    cls_3d_tensor  = torch.cat(cls_3d_list, dim=0)

    max_len = max(t.size(0) for t in atom_3d_list)
    padded_atoms = []
    for t in atom_3d_list:
        pad_size = max_len - t.size(0)
        if pad_size>0:
            pad = torch.zeros(pad_size, t.size(1), dtype=t.dtype)
            t_padded = torch.cat([t,pad], dim=0)
        else:
            t_padded = t
        padded_atoms.append(t_padded)
    atom_3d_tensor = torch.stack(padded_atoms, dim=0)

    y_tensor = torch.tensor(y_list, dtype=torch.float)

    return {
        "feat_1d": feat_1d_tensor,
        "feat_2d": feat_2d_tensor,
        "feat_3d_cls": cls_3d_tensor,
        "feat_3d_atom": atom_3d_tensor,
        "y": y_tensor
    }