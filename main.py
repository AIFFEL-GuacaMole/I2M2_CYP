# main.py

import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader

from datasets.datasets import DataProcessor
from datasets.data_loader import FullFeatureDataset, full_feature_collate_fn

from training_structures.train import train_model
from training_structures.test import test_model

from unimodal.chemberta1d import ChemBERTaModel
from unimodal.gin_2d import GIN
from unimodal.unimol_3d import UniMol_3D

from training_structures.inter_modality import InterModel
from training_structures.inter_and_intra_modality import InterAndIntraModel

from common_fusions.fusions import ConcatFusion, CrossAttentionFusion

def parse_args():
    parser = argparse.ArgumentParser(description="Main script for I2M2 project")
    parser.add_argument('--phase', type=str, default='train', choices=['train','test'])
    parser.add_argument('--mode', type=str, default='unimodal', 
                        choices=['unimodal','intra','inter','inter_intra'])
    parser.add_argument('--task_type', type=str, default='classification', 
                        choices=['classification','regression'])
    parser.add_argument('--model_type', type=str, default='1D', 
                        choices=['1D','2D','3D'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--sweep_config', type=str, default='sweep_config.yaml')
    parser.add_argument('--output_dir', type=str, default='./data')
    return parser.parse_args()

def build_unimodal_model(model_type, task_type):
    if model_type=='1D':
        return ChemBERTaModel(task_type=task_type)
    elif model_type=='2D':
        return GIN(task_type=task_type)
    else:
        return UniMol_3D(task_type=task_type)

def build_intra_models(task_type):
    m1 = ChemBERTaModel(task_type=task_type)
    m2 = GIN(task_type=task_type)
    m3 = UniMol_3D(task_type=task_type)
    return (m1, m2, m3)

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.sweep and args.phase=='train':
        from sweep import run_sweep
        run_sweep(args)
        return

    processor = DataProcessor("./data/train_val.csv","./data/test.csv")
    train_data, val_data, test_data = processor.process_and_save(args.output_dir)

    if args.mode=='unimodal':
        model = build_unimodal_model(args.model_type, args.task_type)

        if args.phase=='train':
            train_ds = FullFeatureDataset(train_data, is_train=True)
            val_ds   = FullFeatureDataset(val_data,   is_val=True)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                                      shuffle=True, collate_fn=full_feature_collate_fn)
            val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                                      shuffle=True, collate_fn=full_feature_collate_fn)

            train_model(model, train_loader, val_loader, 'unimodal', args, device)
        else:
            test_ds  = FullFeatureDataset(test_data, is_test=True)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                     shuffle=False, collate_fn=full_feature_collate_fn)
            ckpt = f"./ckpt/best_unimodal_{args.model_type}_{args.task_type}.pth"
            model.load_state_dict(torch.load(ckpt, map_location='cpu'))
            test_model(model, test_loader, 'unimodal', args, device)

    elif args.mode=='intra':
        (m1,m2,m3) = build_intra_models(args.task_type)
        if args.phase=='train':
            train_ds = FullFeatureDataset(train_data, is_train=True)
            val_ds   = FullFeatureDataset(val_data,   is_val=True)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      shuffle=True, collate_fn=full_feature_collate_fn)
            val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                                      shuffle=False, collate_fn=full_feature_collate_fn)

            train_model((m1,m2,m3), train_loader, val_loader, 'intra', args, device)
        else:
            test_ds  = FullFeatureDataset(test_data, is_test=True)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                     shuffle=False, collate_fn=full_feature_collate_fn)

            path1 = f"./ckpt/best_intra_1d_{args.task_type}.pth"
            path2 = f"./ckpt/best_intra_2d_{args.task_type}.pth"
            path3 = f"./ckpt/best_intra_3d_{args.task_type}.pth"
            m1.load_state_dict(torch.load(path1, map_location='cpu'))
            m2.load_state_dict(torch.load(path2, map_location='cpu'))
            m3.load_state_dict(torch.load(path3, map_location='cpu'))

            test_model((m1,m2,m3), test_loader, 'intra', args, device)

    elif args.mode=='inter':
        fusion = CrossAttentionFusion(embed_dim=256, num_heads=4, out_dim=256)
        m1 = ChemBERTaModel(task_type=args.task_type)
        m2 = GIN(task_type=args.task_type)
        m3 = UniMol_3D(task_type=args.task_type)
        inter_model = InterModel(m1,m2,m3,fusion,out_dim=256)

        if args.phase=='train':
            train_ds = FullFeatureDataset(train_data, is_train=True)
            val_ds   = FullFeatureDataset(val_data, is_val=True)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      shuffle=True, collate_fn=full_feature_collate_fn)
            val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                                      shuffle=False, collate_fn=full_feature_collate_fn)

            train_model(inter_model, train_loader, val_loader, 'inter', args, device)
        else:
            test_ds = FullFeatureDataset(test_data, is_test=True)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                     shuffle=False, collate_fn=full_feature_collate_fn)

            path = f"./ckpt/best_inter_{args.task_type}.pth"
            inter_model.load_state_dict(torch.load(path, map_location='cpu'))
            test_model(inter_model, test_loader, 'inter', args, device)

    elif args.mode=='inter_intra':
        fusion_mod = CrossAttentionFusion(embed_dim=256, num_heads=4, out_dim=256)
        m1 = ChemBERTaModel(task_type=args.task_type)
        m2 = GIN(task_type=args.task_type)
        m3 = UniMol_3D(task_type=args.task_type)
        model = InterAndIntraModel(m1,m2,m3,fusion_mod,out_dim=256)

        if args.phase=='train':
            train_ds = FullFeatureDataset(train_data, is_train=True)
            val_ds   = FullFeatureDataset(val_data,   is_val=True)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                      shuffle=True, collate_fn=full_feature_collate_fn)
            val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                                      shuffle=False, collate_fn=full_feature_collate_fn)

            train_model(model, train_loader, val_loader, 'inter_intra', args, device)
        else:
            test_ds = FullFeatureDataset(test_data, is_test=True)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                     shuffle=False, collate_fn=full_feature_collate_fn)

            path = f"./ckpt/best_inter_intra_{args.task_type}.pth"
            model.load_state_dict(torch.load(path, map_location='cpu'))
            test_model(model, test_loader, 'inter_intra', args, device)

if __name__=="__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.getLogger("unimol_tools").setLevel(logging.WARNING)
    main()