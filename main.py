"""
main.py

process:
1) argparse -> model_type (unimodal/inter/intra/inter_intra), train/test option parsing
2) data_loader.py -> get_cyp2c19_dataloaders
3) load model (unimodal / inter / etc.)
4) train or test
"""

import argparse
import torch
import os
import torch.nn as nn 

# DataLoader
from datasets.data_loader import get_cyp2c19_dataloaders

# Unimodal models
from unimodal.chembert_binary_classifier import ChemBERTBinaryClassifier
from unimodal.cnn_gru_binary_classifier import CNNGRUBinaryClassifier

# Training structures
from training_structures.unimodal import train_unimodal, test_unimodal
from training_structures.intra_modality import train_intra_modality, test_intra_modality
from training_structures.inter_modality import (
    InterModalModel,
    train_inter_modality,
    test_inter_modality
)
from training_structures.inter_and_intra_modality import (
    train_inter_and_intra_modality,
    test_inter_and_intra_modality
)

# Fusion - fusion method (Concat, LowRankTensorFusion, etc.)
from common_fusions.fusions import ConcatFusion 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="unimodal",
                        choices=["unimodal", "intra", "inter", "inter_intra"],
                        help="Choose which model type to train/test.")
    parser.add_argument("--unimodal_arch", type=str, default="chembert",
                        choices=["chembert", "cnn_gru"],
                        help="For unimodal, choose architecture.")
    parser.add_argument("--train", action="store_true", help="If set, run training")
    parser.add_argument("--test", action="store_true", help="If set, run testing")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing cyp2c19_{train,valid,test}.csv")
    parser.add_argument("--save_dir", type=str, default="./ckpts",
                        help="Directory to save model checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Running on device={device}")

    # Load Data
    train_loader, valid_loader, test_loader = get_cyp2c19_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle_train=True,
        is_classification=True
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # 1. UNIMODAL
    if args.model_type == "unimodal":

        # Create unimodal model
        if args.unimodal_arch == "chembert":
            model = ChemBERTBinaryClassifier(
                model_name_or_path="seyonec/PubChem10M_SMILES_BPE_450k",
                num_classes=2,
                dropout_prob=0.1
            )
            model_name = "unimodal_chembert.pt"
        else:
            model = CNNGRUBinaryClassifier(
                # vocab_size, emb_dim, num_filters_list, kernel_sizes, hidden_dim, num_layers, num_classes
                vocab_size=8000, 
                emb_dim=128,
                num_filters_list=[64, 128],
                kernel_sizes=[3, 5],
                hidden_dim=128,
                num_layers=2,
                num_classes=2
            )
            model_name = "unimodal_cnn_gru.pt"

        save_path = os.path.join(args.save_dir, model_name)

        # train
        if args.train:
            best_state = train_unimodal(model, train_loader, valid_loader,
                                        epochs=args.epochs, lr=args.lr, device=device)
            if best_state is not None:
                torch.save(best_state, save_path)
                print(f"[Info] Saved unimodal best state to {save_path}")

        # test
        if args.test:
            if os.path.exists(save_path):
                model.load_state_dict(torch.load(save_path))
                print(f"[Info] Loaded unimodal state from {save_path}")
            else:
                print(f"[Warning] No checkpoint found at {save_path}. Testing untrained model.")
            test_unimodal(model, test_loader, device=device)


    # 2. INTRA MODALITY
    elif args.model_type == "intra":

        # chembert + cnn_gru
        model1 = ChemBERTBinaryClassifier()
        model2 = CNNGRUBinaryClassifier()

        save_path1 = os.path.join(args.save_dir, "intra_chembert_1.pt")
        save_path2 = os.path.join(args.save_dir, "intra_cnn_gru_1.pt")

        models = [model1, model2]

        if args.train:
            best_states = train_intra_modality(models, train_loader, valid_loader,
                                               epochs=args.epochs, lr=args.lr, device=device)
            if best_states is not None:
                torch.save(best_states[0], save_path1)
                torch.save(best_states[1], save_path2)

        if args.test:
            if os.path.exists(save_path1):
                model1.load_state_dict(torch.load(save_path1))
            if os.path.exists(save_path2):
                model2.load_state_dict(torch.load(save_path2))
            test_intra_modality(models, test_loader, device=device)


    # 3. INTER MODALITY
    elif args.model_type == "inter":

        # chembert + cnn_gru -> fusion -> head
        chembert_model = ChemBERTBinaryClassifier()
        cnn_model = CNNGRUBinaryClassifier()

        encoders = [chembert_model, cnn_model]
        fusion = ConcatFusion()
        head = nn.Linear(128 * 2, 2)

        inter_model = InterModalModel(encoders, fusion, head)

        # each loader of modality data - each transform data loader
        train_loaders_list = [train_loader, train_loader]
        valid_loaders_list = [valid_loader, valid_loader]
        test_loaders_list  = [test_loader,  test_loader]

        save_path = os.path.join(args.save_dir, "inter_model.pt")

        if args.train:
            best_state = train_inter_modality(inter_model, train_loaders_list, valid_loaders_list,
                                              epochs=args.epochs, lr=args.lr, device=device)
            if best_state is not None:
                torch.save(best_state, save_path)

        if args.test:
            if os.path.exists(save_path):
                inter_model.load_state_dict(torch.load(save_path))
            test_inter_modality(inter_model, test_loaders_list, device=device)


    # 4. INTER - INTRA MODALITY
    elif args.model_type == "inter_intra":

        # inter_model (bert_for_inter + cnn_for_inter) + unimodal_models=[bert_only, cnn_only]
        bert_only = ChemBERTBinaryClassifier()
        cnn_only  = CNNGRUBinaryClassifier()

        bert_for_inter = ChemBERTBinaryClassifier()  
        cnn_for_inter  = CNNGRUBinaryClassifier()

        fusion = ConcatFusion()
        head = nn.Linear(128*2, 2)
        inter_model = InterModalModel([bert_for_inter, cnn_for_inter], fusion, head)

        train_loaders_list = [train_loader, train_loader]
        valid_loaders_list = [valid_loader, valid_loader]
        test_loaders_list  = [test_loader,  test_loader]

        save_path_inter     = os.path.join(args.save_dir, "inter_intra_inter.pt")
        save_path_uni_bert  = os.path.join(args.save_dir, "inter_intra_uni_bert.pt")
        save_path_uni_cnn   = os.path.join(args.save_dir, "inter_intra_uni_cnn.pt")

        if args.train:
            best_states = train_inter_and_intra_modality(
                inter_model,
                [bert_only, cnn_only],
                train_loaders_list,
                valid_loaders_list,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                alpha=0.5,
                beta=0.5
            )
            if best_states is not None:
                torch.save(best_states["inter_model"], save_path_inter)
                torch.save(best_states["unimodals"][0], save_path_uni_bert)
                torch.save(best_states["unimodals"][1], save_path_uni_cnn)

        if args.test:
            if os.path.exists(save_path_inter):
                inter_model.load_state_dict(torch.load(save_path_inter))
            if os.path.exists(save_path_uni_bert):
                bert_only.load_state_dict(torch.load(save_path_uni_bert))
            if os.path.exists(save_path_uni_cnn):
                cnn_only.load_state_dict(torch.load(save_path_uni_cnn))

            test_inter_and_intra_modality(
                inter_model, [bert_only, cnn_only],
                test_loaders_list, device=device
            )


if __name__ == "__main__":
    main()