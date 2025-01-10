"""
main.py

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
    train_inter_intra_modality,
    test_inter_intra_modality
)

# Fusion methods
from common_fusions.fusions import (
    ConcatFusion, DynamicWeightedFusion, AttentionFusion, MultiplicativeFusion,
    ResidualFusion, LowRankTensorFusion, LateFusion
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="unimodal",
                        choices=["unimodal", "inter","intra","inter_intra"],
                        help="Choose which model type to train/test.")
    parser.add_argument("--unimodal_arch", type=str, default="chembert",
                        choices=["chembert", "cnn_gru"],
                        help="For unimodal, choose architecture.")
    parser.add_argument("--train", action="store_true", help="If set, run training")
    parser.add_argument("--test", action="store_true", help="If set, run testing")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing cyp2c19_{train,valid,test}.csv")
    parser.add_argument("--save_dir", type=str, default="./ckpts",
                        help="Directory to save model checkpoints")
    parser.add_argument("--fusion_type", type=str, default="attention",
                        choices=["concat", "lowrank", "dynamic", "attention", "multiplicative", "residual", "late"],
                        help="Which fusion to use in inter modality")
    parser.add_argument("--loss_type", type=str, default="focal_bce",
                        choices=["focal_bce", "bce"],
                        help="Loss function type for training.")
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

    # 1. Unimodal
    if args.model_type == "unimodal":
        if args.unimodal_arch == "chembert":
            model = ChemBERTBinaryClassifier(
                model_name_or_path="seyonec/PubChem10M_SMILES_BPE_450k",
                num_classes=2,
                dropout_prob=0.1
            )
            model_name = "unimodal_chembert.pt"
        else:
            model = CNNGRUBinaryClassifier(
                vocab_size=800,
                emb_dim=128,
                num_filters_list=[64, 128],
                kernel_sizes=[3, 5],
                hidden_dim=128,
                num_layers=2,
                num_classes=2
            )
            model_name = "unimodal_cnn_gru.pt"

        save_path = os.path.join(args.save_dir, model_name)

        if args.train:
            best_state = train_unimodal(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                epochs=args.epochs,
                lr=args.lr,
                device=device
            )
            if best_state:
                torch.save(best_state, save_path)
                print(f"[Info] Saved unimodal best state to {save_path}")

        if args.test:
            if os.path.exists(save_path):
                model.load_state_dict(torch.load(save_path))
                print(f"[Info] Loaded unimodal state from {save_path}")
            else:
                print(f"[Warning] No checkpoint found at {save_path}. Testing untrained model.")
            test_unimodal(model, test_loader, device=device)

    # 2. Intra Modality
    elif args.model_type == "intra":
        # Load Unimodal Checkpoints
        chembert_path = os.path.join(args.save_dir, "unimodal_chembert.pt")
        cnn_gru_path = os.path.join(args.save_dir, "unimodal_cnn_gru.pt")

        if not os.path.exists(chembert_path) or not os.path.exists(cnn_gru_path):
            print("[Error] Unimodal checkpoints not found. Train unimodal models first.")
            return

        model1 = ChemBERTBinaryClassifier(
            model_name_or_path="seyonec/PubChem10M_SMILES_BPE_450k",
            num_classes=2,
            dropout_prob=0.4
        )
        model1.load_state_dict(torch.load(chembert_path))
        
        model2 = CNNGRUBinaryClassifier(
            vocab_size=800,
            emb_dim=128,
            num_filters_list=[64, 128],
            kernel_sizes=[3, 5],
            hidden_dim=128,
            num_layers=2,
            num_classes=2
        )
        model2.load_state_dict(torch.load(cnn_gru_path))

        save_path1 = os.path.join(args.save_dir, "intra_chembert.pt")
        save_path2 = os.path.join(args.save_dir, "intra_cnn_gru.pt")

        models = [model1, model2]

        if args.train:
            best_states = train_intra_modality(
                unimodal_models=models,
                train_loader=train_loader,
                valid_loader=valid_loader,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                loss_type=args.loss_type
            )
            if best_states:
                torch.save(best_states[0], save_path1)
                torch.save(best_states[1], save_path2)
                print("[Info] Saved intra modality best states")

        if args.test:
            if os.path.exists(save_path1):
                model1.load_state_dict(torch.load(save_path1))
            if os.path.exists(save_path2):
                model2.load_state_dict(torch.load(save_path2))
            test_intra_modality(models, test_loader, device=device)

    # 3. Inter Modality
    elif args.model_type == "inter":
        # load Unimodal checkpoitns
        chembert_path = os.path.join(args.save_dir, "unimodal_chembert.pt")
        cnn_gru_path = os.path.join(args.save_dir, "unimodal_cnn_gru.pt")

        if not os.path.exists(chembert_path) or not os.path.exists(cnn_gru_path):
            raise FileNotFoundError("[Error] Unimodal checkpoints not found. Train unimodal models first.")

        chembert_model = ChemBERTBinaryClassifier(return_features=True)
        chembert_model.load_state_dict(torch.load(chembert_path))
        print(f"[Info] Loaded ChemBERT checkpoint from {chembert_path}")

        cnn_model = CNNGRUBinaryClassifier(
            vocab_size=800,
            emb_dim=128,
            num_filters_list=[64, 128],
            kernel_sizes=[3, 5],
            hidden_dim=128,
            num_layers=2,
            num_classes=2,
            return_features=True
        )
        cnn_model.load_state_dict(torch.load(cnn_gru_path))
        print(f"[Info] Loaded CNN-GRU checkpoint from {cnn_gru_path}")

        # Encoders 
        encoders = [chembert_model, cnn_model]

        # Projectors
        projector_bert = nn.Linear(768, 128)
        projector_cnn = nn.Linear(256, 128)
        projectors = [projector_bert, projector_cnn]

        # Fusion 
        if args.fusion_type == "concat":
            fusion = ConcatFusion()
            fusion_output_dim = 128 * len(projectors)
        elif args.fusion_type == "lowrank":
            fusion = LowRankTensorFusion(input_dims=[128, 128], rank=16, output_dim=128)
            fusion_output_dim = 128
        elif args.fusion_type == "dynamic":
            fusion = DynamicWeightedFusion(num_modalities=len(projectors), input_dim=128)
            fusion_output_dim = 128
        elif args.fusion_type == "attention":
            fusion = AttentionFusion(input_dim=128, num_modalities=len(projectors), hidden_dim=64)
            fusion_output_dim = 128
        elif args.fusion_type == "multiplicative":
            fusion = MultiplicativeFusion()
            fusion_output_dim = 128
        elif args.fusion_type == "residual":
            fusion = ResidualFusion()
            fusion_output_dim = 128
        else:  # late fusion
            fusion = LateFusion(input_dim=128, num_classes=2)
            fusion_output_dim = 2  # Late Fusion directly outputs class logits

        # Define head only if not using LateFusion
        if args.fusion_type != "late":
            head = nn.Sequential(
                nn.Linear(fusion_output_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(64, 2)  # Binary classification
            )
        else:
            head = nn.Identity()

        inter_model = InterModalModel(
            encoders=encoders,
            projectors=projectors,
            fusion=fusion,
            hidden_dim=128,        # Hidden dimension in MLP head
            dropout_prob=0.4,      # Dropout probability
            use_batch_norm=True    # Use BatchNorm in MLP head
        )

        # data loader
        train_loaders_list = [train_loader, train_loader]
        valid_loaders_list = [valid_loader, valid_loader]
        test_loaders_list = [test_loader, test_loader]

        save_path = os.path.join(args.save_dir, "inter_model.pt")

        # train, test
        if args.train:
            best_state = train_inter_modality(
                model=inter_model,
                train_loaders=train_loaders_list,
                valid_loaders=valid_loaders_list,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                loss_type=args.loss_type
            )
            if best_state:
                torch.save(best_state, save_path)
                print(f"[Info] Saved inter model best state to {save_path}")

        if args.test:
            if os.path.exists(save_path):
                inter_model.load_state_dict(torch.load(save_path))
            else:
                print(f"[Warning] No checkpoint found at {save_path}. Testing untrained model.")
            test_inter_modality(inter_model, test_loaders_list, device=device)


    # 4. Inter-Intra Modality
    elif args.model_type == "inter_intra":
        print("Main function started")

        # Load Unimodal Checkpoints
        unimodal_bert_path = os.path.join(args.save_dir, "unimodal_chembert.pt")
        unimodal_cnn_path = os.path.join(args.save_dir, "unimodal_cnn_gru.pt")

        if not os.path.exists(unimodal_bert_path) or not os.path.exists(unimodal_cnn_path):
            raise FileNotFoundError("[Error] Unimodal checkpoints not found. Train unimodal models first.")

        unimodal_bert = ChemBERTBinaryClassifier(return_features=False)
        unimodal_bert.load_state_dict(torch.load(unimodal_bert_path))
        print(f"[Info] Loaded Unimodal ChemBERT checkpoint from {unimodal_bert_path}")

        unimodal_cnn = CNNGRUBinaryClassifier(
            vocab_size=800,
            emb_dim=128,
            num_filters_list=[64, 128],
            kernel_sizes=[3, 5],
            hidden_dim=128,
            num_layers=2,
            num_classes=2
        )
        unimodal_cnn.load_state_dict(torch.load(unimodal_cnn_path))
        print(f"[Info] Loaded Unimodal CNN-GRU checkpoint from {unimodal_cnn_path}")

        # Define Inter-Model Encoders
        inter_bert = ChemBERTBinaryClassifier(return_features=True)
        inter_cnn = CNNGRUBinaryClassifier(
            vocab_size=800,
            emb_dim=128,
            num_filters_list=[64, 128],
            kernel_sizes=[3, 5],
            hidden_dim=128,
            num_layers=2,
            num_classes=2,
            return_features=True
        )

        if args.fusion_type == "concat":
            fusion = ConcatFusion()
            fusion_output_dim = 128 * len(projectors)
        elif args.fusion_type == "lowrank":
            fusion = LowRankTensorFusion(input_dims=[128, 128], rank=16, output_dim=128)
            fusion_output_dim = 128
        elif args.fusion_type == "dynamic":
            fusion = DynamicWeightedFusion(num_modalities=len(projectors), input_dim=128)
            fusion_output_dim = 128
        elif args.fusion_type == "attention":
            fusion = AttentionFusion(input_dim=128, num_modalities=2, hidden_dim=64)
            fusion_output_dim = 128
        elif args.fusion_type == "multiplicative":
            fusion = MultiplicativeFusion()
            fusion_output_dim = 128
        elif args.fusion_type == "residual":
            fusion = ResidualFusion()
            fusion_output_dim = 128
        else:  # late fusion
            fusion = LateFusion(input_dim=128, num_classes=2)
            fusion_output_dim = 2  


        # Load Inter Model Checkpoint if exists
        inter_save_path = os.path.join(args.save_dir, "inter_model.pt")
        if os.path.exists(inter_save_path):
            inter_model = InterModalModel(
                encoders=[inter_bert, inter_cnn],
                projectors=[nn.Linear(768, 128), nn.Linear(256, 128)],
                fusion=AttentionFusion(input_dim=128, num_modalities=2, hidden_dim=64),
                hidden_dim=128,
                dropout_prob=0.4,
                use_batch_norm=True
            )
            inter_model.load_state_dict(torch.load(inter_save_path))
            print(f"[Info] Loaded Inter Model checkpoint from {inter_save_path}")
        else:
            raise FileNotFoundError("[Error] Inter model checkpoint not found. Train inter model first.")

        # Train and Validation Loaders
        train_loaders_list = [train_loader, train_loader]
        valid_loaders_list = [valid_loader, valid_loader]
        test_loaders_list = [test_loader, test_loader]

        # Save paths for Inter-Intra
        inter_intra_save_path = os.path.join(args.save_dir, "inter_intra_inter.pt")
        unimodal_save_paths = [
            os.path.join(args.save_dir, "inter_intra_uni_bert.pt"),
            os.path.join(args.save_dir, "inter_intra_uni_cnn.pt")
        ]

        if args.train:
            best_states = train_inter_intra_modality(
                inter_model=inter_model,
                unimodal_models=[unimodal_bert, unimodal_cnn],
                train_loaders=train_loaders_list,
                valid_loaders=valid_loaders_list,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                alpha=0.25,
                gamma=2.0
            )
            if best_states:
                torch.save(best_states["inter_model"], inter_intra_save_path)
                torch.save(best_states["unimodals"][0], unimodal_save_paths[0])
                torch.save(best_states["unimodals"][1], unimodal_save_paths[1])
                print("[Info] Saved Inter-Intra model checkpoints")

        if args.test:
            # Load Inter-Intra Checkpoints
            if os.path.exists(inter_intra_save_path):
                inter_model.load_state_dict(torch.load(inter_intra_save_path))
                print(f"[Info] Loaded Inter-Intra Inter Model checkpoint from {inter_intra_save_path}")
            else:
                print(f"[Warning] No Inter-Intra Inter Model checkpoint found. Testing with trained Inter Model.")

            for i, path in enumerate(unimodal_save_paths):
                if os.path.exists(path):
                    if i == 0:
                        unimodal_bert.load_state_dict(torch.load(path))
                    elif i == 1:
                        unimodal_cnn.load_state_dict(torch.load(path))
                    print(f"[Info] Loaded Unimodal checkpoint from {path}")
                else:
                    print(f"[Warning] No checkpoint found for unimodal model {i + 1}. Using previously loaded model.")

            # Test Inter-Intra Modality
            test_inter_intra_modality(
                inter_model=inter_model,
                unimodal_models=[unimodal_bert, unimodal_cnn],
                test_loaders=test_loaders_list,
                device=device
            )


if __name__ == "__main__":
    main()