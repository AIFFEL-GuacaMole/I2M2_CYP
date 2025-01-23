# I2M2 ADMET Property Prediction

This repository provides an end-to-end pipeline for multi-modality models (`unimodal`, `intra`, `inter`, `inter_intra`) to predict ADMET properties.  
It supports classification and regression tasks, uses PyTorch-based training flows, and integrates seamlessly with Weights & Biases (wandb) for experiment tracking and hyperparameter sweeps.

---

## Directory Structure

```
I2M2_admet_property/
├── main.py
├── sweep.py
├── sweep_config.yaml
├── unimodal/
│   ├── chemberta_1d.py
│   ├── gin_2d.py
│   └── unimol_3d.py
├── datasets/
│   ├── datasets.py
│   └── data_loader.py
├── common_fusions/
│   └── fusions.py
├── utils/
│   ├── earlystop.py
│   └── loss_fn.py
└── training_structures/
    ├── unimodal.py
    ├── intra_modality.py
    ├── inter_modality.py
    ├── inter_and_intra_modality.py
    ├── train.py
    └── test.py
```

### Key Components
- `main.py`: Entry point for training/testing. Parses arguments, loads data, configures models, and calls the train/test loops.  
- `sweep.py`, `sweep_config.yaml`: Weights & Biases hyperparameter sweep support.  
- `unimodal/`: Definitions for 1D (ChemBERTa), 2D (GIN), and 3D (UniMol) models.  
- `datasets/`: Data loading logic. `datasets.py` handles basic CSV + SMILES -> fingerprint, `data_loader.py` builds a single dataset with 1D/2D/3D features.  
- `common_fusions/fusions.py`: Fusion modules (e.g., ConcatFusion, CrossAttentionFusion).  
- `utils/`: Early stopping and custom loss functions.  
- `training_structures/`:  
  - `train.py`, `test.py`: General training & testing loops with wandb logging, scheduler usage, etc.  
  - `*_modality.py`: Logic for unimodal, intra-modality, inter-modality, and inter+intra forward passes.

---

## Requirements

- Python 3.8+  
- PyTorch 1.10+  
- rdkit  
- scikit-learn  
- wandb  
- molfeat  
- (Optional) GPU with CUDA for accelerated training  

Install all requirements:

```bash
pip install -r requirements.txt
```

---

## Usage

Below examples assume you have `train_val.csv` and `test.csv` under the `./data` directory.

### 1) Train a unimodal ChemBERTa (1D) classification model

```bash
python main.py \
    --phase train \
    --mode unimodal \
    --task_type classification \
    --model_type 1D \
    --epochs 50 \
    --lr 1e-4 \
    --batch_size 32
```

This will read `./data/train_val.csv` and `./data/test.csv`, split train/val, generate embeddings for 1D, and train a single ChemBERTa model.  
Logs and checkpoints (`best_unimodal_1D_classification.pth`) will be saved to `./ckpt`.

### 2) Test the unimodal model

```bash
python main.py \
    --phase test \
    --mode unimodal \
    --task_type classification \
    --model_type 1D
```

This will load `./ckpt/best_unimodal_1D_classification.pth` and evaluate it on `./data/test.csv`.

### 3) Train an intra ensemble of 1D/2D/3D classification models

```bash
python main.py \
    --phase train \
    --mode intra \
    --task_type classification \
    --epochs 40 \
    --lr 1e-4
```

This generates 1D/2D/3D features, trains three separate models (ChemBERTa, GIN, UniMol), and ensembles their outputs (simple sum).

### 4) Inter-modality model training (fusion of modalities)

```bash
python main.py \
    --phase train \
    --mode inter \
    --task_type classification \
    --epochs 40
```

This builds three base models (1D/2D/3D) and uses their middle features along with a fusion module (e.g., CrossAttentionFusion).

### 5) Inter + intra combined approach

```bash
python main.py \
    --phase train \
    --mode inter_intra \
    --task_type regression
```

In `inter_and_intra_modality.py`, we sum the direct 1D/2D/3D outputs (intra) and also fuse their middle outputs (inter).

### 6) WandB Sweep

To run a hyperparameter sweep, add `--use_sweep` when `--phase train`:

```bash
python main.py --phase train --use_sweep --mode unimodal
```

The actual sweep configuration is in `sweep_config.yaml`. This will execute `run_sweep(args)` from `sweep.py`.

---

## Logging & Checkpoints

This repository uses Weights & Biases (wandb) for logging. During training, the following metrics are logged:
- **Train:** Loss, Accuracy (classification) or MSE (regression)  
- **Validation:** Loss, Accuracy/AUPRC (classification) or MSE/MAE (regression)  

A `ReduceLROnPlateau` scheduler monitors `val_loss`. The best model weights are saved to `./ckpt`.

---

## Acknowledgments

- `rdkit` for SMILES handling  
- `molfeat` for pretrained transformers  
- `wandb` for experiment tracking  

Feel free to modify the code for your own ADMET tasks or general molecular property prediction.