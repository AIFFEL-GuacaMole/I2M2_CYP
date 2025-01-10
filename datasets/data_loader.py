import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils

from rdkit import Chem
from rdkit.Chem import MolToSmiles
from rdkit.Chem.MolStandardize import rdMolStandardize


###########################################################
# 1. RDKit-based tokenizer
###########################################################
def rdkit_tokenizer(smiles_str: str):

    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return []
    return [atom.GetAtomicNum() for atom in mol.GetAtoms()]


###########################################################
# 2. SMILES standardization
###########################################################
def standardize_smiles(smiles: str, apply_standardization: bool = True, remove_stereo: bool = True) -> str:

    if not apply_standardization:
        return smiles

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        lfc = rdMolStandardize.LargestFragmentChooser()
        mol = lfc.choose(mol)
        if mol is None:
            return None

        if remove_stereo:
            Chem.rdmolops.RemoveStereochemistry(mol)

        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        Chem.SanitizeMol(mol)
        cano = MolToSmiles(mol, isomericSmiles=False)
        return cano

    except Exception:
        return None


###########################################################
# 3. Dataset class
###########################################################
class CYP2C19Dataset(Dataset):

    def __init__(self, csv_path, transform=None, is_classification=True):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.is_classification = is_classification

        if "SMILES" not in self.df.columns or "Label" not in self.df.columns:
            raise ValueError("CSV must contain 'SMILES' and 'Label' columns.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        smiles_str = self.df.iloc[idx]["SMILES"]
        label_val = self.df.iloc[idx]["Label"]

        # Transform label into a tensor
        if self.is_classification:
            label_tensor = torch.tensor(label_val, dtype=torch.long)
        else:
            label_tensor = torch.tensor(label_val, dtype=torch.float)

        # Transform SMILES to tokenized data
        if self.transform is not None:
            smiles_data = self.transform(smiles_str)
        else:
            smiles_data = []

        return smiles_data, label_tensor


###########################################################
# 4. Collate function for batching
###########################################################
def collate_fn(batch):

    x_list, y_list = [], []
    for (smiles_data, label) in batch:
        x_tensor = torch.tensor(smiles_data, dtype=torch.long)
        x_list.append(x_tensor)
        y_list.append(label)

    # pad_sequence -> shape = [batch_size, max_seq_len]
    x_padded = rnn_utils.pad_sequence(x_list, batch_first=True, padding_value=0)
    labels_tensor = torch.stack(y_list, dim=0)

    return x_padded, labels_tensor


###########################################################
# 5. Preprocessing and saving
###########################################################
def preprocess_and_save_data(
    save_dir: str = "./data",
    apply_standardization: bool = True,
    remove_stereo: bool = True
):

    from tdc.single_pred import ADME

    data = ADME(name='CYP2C19_Veith')
    split = data.get_split()

    def process_data(df):
        processed_smiles, processed_labels = [], []
        for _, row in df.iterrows():
            std_smi = standardize_smiles(row["Drug"], apply_standardization, remove_stereo)
            if std_smi:
                processed_smiles.append(std_smi)
                processed_labels.append(row["Y"])
        return pd.DataFrame({"SMILES": processed_smiles, "Label": processed_labels})

    train_data = process_data(split['train'])
    valid_data = process_data(split['valid'])
    test_data = process_data(split['test'])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_data.to_csv(os.path.join(save_dir, "cyp2c19_train.csv"), index=False)
    valid_data.to_csv(os.path.join(save_dir, "cyp2c19_valid.csv"), index=False)
    test_data.to_csv(os.path.join(save_dir, "cyp2c19_test.csv"), index=False)

    print(f"Processed data saved to {save_dir}")


###########################################################
# 6. DataLoader generation function
###########################################################
def get_cyp2c19_dataloaders(
    data_dir="./data",
    batch_size=32,
    num_workers=0,
    shuffle_train=True,
    is_classification=True
):

    train_csv = os.path.join(data_dir, "cyp2c19_train.csv")
    valid_csv = os.path.join(data_dir, "cyp2c19_valid.csv")
    test_csv = os.path.join(data_dir, "cyp2c19_test.csv")

    train_dataset = CYP2C19Dataset(
        train_csv,
        transform=rdkit_tokenizer,
        is_classification=is_classification
    )
    valid_dataset = CYP2C19Dataset(
        valid_csv,
        transform=rdkit_tokenizer,
        is_classification=is_classification
    )
    test_dataset = CYP2C19Dataset(
        test_csv,
        transform=rdkit_tokenizer,
        is_classification=is_classification
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return train_loader, valid_loader, test_loader