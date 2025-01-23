#datasets.py

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

def smiles_to_1d(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        return smiles, np.array(fp)
    except Exception:
        return None, None

class DataProcessor:

    def __init__(self, train_val_csv, test_csv):
        self.train_val_csv = train_val_csv
        self.test_csv = test_csv

    def split_train_val(self, val_size=0.2, random_state=42):
        data = pd.read_csv(self.train_val_csv)
        train_data, val_data = train_test_split(
            data, test_size=val_size, random_state=random_state, stratify=data['Y']
        )
        return train_data, val_data

    def process_data(self, data):
        smiles_list = data['Drug'].tolist()
        one_d_data  = []
        one_d_smiles= []
        for smi in smiles_list:
            s_raw, fp = smiles_to_1d(smi)
            one_d_smiles.append(s_raw)
            if fp is not None:
                one_d_data.append(fp.tolist())
            else:
                one_d_data.append(None)

        data['SMILES'] = one_d_smiles
        data['Fingerprint'] = one_d_data
        return data

    def load_and_process_existing_data(self, output_dir):
        train_data_path = os.path.join(output_dir, "train_data.csv")
        val_data_path   = os.path.join(output_dir, "val_data.csv")
        test_data_path  = os.path.join(output_dir, "test_data.csv")

        train_data = pd.read_csv(train_data_path)
        val_data   = pd.read_csv(val_data_path)
        test_data  = pd.read_csv(test_data_path)
        train_data = train_data.dropna(axis=1).drop_duplicates()
        val_data   = val_data.dropna(axis=1).drop_duplicates()
        return train_data, val_data, test_data

    def process_and_save(self, output_dir):
        train_data_path = os.path.join(output_dir, "train_data.csv")
        val_data_path   = os.path.join(output_dir,   "val_data.csv")
        test_data_path  = os.path.join(output_dir,   "test_data.csv")

        if os.path.exists(train_data_path) and os.path.exists(val_data_path) and os.path.exists(test_data_path):
            train_data, val_data, test_data = self.load_and_process_existing_data(output_dir)
        else:
            train_data, val_data = self.split_train_val()
            train_data = self.process_data(train_data)
            val_data   = self.process_data(val_data)
            test_data  = pd.read_csv(self.test_csv)
            test_data  = self.process_data(test_data)

            train_data.to_csv(train_data_path, index=False)
            val_data.to_csv(val_data_path,     index=False)
            test_data.to_csv(test_data_path,   index=False)

        return train_data, val_data, test_data