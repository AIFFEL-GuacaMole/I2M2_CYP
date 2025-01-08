"""
datasets/data_loader.py

전처리된 cyp2c19_{train,valid,test}.csv 파일을 불러와,
SMILES -> (int index list) 변환 후 PyTorch DataLoader로 내보냅니다.

구성:
1) naive_char_tokenizer: 아주 간단한 문자 단위 토크나이저 예시
2) CYP2C19Dataset: CSV 로드 + (SMILES -> 토큰 리스트) transform
3) collate_fn: pad_sequence로 [batch_size, max_seq_len] 텐서화
4) get_cyp2c19_dataloaders: train/valid/test DataLoader 생성
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

# 0) naive_char_tokenizer
def naive_char_tokenizer(smiles_str: str, max_ord: int = 2000):
    """
    아주 간단한 문자 단위 -> 정수 변환 예시.
    - 각 문자에 대해 ord(c)를 구하되, 너무 큰 ord는 잘라냄 (max_ord=2000 등)
    - 실제로는 별도의 char2idx 사전 구축 등으로 더 정교하게 구현해야 함.

    returns: List[int]
    """
    indices = []
    for c in smiles_str:
        val = ord(c)
        if val > max_ord:
            val = max_ord  
        indices.append(val)
    return indices

# 1) Dataset
class CYP2C19Dataset(Dataset):
    """
    CSV 파일 -> (SMILES, Label).
    'SMILES' 열, 'Label' 열이 있다고 가정.
    transform: SMILES -> List[int] (or torch.Tensor)
    """
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

        # label to tensor
        if self.is_classification:
            label_tensor = torch.tensor(label_val, dtype=torch.long)
        else:
            label_tensor = torch.tensor(label_val, dtype=torch.float)

        # transform: SMILES -> List[int]
        if self.transform is not None:
            smiles_data = self.transform(smiles_str)
        else:
            # if not transform, return SMILES as string
            smiles_data = smiles_str

        return smiles_data, label_tensor

# 2) collate_fn
def collate_fn(batch):
    """
    batch: list of (smiles_data, label_tensor)
      - smiles_data: List[int]
      - label_tensor: torch.long (이진분류) or float

    반환:
      - x_padded: [batch_size, max_seq_len] (long tensor)
      - labels_tensor: [batch_size] (long)
    """
    x_list = []
    y_list = []
    for (smiles_data, label) in batch:
        # smiles_data -> List[int] 형태
        x_tensor = torch.tensor(smiles_data, dtype=torch.long)
        x_list.append(x_tensor)
        y_list.append(label) 

    # pad_sequence (batch_size, max_seq_len) 
    x_padded = rnn_utils.pad_sequence(x_list, batch_first=True, padding_value=0)
    # label stack
    labels_tensor = torch.stack(y_list, dim=0)

    return x_padded, labels_tensor

# 3) get_cyp2c19_dataloaders
def get_cyp2c19_dataloaders(
    data_dir="./data",
    batch_size=32,
    num_workers=0,
    shuffle_train=True,
    is_classification=True
):
    """
    전처리 완료된 CSV (cyp2c19_train.csv, cyp2c19_valid.csv, cyp2c19_test.csv)를 불러와
    SMILES를 naive_char_tokenizer로 변환 -> pad_sequence -> (x, y) 텐서 반환

    Returns
    -------
    train_loader, valid_loader, test_loader
    """
    train_csv = os.path.join(data_dir, "cyp2c19_train.csv")
    valid_csv = os.path.join(data_dir, "cyp2c19_valid.csv")
    test_csv  = os.path.join(data_dir, "cyp2c19_test.csv")

    # Dataset
    train_dataset = CYP2C19Dataset(
        train_csv,
        transform=naive_char_tokenizer, 
        is_classification=is_classification
    )
    valid_dataset = CYP2C19Dataset(
        valid_csv,
        transform=naive_char_tokenizer,
        is_classification=is_classification
    )
    test_dataset  = CYP2C19Dataset(
        test_csv,
        transform=naive_char_tokenizer,
        is_classification=is_classification
    )

    # DataLoader
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
