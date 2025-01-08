"""
datasets/preprocess.py

1) TDC에서 'CYP2C19_Veith' 데이터셋 로드
2) Train/Valid/Test split
3) RDKit을 활용한 전처리(대표적인 CYP 예측 전처리 예시):
   - 분자 파싱 (MolFromSmiles)
   - LargestFragmentChooser(조합제/염 등 제거)
   - Stereochemistry 제거 (필요시)
   - Neutralization (필요시)
   - Canonical SMILES 변환
4) 전처리 후 유효한 SMILES만 추출
5) CSV 형태로 ./data/ 폴더에 저장
"""

import os
import pandas as pd
from tdc.single_pred import ADME

# RDKit 관련
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import MolToSmiles
from rdkit.Chem.MolStandardize import rdMolStandardize


def standardize_smiles(smiles: str, remove_stereo: bool = True) -> str:
    """
    주어진 SMILES 문자열에 대해 다음 전처리를 수행:
    1) MolFromSmiles로 파싱 (유효하지 않으면 None 반환)
    2) LargestFragmentChooser로 가장 큰 프래그먼트만 남김
    3) 필요시 Stereochemistry(입체화학) 제거
    4) 중성화(NeutralizeCharges) -> ex) 양/음이온 처리
    5) Canonical SMILES로 변환
    6) 만약 어떠한 이유로 None이 되거나 에러 발생 시 반환 None

    Parameters
    ----------
    smiles : str
        원본 SMILES
    remove_stereo : bool
        True면 입체화학 정보 제거(rdmolops.RemoveStereochemistry)

    Returns
    -------
    str or None
        전처리된 canonical SMILES (실패 시 None)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 가장 큰 프래그먼트만 선택 (염 등 제거)
        lfc = rdMolStandardize.LargestFragmentChooser()
        mol = lfc.choose(mol)
        if mol is None:
            return None

        # Optional: 입체화학 정보 제거
        if remove_stereo:
            rdmolops.RemoveStereochemistry(mol)

        # 중성화 (양/음이온 정규화)
        uncharger = rdMolStandardize.Uncharger()  # default
        mol = uncharger.uncharge(mol)

        # 만약 추가로 pH 중성화나 금속 제거 등이 필요하다면
        # rdMolStandardize.MetalDisconnector()(mol) 등등 호출 가능

        # Sanitize (예: valence 에러 체크 등)
        Chem.SanitizeMol(mol)

        # Canonical SMILES
        cano = MolToSmiles(mol, isomericSmiles=False)
        return cano

    except Exception as e:
        return None


def preprocess_cyp2c19_data(
    save_dir: str = "./data",
    remove_stereo: bool = True,
    save_csv: bool = True
):
    """
    TDC에서 'CYP2C19_Veith'를 불러온 뒤,
    RDKit 전처리를 통해 SMILES를 정규화하고,
    ./data 폴더에 csv로 저장(기본).

    Parameters
    ----------
    save_dir : str
        전처리된 CSV를 저장할 디렉토리
    remove_stereo : bool
        True면 입체화학 정보 제거
    save_csv : bool
        True일 경우 csv로 저장, False일 경우 (train_df, val_df, test_df) 반환
    """
    # 1) TDC에서 데이터 불러오기
    data = ADME(name='CYP2C19_Veith')  # 이 작업 시, 자동으로 TDC 데이터 다운로드
    split = data.get_split()          # dict로 train/valid/test 분리
    train_data = split['train']       # pd.DataFrame(["Drug", "Y"])
    valid_data = split['valid']
    test_data  = split['test']

    # 2) 열 이름 정리
    #    TDC: Drug -> SMILES, Y -> Label 등으로 변경
    train_data = train_data.rename(columns={"Drug": "SMILES", "Y": "Label"})
    valid_data = valid_data.rename(columns={"Drug": "SMILES", "Y": "Label"})
    test_data  = test_data.rename(columns={"Drug": "SMILES", "Y": "Label"})

    # 3) RDKit 전처리 (canonicalize 등)
    def rdkit_process_and_filter(df):
        # df["Label"]가 regression or classification인지에 따라 추가 처리 필요
        # (CYP2C19_Veith는 연속값 -> IC50(or some measure)로 알려져 있음)
        # 여기서는 그냥 Label로 사용.
        # 만약 이진 분류가 필요하다면, 특정 cutoff로 binary labeling 추가 가능.

        processed_smiles = []
        processed_labels = []
        for i in range(len(df)):
            smi = df.iloc[i]["SMILES"]
            val = df.iloc[i]["Label"]
            std_smi = standardize_smiles(smi, remove_stereo=remove_stereo)
            if std_smi is not None:
                processed_smiles.append(std_smi)
                processed_labels.append(val)

        new_df = pd.DataFrame({"SMILES": processed_smiles, "Label": processed_labels})
        return new_df

    train_data_processed = rdkit_process_and_filter(train_data)
    valid_data_processed = rdkit_process_and_filter(valid_data)
    test_data_processed  = rdkit_process_and_filter(test_data)

    # 4) 중복 제거, 결측 제거
    train_data_processed = train_data_processed.drop_duplicates(subset=["SMILES"]).dropna()
    valid_data_processed = valid_data_processed.drop_duplicates(subset=["SMILES"]).dropna()
    test_data_processed  = test_data_processed.drop_duplicates(subset=["SMILES"]).dropna()

    # 5) CSV 저장 or 반환
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    train_path = os.path.join(save_dir, "cyp2c19_train.csv")
    valid_path = os.path.join(save_dir, "cyp2c19_valid.csv")
    test_path  = os.path.join(save_dir, "cyp2c19_test.csv")

    if save_csv:
        train_data_processed.to_csv(train_path, index=False)
        valid_data_processed.to_csv(valid_path, index=False)
        test_data_processed.to_csv(test_path,   index=False)
        print(f"[Info] Saved preprocessed CSVs to:\n  {train_path}\n  {valid_path}\n  {test_path}")
    else:
        return train_data_processed, valid_data_processed, test_data_processed
