"""
unimodal/chembert_binary_classifier.py

TDC의 CYP 이진 분류를 위해 Hugging Face ChemBERT 백본 활용.

"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ChemBERTBinaryClassifier(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "seyonec/PubChem10M_SMILES_BPE_450k",
        num_classes: int = 2,
        dropout_prob: float = 0.4,
        return_features: bool = False  
    ):
        
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.bert = AutoModel.from_pretrained(model_name_or_path, config=self.config)

        hidden_size = self.config.hidden_size  
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.return_features = return_features

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output  

        if self.return_features:

            return pooled_output  
        
        else:
            x = self.dropout(pooled_output)
            logits = self.classifier(x)
            
            return logits
            