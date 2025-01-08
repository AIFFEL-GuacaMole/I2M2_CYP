"""
unimodal/chembert_binary_classifier.py

TDC의 CYP 이진 분류를 위해 Hugging Face ChemBERT 백본 활용.

Key Features:
1) Load pre-trained ChemBERT (e.g., HF Hub: "seyonec/PubChem10M_SMILES_BPE_450k")
2) Apply Dropout + Linear on [CLS] embedding (pooler_output) to output (batch_size, 2)
3) In forward pass, receive input_ids and attention_mask, and return logits

Notes:
- Use AutoTokenizer.from_pretrained(...) separately for SMILES tokenization.
- You can use different ChemBERT checkpoints by changing model_name_or_path.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class ChemBERTBinaryClassifier(nn.Module):
    def __init__(
        self,
        model_name_or_path: str = "seyonec/PubChem10M_SMILES_BPE_450k",
        num_classes: int = 2,
        dropout_prob: float = 0.1
    ):

        super().__init__()
        # Load chem BERT 
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        # BERT model(backbone)
        self.bert = AutoModel.from_pretrained(model_name_or_path, config=self.config)

        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None
    ) -> torch.Tensor:

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # outputs.pooler_output: [batch_size, hidden_size]
        pooled_output = outputs.pooler_output

        x = self.dropout(pooled_output)
        logits = self.classifier(x) 
        return logits
    

