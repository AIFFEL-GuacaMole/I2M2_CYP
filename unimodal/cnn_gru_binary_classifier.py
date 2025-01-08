"""
unimodal/cnn_gru_binary_classifier.py

참고:
- SMILES를 사용자 정의 토크나이저로 int 인덱스 시퀀스로 변환 -> [batch_size, seq_len] -> 본 모델에 입력
- CrossEntropyLoss 사용 -> logits.shape=[batch_size, 2], label.shape=[batch_size]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1D CNN + GRU based SMILES binary classfier model
class CNNGRUBinaryClassifier(nn.Module):


    def __init__(
        self,
        vocab_size: int = 8000,
        emb_dim: int = 128,
        num_filters_list = [64, 64],
        kernel_sizes = [3, 5],
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 2,
        pad_idx: int = 0,
        dropout_prob: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()


        # num_filters_list = kernel_sizes
        assert len(num_filters_list) == len(kernel_sizes)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.num_filters_list = num_filters_list
        self.kernel_sizes = kernel_sizes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.pad_idx = pad_idx
        self.dropout_prob = dropout_prob
        self.bidirectional = bidirectional

        # 1) embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=pad_idx
        )

        # 2) stacked Conv1D 
        self.conv_layers = nn.ModuleList()
        in_channels = emb_dim
        for out_channels, ksz in zip(num_filters_list, kernel_sizes):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=ksz,
                padding=ksz // 2
            )
            self.conv_layers.append(conv)
            in_channels = out_channels  


        # 3) GRU
        self.gru = nn.GRU(
            input_size=num_filters_list[-1],
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout_prob if num_layers > 1 else 0.0),
            bidirectional=bidirectional
        )

        # GRU output dimension
        gru_out_dim = hidden_dim * (2 if bidirectional else 1)


        # 4) Dropout + Linear classifier
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(gru_out_dim, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [batch_size, seq_len] (int index)
        Returns: logits => [batch_size, 2]
        """
        # embedding => batch_size, seq_len, emb_dim
        emb = self.embedding(x)
        
        out = emb.transpose(1, 2)

        # Optional: BatchNorm, Residual Connection, etc.
        for i, conv in enumerate(self.conv_layers):
            out = conv(out)
            out = F.relu(out)
        

        # 최종 conv 결과 shape: [batch_size, num_filters_list[-1], seq_len]

        # GRU
        out = out.transpose(1, 2)  # => [batch_size, seq_len, num_filters_last]
        out, h = self.gru(out)

        # bidirectional GRU: concat hidden state of forward, backward 
        if self.bidirectional:
        
            fw = h[-2]  # final layer forward
            bw = h[-1]  # final layer backward
            last_h = torch.cat([fw, bw], dim=-1)
        else:
            # unidirectional GRU: last hidden state
            last_h = h[-1]

        # Dropout + Linear
        dropped = self.dropout(last_h)
        logits = self.classifier(dropped) 
        return logits