import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem


def rdkit_tokenizer(smiles_str: str):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None:
        return []
    return [atom.GetAtomicNum() for atom in mol.GetAtoms()]


class CNNGRUBinaryClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_filters_list,
        kernel_sizes,
        hidden_dim,
        num_layers,
        num_classes,
        pad_idx=0,
        dropout_prob=0.4,
        bidirectional=True,
        use_layer_norm=True,
        return_features=False  
    ):
        super().__init__()
        self.return_features = return_features 

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=pad_idx
        )
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.residual_converters = nn.ModuleList()

        in_channels = emb_dim
        for out_channels, ksz in zip(num_filters_list, kernel_sizes):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=ksz,
                padding=ksz // 2
            )
            self.conv_layers.append(nn.utils.weight_norm(conv))
            if use_layer_norm:
                self.norm_layers.append(nn.LayerNorm(out_channels))
            else:
                self.norm_layers.append(nn.BatchNorm1d(out_channels))

            # Residual connection
            if in_channels != out_channels:
                self.residual_converters.append(
                    nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
                )
            else:
                self.residual_converters.append(nn.Identity())

            in_channels = out_channels

        self.gru = nn.GRU(
            input_size=num_filters_list[-1],
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=(dropout_prob if num_layers > 1 else 0.0)
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.to(next(self.parameters()).device)

    # Embedding
        # [batch_size, seq_len, emb_dim]
        emb = self.embedding(x) 
        # [batch_size, emb_dim, seq_len]
        out = emb.transpose(1, 2) 

        # Conv1D + Normalization + Residual Connection
        for i, (conv, norm, res_conv) in enumerate(zip(self.conv_layers, self.norm_layers, self.residual_converters)):
            res = res_conv(out)  
            conv_out = conv(out).transpose(1, 2)  
            norm_out = norm(conv_out).transpose(1, 2) 
            out = F.relu(norm_out) + res 

        # GRU
        out = out.transpose(1, 2)  
        _, h = self.gru(out)

        if self.gru.bidirectional:
            h_fw, h_bw = h[-2], h[-1]
            last_h = torch.cat([h_fw, h_bw], dim=-1) 
        else:
            last_h = h[-1]  

        # Feature Extraction 
        if self.return_features:
            return last_h

        # Dropout + Linear
        dropped = self.dropout(last_h)
        logits = self.classifier(dropped)  
        return logits