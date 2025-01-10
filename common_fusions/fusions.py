"""
common_fusions/fusions.py

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def check_input_dimensions(inputs, expected_dim):
    for i, inp in enumerate(inputs):
        if inp.shape[-1] != expected_dim:
            raise ValueError(
                f"Input tensor at index {i} has shape {inp.shape[-1]}, "
                f"but expected {expected_dim}. Ensure projectors output consistent dimensions."
            )


class ConcatFusion(nn.Module):
    def forward(self, inputs):
        check_input_dimensions(inputs, inputs[0].shape[-1])
        return torch.cat(inputs, dim=-1)


class DynamicWeightedFusion(nn.Module):
    def __init__(self, num_modalities, input_dim):
        super(DynamicWeightedFusion, self).__init__()
        self.weights = nn.Parameter(torch.ones(num_modalities, input_dim))

    def forward(self, inputs):
        check_input_dimensions(inputs, inputs[0].shape[-1])
        normalized_weights = F.softmax(self.weights, dim=0)
        weighted_inputs = [w * inp for w, inp in zip(normalized_weights, inputs)]
        return sum(weighted_inputs)


class AttentionFusion(nn.Module):
    def __init__(self, input_dim, num_modalities, hidden_dim=64):
        super(AttentionFusion, self).__init__()
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1)  # Attention weights across modalities
        )

    def forward(self, inputs):
        check_input_dimensions(inputs, inputs[0].shape[-1])
        concatenated = torch.cat(inputs, dim=-1)  # [batch_size, input_dim * num_modalities]
        attention_weights = self.attention_layer(concatenated)  # [batch_size, num_modalities]
        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, num_modalities, 1]
        inputs_stacked = torch.stack(inputs, dim=1)  # [batch_size, num_modalities, input_dim]
        fused = (inputs_stacked * attention_weights).sum(dim=1)  # [batch_size, input_dim]
        return fused


class MultiplicativeFusion(nn.Module):
    def forward(self, inputs):
        check_input_dimensions(inputs, inputs[0].shape[-1])
        fused = inputs[0]
        for inp in inputs[1:]:
            fused = fused * inp
        return fused


class ResidualFusion(nn.Module):
    def forward(self, inputs):
        check_input_dimensions(inputs, inputs[0].shape[-1])
        base = inputs[0]
        residual = sum(inputs[1:])
        return base + residual


class LowRankTensorFusion(nn.Module):
    def __init__(self, input_dims, rank, output_dim):
        super(LowRankTensorFusion, self).__init__()
        self.rank = rank
        self.output_dim = output_dim
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.Tensor(input_dim, rank)) for input_dim in input_dims]
        )
        self.core = nn.Parameter(torch.Tensor(rank, rank, output_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for factor in self.factors:
            nn.init.xavier_uniform_(factor)
        nn.init.xavier_uniform_(self.core)
        nn.init.zeros_(self.bias)

    def forward(self, inputs):
        batch_size = inputs[0].size(0)
        factor_outputs = [torch.matmul(inp, factor) for inp, factor in zip(inputs, self.factors)]
        outer_product = factor_outputs[0].unsqueeze(2)
        for factor_output in factor_outputs[1:]:
            outer_product = outer_product * factor_output.unsqueeze(1)
        outer_product = outer_product.view(batch_size, self.rank, -1)
        fused = torch.einsum('brm,rmo->bo', outer_product, self.core) + self.bias
        return fused


class LateFusion(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LateFusion, self).__init__()
        self.classifiers = nn.ModuleList([
            nn.Linear(input_dim, num_classes) for _ in range(num_classes)
        ])

    def forward(self, inputs):
        check_input_dimensions(inputs, inputs[0].shape[-1])
        logits = [classifier(inp) for inp, classifier in zip(inputs, self.classifiers)]
        return sum(logits) / len(logits)
