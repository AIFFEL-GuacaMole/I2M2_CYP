"""
common_fusions/fusions.py

다양한 모달 결합(fusion) 기법을 구현해두는 모듈.
I2M2 (inter/intra modality) 구조에서 'inter_modality' 학습 시
encoders -> [feat1, feat2, ...] -> fusion -> head 순서로 활용 가능.

예시:
1) ConcatFusion: 입력 features를 그대로 concat
2) LowRankTensorFusion: 간단한 저랭크 텐서 결합 예시 (2모달 기준)
"""

import torch
import torch.nn as nn

class ConcatFusion(nn.Module):
    """
    여러 모달의 feature 텐서를 단순 concat.
    예: [feat1, feat2, ...] -> cat(dim=1).
    """
    def __init__(self):
        super().__init__()

    def forward(self, feats_list):
        """
        feats_list: list of [batch_size, feature_dim_i]
        output: [batch_size, sum(feature_dim_i)]
        """
        return torch.cat(feats_list, dim=1)


class LowRankTensorFusion(nn.Module):
    """
    간단한 2모달용 Low-Rank Tensor Fusion 예시:
      - 모달 A dim = in_dim_a
      - 모달 B dim = in_dim_b
      - rank = R
      - 최종 출력 dim = out_dim

    (실제로는 3모달 이상, 더 복잡한 구조로 확장 가능)
    """
    def __init__(self, in_dim_a, in_dim_b, rank, out_dim):
        super().__init__()
        self.in_dim_a = in_dim_a
        self.in_dim_b = in_dim_b
        self.rank = rank
        self.out_dim = out_dim

        # 저랭크 근사 위해 가중치 2개 (각 모달)
        self.Fa = nn.Linear(in_dim_a, rank, bias=False)
        self.Fb = nn.Linear(in_dim_b, rank, bias=False)

        # 최종 변환
        # (rank*rank) -> out_dim
        self.out = nn.Linear(rank * rank, out_dim)

    def forward(self, feats_list):
        """
        feats_list: [feat_a, feat_b]
          feat_a shape: [batch_size, in_dim_a]
          feat_b shape: [batch_size, in_dim_b]
        return: [batch_size, out_dim]
        """
        feat_a = feats_list[0]
        feat_b = feats_list[1]

        # proj
        a_proj = self.Fa(feat_a)   # [batch_size, rank]
        b_proj = self.Fb(feat_b)   # [batch_size, rank]

        # outer product -> [batch_size, rank, rank]
        outer = a_proj.unsqueeze(2) * b_proj.unsqueeze(1)

        # flatten -> [batch_size, rank*rank]
        outer_flat = outer.view(outer.size(0), -1)

        # linear transform -> out_dim
        fused = self.out(outer_flat)  # [batch_size, out_dim]
        return fused


if __name__ == "__main__":
    """
    간단 동작 테스트
    """
    fusion_concat = ConcatFusion()
    fusion_lrtf = LowRankTensorFusion(in_dim_a=8, in_dim_b=16, rank=4, out_dim=10)

    batch_size = 4
    feat_a = torch.randn(batch_size, 8)
    feat_b = torch.randn(batch_size, 16)

    # 1) concat test
    out_concat = fusion_concat([feat_a, feat_b])
    print("[Test] ConcatFusion output shape:", out_concat.shape)
    # shape => [4, 24]

    # 2) low-rank fusion test
    out_lrtf = fusion_lrtf([feat_a, feat_b])
    print("[Test] LowRankTensorFusion output shape:", out_lrtf.shape)
    # shape => [4, 10]