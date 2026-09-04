"""Objective function: Eqs. (11)-(13)."""

import math

import torch

from factorgcl.losses import FactorGCLLoss, info_nce_loss, multi_period_mse


def reference_info_nce(anchor, positive, temperature):
    """Literal transcription of Eq. (11)."""
    num_nodes = anchor.shape[0]
    total = 0.0
    for i in range(num_nodes):
        numerator = math.exp(
            float(torch.cosine_similarity(anchor[i], positive[i], dim=0)) / temperature
        )
        denominator = sum(
            math.exp(float(torch.cosine_similarity(anchor[i], positive[j], dim=0)) / temperature)
            for j in range(num_nodes)
        )
        total += math.log(numerator / denominator)
    return -total / num_nodes


def test_info_nce_matches_equation_11():
    torch.manual_seed(0)
    anchor, positive = torch.randn(11, 8), torch.randn(11, 8)
    got = float(info_nce_loss(anchor, positive, temperature=0.1))
    assert got == torch.tensor(reference_info_nce(anchor, positive, 0.1)).item() or abs(
        got - reference_info_nce(anchor, positive, 0.1)
    ) < 1e-4


def test_info_nce_is_minimal_for_aligned_pairs():
    torch.manual_seed(1)
    embeddings = torch.randn(16, 8)
    aligned = float(info_nce_loss(embeddings, embeddings, temperature=0.1))
    shuffled = float(info_nce_loss(embeddings, embeddings[torch.randperm(16)], temperature=0.1))
    assert aligned < shuffled


def test_info_nce_ignores_padded_stocks():
    torch.manual_seed(2)
    anchor, positive = torch.randn(1, 10, 6), torch.randn(1, 10, 6)
    mask = torch.zeros(1, 10)
    mask[0, :7] = 1.0

    masked = float(info_nce_loss(anchor, positive, 0.1, node_mask=mask))
    trimmed = float(info_nce_loss(anchor[:, :7], positive[:, :7], 0.1))
    assert abs(masked - trimmed) < 1e-5
    assert math.isfinite(masked)


def test_multi_period_mse_ignores_padding():
    prediction = torch.tensor([[[1.0, 2.0], [0.0, 0.0], [5.0, 5.0]]])
    target = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    # only the first two stocks count: (1 + 4 + 0 + 0) / (2 stocks * 2 periods)
    assert abs(float(multi_period_mse(prediction, target, mask)) - 1.25) < 1e-6


def test_total_loss_is_mse_plus_gamma_times_contrastive():
    torch.manual_seed(3)
    prediction, target = torch.randn(12, 4), torch.randn(12, 4)
    anchor, positive = torch.randn(12, 8), torch.randn(12, 8)
    criterion = FactorGCLLoss(gamma=0.1, temperature=0.1)
    out = criterion(prediction, target, anchor, positive)
    assert abs(float(out.total) - (float(out.mse) + 0.1 * float(out.contrastive))) < 1e-6


def test_contrastive_can_be_disabled():
    torch.manual_seed(4)
    prediction, target = torch.randn(6, 4), torch.randn(6, 4)
    anchor, positive = torch.randn(6, 8), torch.randn(6, 8)
    out = FactorGCLLoss(use_contrastive=False)(prediction, target, anchor, positive)
    assert float(out.contrastive) == 0.0
    assert float(out.total) == float(out.mse)
