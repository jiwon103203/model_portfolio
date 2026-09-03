"""Objective function of FactorGCL (Eqs. 11-13).

    L      = L_mse + gamma * L_CL
    L_mse  = 1 / (N * L) * sum_l sum_i (y_hat^(i,l) - y^(i,l))^2
    L_CL   = -1/N * sum_i log[ exp(sim(p(e_a^i), p(e'_a^i)) / tau)
                               / sum_j exp(sim(p(e_a^i), p(e'_a^j)) / tau) ]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossOutput:
    total: torch.Tensor
    mse: torch.Tensor
    contrastive: torch.Tensor


def multi_period_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    node_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean squared error over the ``L`` forward prediction periods (Eq. 12).

    Args:
        prediction / target: ``(N, L)`` or ``(B, N, L)``.
        node_mask: ``(N,)`` / ``(B, N)``; padded stocks are excluded.
    """
    squared_error = (prediction - target) ** 2
    if node_mask is None:
        return squared_error.mean()
    mask = node_mask.unsqueeze(-1).to(squared_error.dtype)
    denominator = mask.sum() * squared_error.shape[-1]
    return (squared_error * mask).sum() / denominator.clamp(min=1.0)


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.1,
    node_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """InfoNCE loss of Eq. (11) with cosine similarity.

    ``anchor[i]`` is ``p(e_a^(i))`` (historical alpha) and ``positive[i]`` is
    ``p(e'_a^(i))`` (future alpha of the *same* stock).  Every other stock in the
    same cross-section provides a negative.  The loss is computed independently
    per cross-section (per trading day) and then averaged, because negatives are
    only meaningful among contemporaneous stocks.

    Args:
        anchor / positive: ``(N, P)`` or ``(B, N, P)`` projected embeddings.
        temperature: ``tau``.
        node_mask: ``(N,)`` / ``(B, N)``; padded stocks are neither anchors nor
            negatives.
    """
    if anchor.dim() == 2:
        anchor = anchor.unsqueeze(0)
        positive = positive.unsqueeze(0)
        if node_mask is not None:
            node_mask = node_mask.unsqueeze(0)

    anchor = F.normalize(anchor, dim=-1, eps=1e-8)
    positive = F.normalize(positive, dim=-1, eps=1e-8)

    # similarity[b, i, j] = cos( p(e_a^i), p(e'_a^j) )
    similarity = torch.matmul(anchor, positive.transpose(-2, -1)) / temperature

    if node_mask is not None:
        mask = node_mask.to(torch.bool)
        # mask out padded columns (invalid negatives) before the log-sum-exp
        similarity = similarity.masked_fill(~mask.unsqueeze(-2), float("-inf"))
    else:
        mask = torch.ones(similarity.shape[:-1], dtype=torch.bool, device=similarity.device)

    log_prob = similarity - torch.logsumexp(similarity, dim=-1, keepdim=True)
    num_nodes = mask.shape[1]
    index = torch.arange(num_nodes, device=similarity.device)
    positive_log_prob = log_prob[:, index, index]

    # A padded anchor has a masked-out diagonal and therefore a -inf log-prob;
    # `where` (rather than a multiplication by 0) keeps it out of the sum
    # without turning it into a NaN.
    positive_log_prob = torch.where(
        mask, positive_log_prob, torch.zeros_like(positive_log_prob)
    )
    valid = mask.to(positive_log_prob.dtype)
    loss = -positive_log_prob.sum() / valid.sum().clamp(min=1.0)
    return loss


class FactorGCLLoss(nn.Module):
    """``L = L_mse + gamma * L_CL`` (Eq. 13)."""

    def __init__(self, gamma: float = 0.1, temperature: float = 0.1, use_contrastive: bool = True) -> None:
        super().__init__()
        self.gamma = gamma
        self.temperature = temperature
        self.use_contrastive = use_contrastive

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        anchor: Optional[torch.Tensor] = None,
        positive: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        mse = multi_period_mse(prediction, target, node_mask=node_mask)
        if self.use_contrastive and anchor is not None and positive is not None:
            contrastive = info_nce_loss(
                anchor, positive, temperature=self.temperature, node_mask=node_mask
            )
        else:
            contrastive = torch.zeros((), dtype=mse.dtype, device=mse.device)
        return LossOutput(total=mse + self.gamma * contrastive, mse=mse, contrastive=contrastive)
