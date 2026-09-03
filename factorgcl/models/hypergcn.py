"""Hypergraph convolution used by both beta modules of FactorGCL.

The layer implements Eq. (1) of the paper::

    e^{(l+1)} = sigma( D_n^{-1/2} H W D_e^{-1} H^T D_n^{-1/2} e^{(l)} w )

which the paper describes as three steps (Figure 4):

1. *message extraction*  -- ``e w``, a linear map of every stock node feature;
2. *message aggregation* -- ``D_e^{-1} H^T (...)``, the degree-normalised mean of
   the nodes connected by a hyperedge, i.e. the shared factor information;
3. *message sharing*     -- ``H (...)``, sending the factor information back to
   the stock nodes it is connected to.

The incidence matrix ``H`` may be binary (prior beta module, industry
membership) or soft with values in ``[0, 1]`` (hidden beta module, Sigmoid of a
similarity), which is why every degree is computed from ``H`` itself instead of
from a hard membership count.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class HypergraphConv(nn.Module):
    """A single HyperGCN layer.

    Args:
        in_dim: dimension ``d_l`` of the incoming node features.
        out_dim: dimension ``d_{l+1}`` of the produced node features.
        bias: add a bias to the message-extraction linear map.  The formula in
            the paper has no bias term, so this defaults to ``False``.
        activation: non-linearity ``sigma``.  The paper uses LeakyReLU.
        eps: floor applied to the degrees before inversion, so that isolated
            stocks (a stock belonging to no factor) do not produce ``inf``.

    Shapes:
        ``x``: ``(N, in_dim)`` or ``(B, N, in_dim)``
        ``incidence``: ``(N, E)`` or ``(B, N, E)``
        ``edge_weight``: ``(E,)`` or ``(B, E)``; ``None`` means ``W = I``
        ``node_mask``: ``(N,)`` or ``(B, N)``; ``0`` marks padded stocks
        returns: same leading shape as ``x`` with ``out_dim`` channels
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = False,
        activation: Optional[nn.Module] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.eps = eps
        # `w` of Eq. (1): the learnable message-extraction matrix.
        self.weight = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation if activation is not None else nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight.weight)
        if self.weight.bias is not None:
            nn.init.zeros_(self.weight.bias)

    def forward(
        self,
        x: torch.Tensor,
        incidence: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if node_mask is not None:
            # Padded stocks must neither contribute to nor receive messages.
            incidence = incidence * node_mask.unsqueeze(-1).to(incidence.dtype)

        if edge_weight is None:
            weighted_incidence = incidence
        else:
            weighted_incidence = incidence * edge_weight.unsqueeze(-2).to(incidence.dtype)

        # D_n = diag(sum_e w(e) H(:, e)),  D_e = diag(sum_v H(v, :))
        node_degree = weighted_incidence.sum(dim=-1)
        edge_degree = incidence.sum(dim=-2)
        node_inv_sqrt = node_degree.clamp(min=self.eps).pow(-0.5)
        edge_inv = edge_degree.clamp(min=self.eps).pow(-1.0)
        if node_mask is not None:
            node_inv_sqrt = node_inv_sqrt * node_mask.to(node_inv_sqrt.dtype)

        # 1. message extraction: e w
        h = self.weight(x)
        # D_n^{-1/2} e w
        h = h * node_inv_sqrt.unsqueeze(-1)
        # 2. message aggregation: W D_e^{-1} H^T (...)
        edge_message = torch.matmul(incidence.transpose(-2, -1), h)
        edge_message = edge_message * edge_inv.unsqueeze(-1)
        if edge_weight is not None:
            edge_message = edge_message * edge_weight.unsqueeze(-1).to(edge_message.dtype)
        # 3. message sharing: D_n^{-1/2} H (...)
        out = torch.matmul(incidence, edge_message)
        out = out * node_inv_sqrt.unsqueeze(-1)

        out = self.activation(out)
        if node_mask is not None:
            out = out * node_mask.unsqueeze(-1).to(out.dtype)
        return out

    def extra_repr(self) -> str:  # pragma: no cover - debugging helper
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}"
