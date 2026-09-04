"""FactorGCL: a hypergraph-based factor model with temporal residual
contrastive learning (Duan, Wang & Li, AAAI-25).

The cascading residual hypergraph architecture of Figure 3 decomposes the
predicted stock returns into three components::

    e_s = phi_feat(x)                            # stock temporal features
    e_p = phi_prior(e_s, beta)                   # prior beta      (Eq. 5)
    e_r = e_s - e_p                              # residual after prior factors
    beta_h = Sigmoid(e_r c^T)                    # hidden factor exposures
    e_h = phi_hidden(e_r, beta_h)                # hidden beta     (Eq. 6)
    e_a = LeakyReLU(w_a (e_s - e_p - e_h) + b_a) # individual alpha (Eq. 7)
    y^(l) = w_o1^(l) e_p + w_o2^(l) e_h + w_o3^(l) e_a + b_o^(l)  (Eq. 8)

and the contrastive branch re-runs the same (shared) beta modules on future
market data to obtain ``e'_a`` (Eqs. 9-10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..config import ModelConfig
from .hypergcn import HypergraphConv


@dataclass
class FactorGCLOutput:
    """Everything the losses and the analysis scripts need from one forward pass."""

    #: predicted returns for every forward period, ``(N, L)`` or ``(B, N, L)``
    prediction: torch.Tensor
    #: hidden factor exposures ``beta_h``, ``(N, M)`` or ``(B, N, M)``
    hidden_exposure: torch.Tensor
    #: stock temporal features ``e_s``
    stock_embedding: torch.Tensor
    #: prior beta embeddings ``e_p``
    prior_embedding: torch.Tensor
    #: hidden beta embeddings ``e_h``
    hidden_embedding: torch.Tensor
    #: individual alpha embeddings ``e_a``
    alpha_embedding: torch.Tensor
    #: future individual alpha embeddings ``e'_a`` (``None`` without future data)
    future_alpha_embedding: Optional[torch.Tensor] = None


class FeatureExtractor(nn.Module):
    """``phi_feat``: a GRU with batch normalisation over the raw market sequence.

    "To capture long-term dependencies in sequential data, we utilize a gated
    recurrent unit with a batch normalization as the feature extractor, using
    the hidden state at the last time step as the stock feature embeddings."
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(N, T, D)`` or ``(B, N, T, D)`` -> ``(N, H)`` / ``(B, N, H)``."""
        leading = x.shape[:-2]
        seq_len, feat_dim = x.shape[-2], x.shape[-1]
        flat = x.reshape(-1, seq_len, feat_dim)
        # BatchNorm1d expects (batch, channels, length): normalise every raw
        # market field across the whole batch of stock-days.
        flat = self.input_norm(flat.transpose(1, 2)).transpose(1, 2)
        _, hidden = self.rnn(flat)
        # hidden state of the last layer at the last time step
        out = hidden[-1]
        return out.reshape(*leading, self.hidden_dim)


class PriorBetaModule(nn.Module):
    """``phi_prior``: HyperGCN over the prior hypergraph ``G_p`` (Eq. 5).

    Stocks are nodes, the ``K`` human-designed prior factors are hyperedges and
    the factor exposure matrix ``beta`` is the incidence matrix.  ``W = I``.
    """

    def __init__(self, hidden_dim: int, leaky_slope: float = 0.01) -> None:
        super().__init__()
        self.conv = HypergraphConv(
            hidden_dim, hidden_dim, activation=nn.LeakyReLU(negative_slope=leaky_slope)
        )

    def forward(
        self,
        stock_embedding: torch.Tensor,
        prior_exposure: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.conv(stock_embedding, prior_exposure, node_mask=node_mask)


class HiddenBetaModule(nn.Module):
    """``phi_hidden``: hyperedge generation + HyperGCN over ``G_h`` (Eq. 6).

    ``M`` learnable *hidden factor prototypes* ``c^(i) in R^H`` turn the residual
    embeddings into soft hyperedges ``beta_h(i, j) = Sigmoid(e_r^(i) . c^(j)T)``
    with values in ``[0, 1]``.
    """

    def __init__(self, hidden_dim: int, num_hidden_factors: int, leaky_slope: float = 0.01) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.empty(num_hidden_factors, hidden_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.conv = HypergraphConv(
            hidden_dim, hidden_dim, activation=nn.LeakyReLU(negative_slope=leaky_slope)
        )

    def compute_exposure(self, residual_embedding: torch.Tensor) -> torch.Tensor:
        """Mine the hidden factor exposure matrix ``beta_h`` from the residuals."""
        return torch.sigmoid(torch.matmul(residual_embedding, self.prototypes.t()))

    def propagate(
        self,
        residual_embedding: torch.Tensor,
        hidden_exposure: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the HyperGCN with a *given* exposure matrix.

        The future branch of the contrastive loss (Eq. 10) reuses the ``beta_h``
        mined from historical data, so exposure generation and propagation are
        kept separate.
        """
        return self.conv(residual_embedding, hidden_exposure, node_mask=node_mask)

    def forward(
        self,
        residual_embedding: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_exposure = self.compute_exposure(residual_embedding)
        hidden_embedding = self.propagate(residual_embedding, hidden_exposure, node_mask=node_mask)
        return hidden_embedding, hidden_exposure


class IndividualAlphaModule(nn.Module):
    """``phi_alpha``: ``e_a = LeakyReLU(w_a (e_s - e_p - e_h) + b_a)`` (Eq. 7)."""

    def __init__(self, hidden_dim: int, leaky_slope: float = 0.01) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=leaky_slope)

    def forward(self, residual_embedding: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(residual_embedding))


class PredictionHead(nn.Module):
    """Multi-label prediction head of Eq. (8).

    One independent linear map per component (prior beta, hidden beta, alpha)
    and per forward period ``l``, plus a shared bias ``b_o^(l)``.
    """

    def __init__(self, hidden_dim: int, num_periods: int) -> None:
        super().__init__()
        self.prior_head = nn.Linear(hidden_dim, num_periods, bias=False)
        self.hidden_head = nn.Linear(hidden_dim, num_periods, bias=False)
        self.alpha_head = nn.Linear(hidden_dim, num_periods, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_periods))

    def forward(
        self,
        prior_embedding: torch.Tensor,
        hidden_embedding: torch.Tensor,
        alpha_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.prior_head(prior_embedding)
            + self.hidden_head(hidden_embedding)
            + self.alpha_head(alpha_embedding)
            + self.bias
        )


class ProjectionHead(nn.Module):
    """``p(.)`` of Eq. (11): "a 2-layer MLP with LeakyReLU activation functions"."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, leaky_slope: float = 0.01) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=leaky_slope),
            nn.Linear(hidden_dim, out_dim),
            nn.LeakyReLU(negative_slope=leaky_slope),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FactorGCL(nn.Module):
    """The full cascading residual hypergraph model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        cfg = config

        self.feature_extractor = FeatureExtractor(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_rnn_layers,
            dropout=cfg.dropout,
        )
        # phi'_feat: the paper only states that phi'_prior / phi'_hidden share
        # parameters with their historical counterparts, so the future feature
        # extractor is separate unless explicitly tied.
        if cfg.share_feature_extractor:
            self.future_feature_extractor = self.feature_extractor
        else:
            self.future_feature_extractor = FeatureExtractor(
                input_dim=cfg.input_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_rnn_layers,
                dropout=cfg.dropout,
            )

        self.prior_module = (
            PriorBetaModule(cfg.hidden_dim, cfg.leaky_slope) if cfg.use_prior else None
        )
        self.hidden_module = (
            HiddenBetaModule(cfg.hidden_dim, cfg.num_hidden_factors, cfg.leaky_slope)
            if cfg.use_hidden
            else None
        )
        self.alpha_module = (
            IndividualAlphaModule(cfg.hidden_dim, cfg.leaky_slope) if cfg.use_alpha else None
        )
        self.prediction_head = PredictionHead(cfg.hidden_dim, cfg.num_periods)
        self.projection_head = ProjectionHead(
            cfg.hidden_dim,
            cfg.projection_hidden_dim or cfg.hidden_dim,
            cfg.projection_dim,
            cfg.leaky_slope,
        )

    # ------------------------------------------------------------------
    def _encode_betas(
        self,
        stock_embedding: torch.Tensor,
        prior_exposure: torch.Tensor,
        node_mask: Optional[torch.Tensor],
        hidden_exposure: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the two cascading beta modules.

        When ``hidden_exposure`` is given it is used as-is instead of being mined
        from the residuals; this is what the future branch of Eq. (10) needs.
        Returns ``(e_p, e_h, beta_h)``.
        """
        if self.prior_module is not None:
            prior_embedding = self.prior_module(stock_embedding, prior_exposure, node_mask=node_mask)
        else:
            prior_embedding = torch.zeros_like(stock_embedding)

        residual = stock_embedding - prior_embedding

        if self.hidden_module is not None:
            if hidden_exposure is None:
                hidden_exposure = self.hidden_module.compute_exposure(residual)
            hidden_embedding = self.hidden_module.propagate(
                residual, hidden_exposure, node_mask=node_mask
            )
        else:
            hidden_embedding = torch.zeros_like(stock_embedding)
            hidden_exposure = stock_embedding.new_zeros(
                (*stock_embedding.shape[:-1], self.config.num_hidden_factors)
            )
        return prior_embedding, hidden_embedding, hidden_exposure

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        prior_exposure: torch.Tensor,
        future_x: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> FactorGCLOutput:
        """Predict stock returns and, optionally, the future alpha embeddings.

        Args:
            x: historical market sequences, ``(N, T, D)`` or ``(B, N, T, D)``.
            prior_exposure: prior factor exposures ``beta``, ``(N, K)`` / ``(B, N, K)``.
            future_x: future market sequences ``x'``, ``(N, T', D)`` / ``(B, N, T', D)``.
                Only needed while training with the contrastive loss.
            node_mask: ``1`` for real stocks, ``0`` for padding.
        """
        stock_embedding = self.feature_extractor(x)
        if node_mask is not None:
            stock_embedding = stock_embedding * node_mask.unsqueeze(-1).to(stock_embedding.dtype)

        prior_embedding, hidden_embedding, hidden_exposure = self._encode_betas(
            stock_embedding, prior_exposure, node_mask
        )

        alpha_residual = stock_embedding - prior_embedding - hidden_embedding
        if self.alpha_module is not None:
            alpha_embedding = self.alpha_module(alpha_residual)
        else:
            alpha_embedding = torch.zeros_like(stock_embedding)

        prediction = self.prediction_head(prior_embedding, hidden_embedding, alpha_embedding)

        future_alpha_embedding = None
        if future_x is not None:
            future_alpha_embedding = self.encode_future_alpha(
                future_x, prior_exposure, hidden_exposure, node_mask=node_mask
            )

        return FactorGCLOutput(
            prediction=prediction,
            hidden_exposure=hidden_exposure,
            stock_embedding=stock_embedding,
            prior_embedding=prior_embedding,
            hidden_embedding=hidden_embedding,
            alpha_embedding=alpha_embedding,
            future_alpha_embedding=future_alpha_embedding,
        )

    # ------------------------------------------------------------------
    def encode_future_alpha(
        self,
        future_x: torch.Tensor,
        prior_exposure: torch.Tensor,
        hidden_exposure: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Future alpha embeddings ``e'_a`` of Eqs. (9)-(10).

        ``phi'_prior`` and ``phi'_hidden`` share their parameters with the
        historical modules, and the exposures ``beta`` / ``beta_h`` are the ones
        obtained from historical data.
        """
        future_stock_embedding = self.future_feature_extractor(future_x)
        if node_mask is not None:
            future_stock_embedding = future_stock_embedding * node_mask.unsqueeze(-1).to(
                future_stock_embedding.dtype
            )

        future_prior, future_hidden, _ = self._encode_betas(
            future_stock_embedding, prior_exposure, node_mask, hidden_exposure=hidden_exposure
        )
        residual = future_stock_embedding - future_prior - future_hidden
        # Eq. (10) states the future alpha as the plain residual; the symmetric
        # variant additionally applies the shared individual alpha module.
        if self.config.future_alpha_uses_module and self.alpha_module is not None:
            return self.alpha_module(residual)
        return residual

    # ------------------------------------------------------------------
    def project(self, embedding: torch.Tensor) -> torch.Tensor:
        """``p(.)`` used by the InfoNCE objective."""
        return self.projection_head(embedding)
