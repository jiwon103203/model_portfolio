"""Configuration objects for FactorGCL.

Default values follow the paper and its supplementary material:

* ``M = 32``  hidden factors
* ``H = 32``  feature embedding dimension
* ``L = 2``   RNN layers
* ``tau = 0.1``  contrastive temperature
* ``gamma = 0.1``  weight of the contrastive loss
* Adam, ``lr = 1e-3``, 100 epochs, early stopping patience 20, seed 0
* ``T = 60`` historical days, ``T' = 20`` future days, ``D = 6`` raw features
* Labels over ``dt in {1, 5, 10, 20}`` forward periods
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DataConfig:
    """Dataset construction, mirroring the "Data Preprocessing" appendix section."""

    #: length of the historical sequence (paper: T = 60)
    seq_len: int = 60
    #: length of the future sequence used by the contrastive branch (paper: T' = 20)
    future_seq_len: int = 20
    #: forward prediction periods used as multi-labels (paper: dt = 1, 5, 10, 20)
    forward_periods: Tuple[int, ...] = (1, 5, 10, 20)
    #: raw day-level market fields; order defines the feature dimension D
    price_fields: Tuple[str, ...] = ("high", "open", "low", "close", "vwap")
    volume_fields: Tuple[str, ...] = ("volume",)
    #: field used to build the labels (paper uses the volume weighted average price)
    label_price_field: str = "vwap"
    #: number of prior factors, i.e. hyperedges of the prior hypergraph
    #: (paper: 83 secondary industries of the China A-share market)
    num_prior_factors: int = 83

    #: how the future window is made dimensionless.  ``"current"`` divides by the
    #: closing price of the prediction day t (so past and future share one anchor);
    #: ``"window"`` divides by the last close inside the future window itself.
    future_anchor: str = "current"

    #: a stock is dropped on a given day when more than this fraction of its
    #: historical window is missing ("drop samples with too many missing values")
    max_missing_ratio: float = 0.1
    #: value used to fill the remaining missing entries
    fill_value: float = 0.0
    #: symmetric clipping applied to the dimensionless features and to the labels
    #: ("clip extreme values in the dataset")
    feature_clip: float = 10.0
    label_clip: float = 10.0
    #: minimum number of stocks required for a cross-section to be usable
    min_stocks_per_day: int = 10

    @property
    def num_features(self) -> int:
        """Feature dimension ``D`` of the raw sequences (paper: 6)."""
        return len(self.price_fields) + len(self.volume_fields)

    @property
    def num_periods(self) -> int:
        """Number of forward prediction periods ``L`` (paper: 4)."""
        return len(self.forward_periods)


@dataclass
class ModelConfig:
    """FactorGCL architecture hyper-parameters."""

    #: raw feature dimension D
    input_dim: int = 6
    #: feature embedding dimension H (paper: 32)
    hidden_dim: int = 32
    #: number of hidden factors M (paper: 32)
    num_hidden_factors: int = 32
    #: number of prior factors K (paper: 83)
    num_prior_factors: int = 83
    #: number of GRU layers (paper: 2)
    num_rnn_layers: int = 2
    #: number of forward prediction periods L
    num_periods: int = 4
    #: negative slope of every LeakyReLU in the model
    leaky_slope: float = 0.01
    dropout: float = 0.0
    #: dimension of the contrastive projection head p(.)
    projection_dim: int = 32
    projection_hidden_dim: Optional[int] = None

    # -- ablation switches (Table 2 of the paper) -------------------------
    use_prior: bool = True
    use_hidden: bool = True
    use_alpha: bool = True

    # -- faithfulness switches -------------------------------------------
    #: Eq. (10) writes the future alpha as the plain residual
    #: ``e'_a = e'_s - phi'_prior - phi'_hidden`` without the linear layer of
    #: Eq. (7).  Set to ``True`` to instead reuse the individual alpha module on
    #: the future branch as well (symmetric variant).
    future_alpha_uses_module: bool = False
    #: The paper only states that ``phi'_prior``/``phi'_hidden`` share parameters
    #: with their historical counterparts, so by default the future branch owns a
    #: separate feature extractor ``phi'_feat``.
    share_feature_extractor: bool = False

    def __post_init__(self) -> None:
        if self.projection_hidden_dim is None:
            self.projection_hidden_dim = self.hidden_dim


@dataclass
class TrainConfig:
    """Optimisation settings, from the "Implementation Details" appendix section."""

    lr: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 100
    early_stopping_patience: int = 20
    #: number of trading days per optimisation step; one day is one cross-section
    batch_days: int = 1
    #: temperature of the InfoNCE loss (paper: 0.1)
    temperature: float = 0.1
    #: weight of the contrastive loss in Eq. (13) (paper: 0.1)
    gamma: float = 0.1
    #: disable the temporal residual contrastive learning term (``-wo CL``)
    use_contrastive: bool = True
    seed: int = 0
    device: str = "auto"
    grad_clip: Optional[float] = 3.0
    #: forward period (in days) whose validation IC drives model selection
    monitor_period: int = 10
    num_workers: int = 0
    shuffle: bool = True
    verbose: bool = True


@dataclass
class BacktestConfig:
    """TopK investment simulation settings (paper: TopK = 30, dt = 10, cost 0.3%)."""

    top_k: int = 30
    holding_days: int = 10
    transaction_cost: float = 0.003
    trading_days_per_year: int = 252
    #: equally weighted universe portfolio is used as benchmark
    benchmark: str = "equal_weight"


@dataclass
class RollingConfig:
    """Rolling train/valid/test protocol (paper: 5 years : 1 year : 2 years)."""

    train_years: int = 5
    valid_years: int = 1
    test_years: int = 2
    #: how far each rolling window advances; the paper rolls by the test length
    step_years: int = 2
    test_start: Optional[str] = "2020-01-01"
    test_end: Optional[str] = "2023-06-30"


@dataclass
class ExperimentConfig:
    """Full experiment description."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    rolling: RollingConfig = field(default_factory=RollingConfig)

    def sync(self) -> "ExperimentConfig":
        """Propagate the data-derived dimensions into the model configuration."""
        self.model.input_dim = self.data.num_features
        self.model.num_periods = self.data.num_periods
        self.model.num_prior_factors = self.data.num_prior_factors
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentConfig":
        def _tuples(section: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
            return {k: (tuple(v) if k in keys and v is not None else v) for k, v in section.items()}

        data = DataConfig(
            **_tuples(
                payload.get("data", {}),
                ["forward_periods", "price_fields", "volume_fields"],
            )
        )
        return cls(
            data=data,
            model=ModelConfig(**payload.get("model", {})),
            train=TrainConfig(**payload.get("train", {})),
            backtest=BacktestConfig(**payload.get("backtest", {})),
            rolling=RollingConfig(**payload.get("rolling", {})),
        ).sync()

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
