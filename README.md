# FactorGCL

A faithful PyTorch implementation of

> **FactorGCL: A Hypergraph-Based Factor Model with Temporal Residual Contrastive Learning for Stock Returns Prediction**
> Yitong Duan, Weiran Wang, Jian Li (Tsinghua University) — AAAI-25, pp. 173–180

FactorGCL predicts cross-sectional stock returns by decomposing them into
**prior beta**, **hidden beta**, and **individual alpha**, each extracted from the
residual left by the previous component (the *cascading residual hypergraph
architecture*), and guides the mining of hidden factors with **temporal residual
contrastive learning**.

---

## The method

### 1. Cascading residual hypergraph architecture (Fig. 3)

```
x  ──▶ φ_feat ──▶  e_s                                        # GRU + BatchNorm, last hidden state
                    │
                    ├─▶ φ_prior(e_s, β)      ──▶ e_p          # Eq. (5)  HyperGCN over G_p
                    │                                          #          stocks = nodes, prior factors = hyperedges
              e_r = e_s − e_p
                    │
                    ├─▶ β_h = Sigmoid(e_r cᵀ)                  #          M learnable hidden factor prototypes
                    ├─▶ φ_hidden(e_r, β_h)   ──▶ e_h          # Eq. (6)  HyperGCN over the generated G_h
                    │
        e_s − e_p − e_h
                    │
                    └─▶ LeakyReLU(w_α · + b_α) ──▶ e_α        # Eq. (7)

ŷ⁽ˡ⁾ = w_o1⁽ˡ⁾ e_p + w_o2⁽ˡ⁾ e_h + w_o3⁽ˡ⁾ e_α + b_o⁽ˡ⁾        # Eq. (8), one head per forward period
```

The hypergraph convolution is Eq. (1), implemented exactly as written:

```
e⁽ˡ⁺¹⁾ = σ( D_n^(−1/2) H W D_e^(−1) Hᵀ D_n^(−1/2) e⁽ˡ⁾ w ),   σ = LeakyReLU,  W = I
```

decomposed into the paper's three steps — *message extraction* (`e w`),
*message aggregation* (`D_e^{-1} Hᵀ ·`), *message sharing* (`H ·`).
The prior module's incidence matrix is the binary industry membership β;
the hidden module's is the **soft** matrix β_h ∈ [0,1]^{N×M}, so every degree is
computed from the incidence values themselves rather than a membership count.

### 2. Temporal residual contrastive learning (Fig. 5)

The same (parameter-shared) beta modules are re-run on **future** market data `x'`
with the exposures β and β_h mined from history:

```
e'_s = φ'_feat(x')                                            # Eq. (9)
e'_α = e'_s − φ'_prior(e'_s, β) − φ'_hidden(e'_r, β_h)        # Eq. (10)
```

The past/future alpha of the *same* stock is a positive pair; every other stock
in the same cross-section is a negative. InfoNCE with cosine similarity and a
2-layer LeakyReLU MLP projection `p(·)`:

```
L_CL = −1/N Σᵢ log [ exp(sim(p(e_αⁱ), p(e'_αⁱ))/τ) / Σⱼ exp(sim(p(e_αⁱ), p(e'_αʲ))/τ) ]   # Eq. (11)
L    = L_mse + γ · L_CL                                                                      # Eq. (13)
```

`L_mse` is averaged over the `L` forward prediction periods (Eq. 12).

---

## Repository layout

```
factorgcl/
├── config.py              # every hyper-parameter, defaulted to the paper's values
├── models/
│   ├── hypergcn.py        # Eq. (1), supports binary and soft incidence matrices
│   └── factorgcl.py       # Eqs. (5)–(10): the cascading residual architecture
├── losses.py              # Eqs. (11)–(13): multi-period MSE + InfoNCE
├── metrics.py             # IC, ICIR, RankIC, MSE, sign-based F1 (appendix definitions)
├── backtest.py            # TopK strategy; CR, CER, AR, IR, RoMaD
├── data/
│   ├── preprocess.py      # MarketPanel + the appendix's dimensionless processing
│   ├── dataset.py         # daily cross-sections, padded batching with a node mask
│   └── synthetic.py       # a market with a known latent factor structure
├── engine.py              # Adam / early stopping training loop
└── rolling.py             # the 5y : 1y : 2y rolling protocol
scripts/
├── make_synthetic_data.py
├── run_experiment.py      # train → evaluate → backtest
└── run_ablation.py        # Table 2
tests/                     # 63 tests, incl. formula-level checks against the paper
```

---

## Quick start

```bash
pip install -e ".[dev]"

# 1. a synthetic market (the paper's A-share / HK data is proprietary)
python scripts/make_synthetic_data.py --out data/synthetic_panel.npz \
    --num-stocks 300 --num-days 2400 --num-industries 40

# 2. train, evaluate, and backtest
python scripts/run_experiment.py --panel data/synthetic_panel.npz

# 3. the paper's rolling protocol (5y train : 1y valid : 2y test)
python scripts/run_experiment.py --panel data/synthetic_panel.npz --rolling

# 4. the ablation study of Table 2
python scripts/run_ablation.py --panel data/synthetic_panel.npz
```

As a library:

```python
from factorgcl import ExperimentConfig, Trainer, build_model, split_by_date, backtest_predictions
from factorgcl.data import MarketPanel

panel  = MarketPanel.load("data/synthetic_panel.npz")
config = ExperimentConfig().sync()

train, valid, test = split_by_date(panel, config.data,
    [("2014-01-01", "2018-12-31"), ("2019-01-01", "2019-12-31"), ("2020-01-01", "2021-12-31")])

trainer = Trainer(build_model(config.model, seed=0), config.train, config.data)
trainer.fit(train, valid)

metrics, predictions = trainer.evaluate(build_dataloader(test))
result = backtest_predictions(predictions, panel, config.backtest)
```

---

## Using your own market data

Build a `MarketPanel` — a `(T_total, S, D)` cube of day-level fields with `NaN`
for suspended stocks, plus an industry code per stock (static or time-varying):

```python
from factorgcl.data import MarketPanel

panel = MarketPanel.from_long_dataframe(
    frame,                                   # columns: date, stock_id, high, open, low, close, vwap, volume, industry
    fields=("high", "open", "low", "close", "vwap", "volume"),
)
panel.save("data/my_panel.npz")
```

Everything downstream — the dimensionless processing, the labels, the industry
one-hot β, the rolling splits — is derived from that panel.

---

## Hyper-parameters (all from the paper / supplementary material)

| Setting | Value | Source |
|---|---|---|
| Historical sequence length `T` | 60 | Experiment Settings |
| Future sequence length `T'` | 20 | Experiment Settings |
| Raw features `D` | 6 (high, open, low, close, vwap, volume) | Data Preprocessing |
| Forward periods `Δt` | 1, 5, 10, 20 | Experiment Settings |
| Prior factors `K` | 83 secondary industries | Experiment Settings |
| Hidden factors `M` | 32 | Implementation Details |
| Embedding dimension `H` | 32 | Implementation Details |
| GRU layers | 2 | Implementation Details |
| Temperature `τ` | 0.1 | Implementation Details |
| Loss balance `γ` | 0.1 | Implementation Details |
| Optimizer | Adam, lr 1e-3 | Implementation Details |
| Epochs / early stopping | 100 / 20 | Implementation Details |
| Seed | 0 | Implementation Details |
| Split | 5y : 1y : 2y, rolling | Experiment Settings |
| TopK / holding / cost | 30 / 10 days / 0.3% | Investment Simulation |

`ExperimentConfig()` reproduces this table verbatim.

---

## Preprocessing (appendix, verbatim)

* prices ÷ the current closing price → relative price;
* volume ÷ the average trading volume in the window → relative volume;
* labels `ỹ_t = (price_{t+Δt+1} − price_{t+1}) / price_{t+1}` on the **vwap**,
  then cross-sectionally standardised `y_t = (ỹ_t − μ_t) / σ_t`;
* stocks with too many missing values are dropped, remaining gaps filled with 0,
  extreme values clipped.

Each of these is covered by a test in `tests/test_data.py`.

---

## Metrics

Prediction (`factorgcl/metrics.py`), computed per trading day then averaged:

```
IC_t = 1/N · (ŷ_t − mean ŷ_t)ᵀ(y_t − mean y_t) / (std ŷ_t · std y_t)
IC   = mean(IC_t)          ICIR = mean(IC_t) / std(IC_t)
```

plus MSE and the appendix's sign-based F1 (`TP = #{y>0 ∧ ŷ>0}`, `FP = #{y<0 ∧ ŷ>0}`,
`FN = #{y>0 ∧ ŷ<0}`).

Investment simulation (`factorgcl/backtest.py`): buy the TopK stocks by predicted
score each day, hold Δt days. Because a new basket is opened daily while the
previous ones are still held, capital is split into Δt tranches and one is rolled
per day. Reported as in the appendix:

```
CR(t) = Σ r(i)     CER(t) = Σ (r(i) − r_b(i))     AR = CER(T)/T × 252
IR    = E[r − r_b]/σ × √252                        RoMaD = AR / MaxDrawdown
```

---

## Two places the paper leaves a choice

Both are exposed as flags; the default is the literal reading of the text.

| Question | Default | Flag |
|---|---|---|
| Eq. (10) writes `e'_α` as a plain residual, while Eq. (7) puts the historical `e_α` through a linear layer + LeakyReLU. | literal Eq. (10) | `ModelConfig.future_alpha_uses_module=True` for the symmetric variant |
| The paper notes only that `φ'_prior` / `φ'_hidden` share parameters, which implies `φ'_feat` is separate (`T' = 20 ≠ T = 60`). | separate future encoder | `ModelConfig.share_feature_extractor=True` to tie them |

One genuine ambiguity in preprocessing: the appendix says the future window is
built "similarly" to the historical one, without saying what "the current
closing price" means for a forward window. The default anchors both windows on
the close of the prediction day `t` (`DataConfig.future_anchor="current"`);
`"window"` normalises the future window by its own last close.

The splits are cut on the *prediction* day, as the paper's 5y : 1y : 2y protocol
describes. A sample also reaches up to `max(T', Δt_max + 1) = 21` days forward
for its future window and longest label, so the tail of one window overlaps the
head of the next in calendar time. The paper does not purge that overlap and
neither does the default; `split_by_date(..., purge_days=21)` removes it if you
want a strictly non-overlapping evaluation.

---

## Tests

```bash
python -m pytest tests/ -q
```

The suite checks the implementation against the equations, not just the shapes:

* `test_hypergcn.py` — the layer's output equals Eq. (1) computed with explicit
  `D_n`, `H`, `W`, `D_e` matrices, for binary and soft incidence, with and
  without hyperedge weights; padding cannot leak into real stocks; isolated
  nodes stay finite.
* `test_losses.py` — the InfoNCE implementation equals a literal per-element
  transcription of Eq. (11); the total loss equals `L_mse + γ L_CL`.
* `test_model.py` — β_h stays in [0,1]; `e_α` is the residual after both betas;
  the future branch reuses the historical β and β_h through the shared modules;
  batching and padding are exact; each ablation zeroes its component.
* `test_data.py` — `close_t / close_t = 1` in the last row of every window,
  volumes average to 1, the label formula matches the appendix, splits do not
  leak.
* `test_metrics.py`, `test_backtest.py` — every metric against its definition.
* `test_integration.py` — training reduces the loss, beats an untrained model,
  the contrastive term decreases, the run is reproducible under seed 0.

---

## A note on reproducing the paper's numbers

The reported results use a proprietary China A-share dataset (5028 stocks,
2014-01-01 → 2023-06-30, 83 secondary-industry prior factors) and a Hong Kong
dataset (1834 stocks). Neither is redistributable, so this repository ships a
synthetic market instead. Its signal-to-noise ratio is far higher than a real
market's, so absolute IC values on synthetic data are **not** comparable to
Table 1 — point the pipeline at real data to compare.

## Citation

```bibtex
@inproceedings{duan2025factorgcl,
  title     = {FactorGCL: A Hypergraph-Based Factor Model with Temporal Residual
               Contrastive Learning for Stock Returns Prediction},
  author    = {Duan, Yitong and Wang, Weiran and Li, Jian},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI-25)},
  pages     = {173--180},
  year      = {2025}
}
```
