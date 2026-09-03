"""The cascading residual architecture of Figure 3 / Eqs. (5)-(10)."""

import torch

from factorgcl.config import ModelConfig
from factorgcl.models.factorgcl import FactorGCL


def make_inputs(num_stocks=17, seq_len=60, future_len=20, num_features=6, num_factors=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(num_stocks, seq_len, num_features, generator=generator)
    future_x = torch.randn(num_stocks, future_len, num_features, generator=generator)
    prior = torch.zeros(num_stocks, num_factors)
    prior[torch.arange(num_stocks), torch.randint(0, num_factors, (num_stocks,), generator=generator)] = 1.0
    return x, future_x, prior


def base_config(**overrides):
    config = ModelConfig(
        input_dim=6, hidden_dim=16, num_hidden_factors=8, num_prior_factors=5, num_periods=4
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def test_output_shapes():
    model = base_config()
    net = FactorGCL(model).eval()
    x, future_x, prior = make_inputs()
    out = net(x, prior, future_x=future_x)

    assert out.prediction.shape == (17, 4)
    assert out.hidden_exposure.shape == (17, 8)
    assert out.prior_embedding.shape == (17, 16)
    assert out.hidden_embedding.shape == (17, 16)
    assert out.alpha_embedding.shape == (17, 16)
    assert out.future_alpha_embedding.shape == (17, 16)


def test_hidden_exposures_are_soft_hyperedges():
    """"we generate 'soft' hyperedges beta_h [...] with values ranging between [0, 1]"."""
    net = FactorGCL(base_config()).eval()
    x, _, prior = make_inputs()
    exposure = net(x, prior).hidden_exposure
    assert torch.all(exposure >= 0) and torch.all(exposure <= 1)


def test_prediction_is_the_sum_of_the_three_components():
    """Eq. (8): y = w_o1 e_p + w_o2 e_h + w_o3 e_a + b_o."""
    net = FactorGCL(base_config()).eval()
    x, _, prior = make_inputs()
    out = net(x, prior)
    head = net.prediction_head
    manual = (
        head.prior_head(out.prior_embedding)
        + head.hidden_head(out.hidden_embedding)
        + head.alpha_head(out.alpha_embedding)
        + head.bias
    )
    torch.testing.assert_close(out.prediction, manual)


def test_alpha_is_the_residual_after_both_betas():
    """Eq. (7): e_a = LeakyReLU(w_a (e_s - e_p - e_h) + b_a)."""
    net = FactorGCL(base_config()).eval()
    x, _, prior = make_inputs()
    out = net(x, prior)
    residual = out.stock_embedding - out.prior_embedding - out.hidden_embedding
    torch.testing.assert_close(out.alpha_embedding, net.alpha_module(residual))


def test_hidden_exposure_comes_from_the_post_prior_residual():
    """beta_h = Sigmoid(e_r c^T) with e_r = e_s - e_p."""
    net = FactorGCL(base_config()).eval()
    x, _, prior = make_inputs()
    out = net(x, prior)
    residual = out.stock_embedding - out.prior_embedding
    expected = torch.sigmoid(residual @ net.hidden_module.prototypes.t())
    torch.testing.assert_close(out.hidden_exposure, expected)


def test_future_branch_reuses_historical_exposures():
    """Eq. (10) runs the shared beta modules on x' with beta and beta_h fixed."""
    net = FactorGCL(base_config()).eval()
    x, future_x, prior = make_inputs()
    out = net(x, prior, future_x=future_x)

    future_embedding = net.future_feature_extractor(future_x)
    future_prior = net.prior_module(future_embedding, prior)
    future_residual = future_embedding - future_prior
    future_hidden = net.hidden_module.propagate(future_residual, out.hidden_exposure)
    expected = future_embedding - future_prior - future_hidden
    torch.testing.assert_close(out.future_alpha_embedding, expected)


def test_beta_modules_are_shared_between_the_two_branches():
    net = FactorGCL(base_config())
    prior_params = {id(p) for p in net.prior_module.parameters()}
    hidden_params = {id(p) for p in net.hidden_module.parameters()}
    all_params = {id(p) for p in net.parameters()}
    assert prior_params <= all_params and hidden_params <= all_params
    # exactly one prior and one hidden module exist, so both branches share them
    assert sum(1 for _ in net.named_modules() if isinstance(_[1], type(net.prior_module))) == 1


def test_separate_future_feature_extractor_by_default():
    net = FactorGCL(base_config())
    assert net.future_feature_extractor is not net.feature_extractor
    tied = FactorGCL(base_config(share_feature_extractor=True))
    assert tied.future_feature_extractor is tied.feature_extractor


def test_ablations_zero_out_their_component():
    x, future_x, prior = make_inputs()

    without_prior = FactorGCL(base_config(use_prior=False)).eval()(x, prior)
    assert torch.all(without_prior.prior_embedding == 0)

    without_hidden = FactorGCL(base_config(use_hidden=False)).eval()(x, prior)
    assert torch.all(without_hidden.hidden_embedding == 0)

    without_alpha = FactorGCL(base_config(use_alpha=False)).eval()(x, prior)
    assert torch.all(without_alpha.alpha_embedding == 0)


def test_batched_matches_single_day():
    net = FactorGCL(base_config()).eval()
    x, future_x, prior = make_inputs()
    single = net(x, prior, future_x=future_x)
    batched = net(
        torch.stack([x, x]), torch.stack([prior, prior]), future_x=torch.stack([future_x, future_x])
    )
    # the feature extractor's BatchNorm is in eval mode, so batching is exact
    torch.testing.assert_close(batched.prediction[0], single.prediction, rtol=1e-5, atol=1e-6)


def test_padded_stocks_do_not_affect_real_predictions():
    net = FactorGCL(base_config()).eval()
    x, future_x, prior = make_inputs(num_stocks=12)
    pad_x, pad_future, pad_prior = make_inputs(num_stocks=5, seed=9)

    full_x = torch.cat([x, pad_x])
    full_future = torch.cat([future_x, pad_future])
    full_prior = torch.cat([prior, pad_prior])
    mask = torch.cat([torch.ones(12), torch.zeros(5)])

    padded = net(
        full_x.unsqueeze(0),
        full_prior.unsqueeze(0),
        future_x=full_future.unsqueeze(0),
        node_mask=mask.unsqueeze(0),
    )
    clean = net(x, prior, future_x=future_x)
    torch.testing.assert_close(padded.prediction[0, :12], clean.prediction, rtol=1e-5, atol=1e-6)


def test_gradients_reach_every_component():
    net = FactorGCL(base_config())
    x, future_x, prior = make_inputs()
    out = net(x, prior, future_x=future_x)
    loss = out.prediction.sum() + net.project(out.future_alpha_embedding).sum()
    loss.backward()

    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has a non-finite gradient"


def test_paper_default_sizes():
    """M = 32 hidden factors, H = 32 embedding, 2 GRU layers."""
    config = ModelConfig()
    assert (config.num_hidden_factors, config.hidden_dim, config.num_rnn_layers) == (32, 32, 2)
    net = FactorGCL(config)
    assert net.feature_extractor.rnn.num_layers == 2
    assert net.hidden_module.prototypes.shape == (32, 32)
