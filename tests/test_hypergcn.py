"""The HyperGCN layer must equal Eq. (1) computed with explicit matrices."""

import torch

from factorgcl.models.hypergcn import HypergraphConv


def reference_hypergcn(x, incidence, weight, edge_weight=None, eps=1e-8):
    """Literal transcription of e' = sigma(Dn^-1/2 H W De^-1 H^T Dn^-1/2 e w)."""
    num_edges = incidence.shape[1]
    edge_weight = torch.ones(num_edges) if edge_weight is None else edge_weight
    W = torch.diag(edge_weight)
    node_degree = (incidence * edge_weight.unsqueeze(0)).sum(dim=1).clamp(min=eps)
    edge_degree = incidence.sum(dim=0).clamp(min=eps)
    Dn = torch.diag(node_degree.pow(-0.5))
    De = torch.diag(edge_degree.pow(-1.0))
    propagation = Dn @ incidence @ W @ De @ incidence.t() @ Dn
    return torch.nn.functional.leaky_relu(propagation @ x @ weight)


def test_matches_explicit_matrix_formula():
    torch.manual_seed(0)
    num_nodes, num_edges, dim = 12, 5, 7
    layer = HypergraphConv(dim, dim)
    x = torch.randn(num_nodes, dim)
    incidence = (torch.rand(num_nodes, num_edges) > 0.5).float()
    incidence[:, 0] = 1.0  # make sure no hyperedge is empty

    expected = reference_hypergcn(x, incidence, layer.weight.weight.t())
    torch.testing.assert_close(layer(x, incidence), expected, rtol=1e-5, atol=1e-6)


def test_matches_formula_with_soft_incidence_and_edge_weights():
    torch.manual_seed(1)
    num_nodes, num_edges, dim = 9, 4, 6
    layer = HypergraphConv(dim, dim)
    x = torch.randn(num_nodes, dim)
    incidence = torch.rand(num_nodes, num_edges)  # soft hyperedges of the hidden module
    edge_weight = torch.rand(num_edges) + 0.1

    expected = reference_hypergcn(x, incidence, layer.weight.weight.t(), edge_weight)
    got = layer(x, incidence, edge_weight=edge_weight)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_batched_equals_per_sample():
    torch.manual_seed(2)
    batch, num_nodes, num_edges, dim = 3, 8, 4, 5
    layer = HypergraphConv(dim, dim)
    x = torch.randn(batch, num_nodes, dim)
    incidence = (torch.rand(batch, num_nodes, num_edges) > 0.4).float()
    incidence[:, :, 0] = 1.0

    batched = layer(x, incidence)
    for b in range(batch):
        torch.testing.assert_close(batched[b], layer(x[b], incidence[b]), rtol=1e-5, atol=1e-6)


def test_padding_does_not_change_real_nodes():
    torch.manual_seed(3)
    num_nodes, num_edges, dim = 10, 4, 6
    layer = HypergraphConv(dim, dim)
    x = torch.randn(num_nodes, dim)
    incidence = (torch.rand(num_nodes, num_edges) > 0.4).float()
    incidence[:, 0] = 1.0

    padded_x = torch.cat([x, torch.randn(4, dim)], dim=0)
    padded_incidence = torch.cat([incidence, torch.rand(4, num_edges)], dim=0)
    mask = torch.cat([torch.ones(num_nodes), torch.zeros(4)])

    out = layer(padded_x, padded_incidence, node_mask=mask)
    torch.testing.assert_close(out[:num_nodes], layer(x, incidence), rtol=1e-5, atol=1e-6)
    assert torch.all(out[num_nodes:] == 0)


def test_isolated_node_is_finite():
    """A stock exposed to no factor has degree 0 and must not produce inf/nan."""
    layer = HypergraphConv(4, 4)
    x = torch.randn(3, 4)
    incidence = torch.zeros(3, 2)
    incidence[0, 0] = 1.0
    assert torch.isfinite(layer(x, incidence)).all()
