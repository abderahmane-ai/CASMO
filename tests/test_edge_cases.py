"""Tests for edge cases, degenerate inputs and error handling."""

import pytest
import torch

from casmo import CASMO


class TestGradientEdgeCases:
    def test_zero_gradients_leave_params_untouched(self):
        param = torch.nn.Parameter(torch.randn(5, 5))
        optimizer = CASMO([param], lr=1e-2)
        param.grad = torch.zeros(5, 5)
        optimizer.step()

        before = param.clone()
        optimizer.step()
        assert torch.allclose(before, param), "zero gradients must not move parameters"

    def test_very_small_gradients(self):
        param = torch.nn.Parameter(torch.randn(5, 5))
        optimizer = CASMO([param], lr=1e-3)
        for _ in range(20):
            param.grad = torch.ones(5, 5) * 1e-10
            optimizer.step()
        assert torch.isfinite(param).all()

    def test_very_large_gradients(self):
        param = torch.nn.Parameter(torch.randn(5, 5))
        optimizer = CASMO([param], lr=1e-3)
        for _ in range(20):
            param.grad = torch.ones(5, 5) * 1e6
            optimizer.step()
        assert torch.isfinite(param).all(), "large gradients must not produce NaN/Inf"

    def test_gradient_outliers_do_not_produce_nan(self):
        torch.manual_seed(0)
        param = torch.nn.Parameter(torch.randn(10, 10))
        optimizer = CASMO([param], lr=1e-3)
        for i in range(60):
            grad = torch.randn(10, 10)
            if i % 10 == 0:
                grad[torch.rand_like(grad) < 0.01] += 1000.0
            param.grad = grad
            optimizer.step()
            assert torch.isfinite(param).all()

    def test_mixed_none_and_present_gradients(self):
        params = [torch.nn.Parameter(torch.randn(3, 3)), torch.nn.Parameter(torch.randn(2, 2))]
        optimizer = CASMO(params, lr=1e-3)
        params[0].grad = torch.randn(3, 3)
        params[1].grad = None
        optimizer.step()
        assert params[1].grad is None

    def test_sparse_gradient_raises(self):
        param = torch.nn.Parameter(torch.randn(10, 10))
        optimizer = CASMO([param], lr=1e-3)
        param.grad = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [2, 3]]), torch.tensor([1.0, 2.0]), (10, 10)
        )
        with pytest.raises(NotImplementedError, match="sparse gradients"):
            optimizer.step()

    def test_nan_guard_raises_when_enabled(self):
        param = torch.nn.Parameter(torch.randn(5, 5))
        optimizer = CASMO([param], lr=1e-3, nan_guard=True)
        param.grad = torch.full((5, 5), float("nan"))
        with pytest.raises(RuntimeError, match="Non-finite"):
            optimizer.step()

    def test_inf_guard_raises_when_enabled(self):
        param = torch.nn.Parameter(torch.randn(5, 5))
        optimizer = CASMO([param], lr=1e-3, nan_guard=True)
        param.grad = torch.full((5, 5), float("inf"))
        with pytest.raises(RuntimeError, match="Non-finite"):
            optimizer.step()

    def test_nan_guard_disabled_by_default(self):
        """Default path must not pay for a host sync; it does not raise."""
        param = torch.nn.Parameter(torch.randn(5, 5))
        optimizer = CASMO([param], lr=1e-3)
        param.grad = torch.full((5, 5), float("nan"))
        optimizer.step()


class TestParameterEdgeCases:
    def test_empty_parameter_list_raises(self):
        with pytest.raises(ValueError, match="empty parameter list"):
            CASMO([], lr=1e-3)

    def test_single_parameter(self):
        param = torch.nn.Parameter(torch.randn(10))
        optimizer = CASMO([param], lr=1e-3)
        for _ in range(20):
            param.grad = torch.randn(10)
            optimizer.step()
        assert torch.isfinite(param).all()

    def test_single_element_parameter(self):
        param = torch.nn.Parameter(torch.randn(1))
        optimizer = CASMO([param], lr=1e-3)
        for _ in range(20):
            param.grad = torch.randn(1)
            optimizer.step()
        assert optimizer.group_metrics(0)["agar"] is not None

    def test_many_parameters(self):
        params = [torch.nn.Parameter(torch.randn(10, 10)) for _ in range(100)]
        optimizer = CASMO(params, lr=1e-3)
        for _ in range(5):
            for p in params:
                p.grad = torch.randn_like(p)
            optimizer.step()
        assert all(torch.isfinite(p).all() for p in params)

    def test_optimizer_state_matches_adam_footprint(self):
        """CASMO tracks exactly two EMAs per parameter, like Adam."""
        param = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = CASMO([param], lr=1e-3)
        param.grad = torch.randn(4, 4)
        optimizer.step()
        assert set(optimizer.state[param].keys()) == {"step", "m", "s"}
