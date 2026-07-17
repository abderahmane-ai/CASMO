"""Tests for checkpointing: state_dict save/load and training resumption."""

import io

import torch
import torch.nn as nn

from casmo import CASMO


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def _train(model, optimizer, x, y, steps):
    for _ in range(steps):
        optimizer.zero_grad()
        nn.CrossEntropyLoss()(model(x), y).backward()
        optimizer.step()


class TestStateDict:
    def test_state_dict_contains_moments(self):
        torch.manual_seed(0)
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-3)
        _train(model, optimizer, torch.randn(4, 10), torch.randint(0, 2, (4,)), 10)

        state_dict = optimizer.state_dict()
        assert "state" in state_dict and state_dict["state"]
        first = list(state_dict["state"].values())[0]
        assert {"step", "m", "s"} <= set(first.keys())

    def test_save_and_load_round_trip(self):
        torch.manual_seed(0)
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-3)
        _train(model, optimizer, torch.randn(4, 10), torch.randint(0, 2, (4,)), 20)

        buf = io.BytesIO()
        torch.save(optimizer.state_dict(), buf)
        buf.seek(0)

        restored = CASMO(model.parameters(), lr=1e-3)
        restored.load_state_dict(torch.load(buf, weights_only=False))

        for p in model.parameters():
            assert torch.allclose(optimizer.state[p]["m"], restored.state[p]["m"])
            assert torch.allclose(optimizer.state[p]["s"], restored.state[p]["s"])
            assert optimizer.state[p]["step"] == restored.state[p]["step"]

    def test_resumed_training_matches_continuous(self):
        """The headline checkpoint guarantee: resume == never having stopped."""
        torch.manual_seed(1)
        x, y = torch.randn(20, 10), torch.randint(0, 2, (20,))

        torch.manual_seed(2)
        model_a = SimpleModel()
        opt_a = CASMO(model_a.parameters(), lr=1e-2)
        _train(model_a, opt_a, x, y, 10)

        buf = io.BytesIO()
        torch.save({"m": model_a.state_dict(), "o": opt_a.state_dict()}, buf)
        buf.seek(0)
        ckpt = torch.load(buf, weights_only=False)

        model_b = SimpleModel()
        opt_b = CASMO(model_b.parameters(), lr=1e-2)
        model_b.load_state_dict(ckpt["m"])
        opt_b.load_state_dict(ckpt["o"])
        _train(model_b, opt_b, x, y, 10)

        torch.manual_seed(2)
        model_c = SimpleModel()
        opt_c = CASMO(model_c.parameters(), lr=1e-2)
        _train(model_c, opt_c, x, y, 20)

        for p_b, p_c in zip(model_b.parameters(), model_c.parameters()):
            assert torch.allclose(p_b, p_c, atol=1e-6), "resumed run must match continuous run"

    def test_hyperparameters_survive_round_trip(self):
        model = SimpleModel()
        optimizer = CASMO(model.parameters(), lr=1e-3, robustness=0.8, c_min=0.25, rel_floor=0.2)
        _train(model, optimizer, torch.randn(4, 10), torch.randint(0, 2, (4,)), 5)

        restored = CASMO(model.parameters(), lr=1e-3)
        restored.load_state_dict(optimizer.state_dict())

        assert restored.param_groups[0]["robustness"] == 0.8
        assert restored.param_groups[0]["c_min"] == 0.25
        assert restored.param_groups[0]["rel_floor"] == 0.2

    def test_multiple_parameter_groups(self):
        model = SimpleModel()
        optimizer = CASMO(
            [
                {"params": [model.fc.weight], "lr": 1e-3},
                {"params": [model.fc.bias], "lr": 1e-4},
            ]
        )
        _train(model, optimizer, torch.randn(4, 10), torch.randint(0, 2, (4,)), 20)

        restored = CASMO(
            [
                {"params": [model.fc.weight], "lr": 1e-3},
                {"params": [model.fc.bias], "lr": 1e-4},
            ]
        )
        restored.load_state_dict(optimizer.state_dict())

        assert len(restored.param_groups) == 2
        assert restored.param_groups[0]["lr"] == 1e-3
        assert restored.param_groups[1]["lr"] == 1e-4
