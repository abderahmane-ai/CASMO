"""End-to-end integration tests: convergence, parity with Adam, real training loops."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from casmo import CASMO


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def make_dataset(n_samples=500, input_dim=10, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(n_samples, input_dim)
    w = torch.randn(input_dim)
    y = ((x @ w) > 0).long()
    return TensorDataset(x, y)


def train(model, optimizer, loader, epochs=20):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total = 0.0
    for _ in range(epochs):
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item()
    return total / len(loader)


class TestIntegration:
    def test_converges_on_simple_problem(self):
        loader = DataLoader(make_dataset(), batch_size=32, shuffle=True)
        torch.manual_seed(42)
        model = SimpleMLP()
        optimizer = CASMO(model.parameters(), lr=3e-3)
        assert train(model, optimizer, loader) < 0.5

    def test_converges_with_weight_decay(self):
        loader = DataLoader(make_dataset(), batch_size=32, shuffle=True)
        torch.manual_seed(42)
        model = SimpleMLP()
        optimizer = CASMO(model.parameters(), lr=3e-3, weight_decay=0.01)
        assert train(model, optimizer, loader) < 0.6

    def test_comparable_to_adam_on_clean_data(self):
        dataset = make_dataset()

        torch.manual_seed(42)
        model_c = SimpleMLP()
        loss_c = train(
            model_c,
            CASMO(model_c.parameters(), lr=3e-3),
            DataLoader(dataset, batch_size=32, shuffle=False),
        )

        torch.manual_seed(42)
        model_a = SimpleMLP()
        loss_a = train(
            model_a,
            torch.optim.Adam(model_a.parameters(), lr=3e-3),
            DataLoader(dataset, batch_size=32, shuffle=False),
        )

        assert abs(loss_c - loss_a) < 0.2, f"CASMO {loss_c:.3f} vs Adam {loss_a:.3f}"

    def test_multiple_parameter_groups(self):
        loader = DataLoader(make_dataset(300), batch_size=32, shuffle=True)
        torch.manual_seed(42)
        model = SimpleMLP()
        optimizer = CASMO(
            [
                {"params": model.fc1.parameters(), "lr": 3e-3},
                {"params": model.fc2.parameters(), "lr": 1e-3},
            ]
        )
        assert train(model, optimizer, loader, epochs=15) < 0.7

    def test_per_group_robustness(self):
        """Different robustness per group must train without error."""
        loader = DataLoader(make_dataset(300), batch_size=32, shuffle=True)
        torch.manual_seed(42)
        model = SimpleMLP()
        optimizer = CASMO(
            [
                {"params": model.fc1.parameters(), "robustness": 1.0},
                {"params": model.fc2.parameters(), "robustness": 0.0},
            ],
            lr=3e-3,
        )
        train(model, optimizer, loader, epochs=10)
        assert optimizer.group_metrics(0)["confidence"] is not None
        assert optimizer.group_metrics(1)["confidence"] is not None

    def test_gradient_accumulation(self):
        loader = DataLoader(make_dataset(300), batch_size=16, shuffle=True)
        torch.manual_seed(42)
        model = SimpleMLP()
        optimizer = CASMO(model.parameters(), lr=3e-3)
        criterion = nn.CrossEntropyLoss()
        accumulation = 2

        for _ in range(10):
            for i, (x, y) in enumerate(loader):
                (criterion(model(x), y) / accumulation).backward()
                if (i + 1) % accumulation == 0:
                    optimizer.step()
                    optimizer.zero_grad()

        assert all(torch.isfinite(p).all() for p in model.parameters())

    def test_checkpoint_and_resume(self):
        loader = DataLoader(make_dataset(300), batch_size=32, shuffle=True)
        torch.manual_seed(42)
        model = SimpleMLP()
        optimizer = CASMO(model.parameters(), lr=3e-3)
        train(model, optimizer, loader, epochs=5)

        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}

        model2 = SimpleMLP()
        optimizer2 = CASMO(model2.parameters(), lr=3e-3)
        model2.load_state_dict(checkpoint["model"])
        optimizer2.load_state_dict(checkpoint["optimizer"])

        assert train(model2, optimizer2, loader, epochs=10) < 0.7

    def test_works_with_lr_scheduler_end_to_end(self):
        loader = DataLoader(make_dataset(300), batch_size=32, shuffle=True)
        torch.manual_seed(42)
        model = SimpleMLP()
        optimizer = CASMO(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        criterion = nn.CrossEntropyLoss()

        for _ in range(10):
            for x, y in loader:
                optimizer.zero_grad()
                criterion(model(x), y).backward()
                optimizer.step()
            scheduler.step()

        assert optimizer.param_groups[0]["lr"] < 1e-2
        assert all(torch.isfinite(p).all() for p in model.parameters())
