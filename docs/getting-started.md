# Getting Started

## Installation

```bash
pip install casmo-optimizer
```

From source:

```bash
git clone https://github.com/abderahmane-ai/CASMO.git
cd CASMO
pip install -e .
```

Requires Python 3.8+, PyTorch 1.10+, NumPy. Verify:

```bash
python verify_installation.py
```

---

## Your first training loop

CASMO is a drop-in replacement for `torch.optim.AdamW`:

```python
import torch
import torch.nn as nn
from casmo import CASMO

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3))
optimizer = CASMO(model.parameters(), lr=1e-3, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for x, y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
```

That is the whole integration. No calibration, no warm-up, no scheduling of CASMO's own
parameters.

---

## Choosing `robustness`

`robustness` is the one knob worth tuning. It sets how aggressively CASMO slows down when
gradients are noise-dominated.

```python
CASMO(model.parameters(), lr=1e-3, robustness=1.0)   # noisy labels
CASMO(model.parameters(), lr=1e-3)                    # default 0.5, general purpose
CASMO(model.parameters(), lr=1e-3, robustness=0.0)   # clean data / DP-SGD / max speed
```

| Your data | Start with |
|---|---|
| Crowd-sourced or web-scraped labels | `1.0` |
| Long-tailed / imbalanced classes | `1.0` |
| Sequential fine-tuning (continual learning) | `1.0` |
| Curated, clean labels | `0.0` – `0.5` |
| DP-SGD (injected isotropic gradient noise) | `0.0` – `0.5` |
| Unsure | `0.5` (the default) |

Higher `robustness` makes the training loss fall more slowly. On noisy data that is the
point — the model is declining to memorize. Watch **validation** accuracy, not training
loss, when tuning it.

---

## Learning rate

Start with the LR you already use for AdamW. CASMO is more tolerant of aggressive
learning rates than Adam (measured: 36 steps to converge at `lr=3e-2` versus Adam's 70),
so if your LR is currently limited by instability, try raising it.

---

## Monitoring what CASMO sees

`group_metrics()` exposes the optimizer's live view of gradient quality:

```python
optimizer.step()
m = optimizer.group_metrics(0)
print(f"agar={m['agar']:.3f} confidence={m['confidence']:.3f}")
```

- **`agar`** — the mean signal fraction, in `[0, 1]`. Near 1 means consistent gradients;
  near 0 means noise.
- **`confidence`** — the mean multiplier actually applied to the update.

Reading these is the fastest way to understand a run. If `agar` collapses when you add a
data source, that source is probably noisy.

---

## Parameter groups

Everything is settable per group, including `robustness`:

```python
optimizer = CASMO([
    {"params": model.backbone.parameters(), "lr": 1e-4, "robustness": 1.0},
    {"params": model.head.parameters(),     "lr": 1e-3, "robustness": 0.0},
])
```

This is useful when fine-tuning: protect the pretrained backbone with high robustness
while letting a freshly initialized head move quickly.

---

## Checkpointing

Standard PyTorch semantics; resuming reproduces uninterrupted training exactly.

```python
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "ckpt.pt")

ckpt = torch.load("ckpt.pt")
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
```

---

## Next steps

- [API Reference](api-reference.md) — every parameter and method.
- [FAQ](faq.md) — common questions and troubleshooting.
- [Migration Guide](migration-guide.md) — coming from AdamW or CASMO v0.3.
- [Theory](../CASMO_THEORY.md) — the derivation.
- [Redesign evidence](../research/REDESIGN.md) — the experiments behind the design.
