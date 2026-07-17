import torch.optim as optim

try:
    from opacus import PrivacyEngine

    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False


# Held identical across every arm so the comparison isolates the optimizer.
# torch.optim.AdamW defaults to weight_decay=1e-2, so leaving it unset previously handed
# AdamW 100x the decay CASMO got (1e-4) and SGD none at all -- a confound, not a baseline.
WEIGHT_DECAY = 1e-4


def get_optimizer(model, optimizer_name, lr):
    if optimizer_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=WEIGHT_DECAY)
    elif optimizer_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    elif optimizer_name == "casmo":
        from casmo import CASMO

        return CASMO(
            model.parameters(),
            lr=lr,
            weight_decay=WEIGHT_DECAY,
            # DP noise is injected isotropically, so the relative (focus) axis does
            # the useful work here; heavy absolute suppression would only slow
            # convergence under a fixed privacy budget.
            robustness=0.5,
            c_min=0.1,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def make_private(model, optimizer, train_loader, noise_multiplier, max_grad_norm, epochs):
    if not OPACUS_AVAILABLE:
        raise ImportError("Opacus is not installed. Please install it to run DP benchmarks.")

    privacy_engine = PrivacyEngine()

    # Opacus 1.0+ style
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    return model, optimizer, train_loader, privacy_engine
