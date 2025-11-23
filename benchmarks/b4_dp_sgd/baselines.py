import torch
import torch.optim as optim
try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

def get_optimizer(model, optimizer_name, lr):
    if optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'casmo':
        # Assuming CASMO is available in the path as per original train.py
        from casmo import CASMO
        return CASMO(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,
            granularity='group',
            tau_init_steps=100,
            c_min=0.1,
            log_level=0
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
