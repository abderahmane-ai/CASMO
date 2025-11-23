"""
GAN Stability Benchmark: Tiny ImageNet with DCGAN
Benchmark ID: B5

Tests CASMO's ability to stabilize GAN training under high-variance adversarial gradients.

Task:
    Dataset: Tiny ImageNet (native 64×64, 20k subset, 237MB download)
    Architecture: DCGAN (Radford et al. 2016)
    Optimizer Configs: All combinations of CASMO/Adam for G and D
    
    Hypothesis:
    - Adam/Adam: Baseline performance, potential instability
    - CASMO on G: Stabilizes generator under noisy D feedback
    - CASMO on D: Prevents discriminator from overpowering generator
    - CASMO/CASMO: Most stable training dynamics

Hardware Target: RTX 4050 Laptop (6GB VRAM)
Expected Runtime: ~15 minutes per config
"""

import sys
import os
import argparse
import time
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

# Add parent directory to import casmo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from casmo import CASMO

# -----------------------------------------------------------------------------
# 1. DCGAN Architecture for 64×64 Images
# -----------------------------------------------------------------------------

def weights_init(m):
    """Initialize weights with N(0, 0.02) as per DCGAN paper."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """DCGAN Generator for 64×64 images."""
    def __init__(self, z_dim=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        
        # Input: (batch, z_dim, 1, 1)
        # Output: (batch, 3, 64, 64)
        self.main = nn.Sequential(
            # Input: z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State: (ngf*8) x 4 x 4
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State: (ngf*4) x 8 x 8
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State: (ngf*2) x 16 x 16
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: (ngf) x 32 x 32
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: nc x 64 x 64
        )
    
    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """DCGAN Discriminator for 64×64 images."""
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        
        # Input: (batch, 3, 64, 64)
        # Output: (batch, 1)
        self.main = nn.Sequential(
            # Input: nc x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: ndf x 32 x 32
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*2) x 16 x 16
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*4) x 8 x 8
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*8) x 4 x 4
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output: 1 x 1 x 1
        )
    
    def forward(self, img):
        output = self.main(img)
        return output.view(-1, 1).squeeze(1)


# -----------------------------------------------------------------------------
# 2. Utilities
# -----------------------------------------------------------------------------

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_agar_confidence(optimizer):
    """Extract current AGAR and confidence from CASMO optimizer."""
    if not hasattr(optimizer, '_group_states'):
        return None, None
    group_state = optimizer._group_states.get(0, {})
    return group_state.get('current_agar'), group_state.get('current_confidence')


def compute_gradient_variance(model):
    """Compute variance of gradients across all parameters."""
    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    
    if len(grad_norms) == 0:
        return 0.0
    return float(np.var(grad_norms))


def create_optimizers(G, D, config, lr_g=0.0002, lr_d=0.0002, betas=(0.5, 0.999), 
                     tau_init_steps=500):
    """
    Create optimizers based on configuration string.
    
    Args:
        G: Generator model
        D: Discriminator model
        config: String like 'casmo_adam', 'adam_casmo', 'casmo_casmo', or 'adam_adam'
        lr_g: Generator learning rate
        lr_d: Discriminator learning rate
        betas: Adam/CASMO beta parameters
        tau_init_steps: CASMO calibration steps
    
    Returns:
        opt_g, opt_d: Optimizer instances
    """
    g_opt_name, d_opt_name = config.split('_')
    
    # Generator optimizer
    if g_opt_name == 'casmo':
        opt_g = CASMO(
            G.parameters(),
            lr=lr_g,
            betas=betas,
            weight_decay=0.0,
            tau_init_steps=tau_init_steps,
            granularity='group',
            c_min=0.1,
            tau_dead_zone=0.2,
            log_level=1
        )
    else:
        opt_g = optim.Adam(G.parameters(), lr=lr_g, betas=betas)
    
    # Discriminator optimizer
    if d_opt_name == 'casmo':
        opt_d = CASMO(
            D.parameters(),
            lr=lr_d,
            betas=betas,
            weight_decay=0.0,
            tau_init_steps=tau_init_steps,
            granularity='group',
            c_min=0.1,
            tau_dead_zone=0.2,
            log_level=1
        )
    else:
        opt_d = optim.Adam(D.parameters(), lr=lr_d, betas=betas)
    
    return opt_g, opt_d


def save_sample_grid(generator, epoch, save_dir, n_images=64, z_dim=100, device='cuda'):
    """Generate and save a grid of sample images."""
    generator.eval()
    with torch.no_grad():
        # Generate samples
        z = torch.randn(n_images, z_dim, 1, 1, device=device)
        fake_imgs = generator(z)
        
        # Denormalize from [-1, 1] to [0, 1]
        fake_imgs = (fake_imgs + 1) / 2.0
        
        # Create grid
        grid = torchvision.utils.make_grid(fake_imgs, nrow=8, padding=2, normalize=False)
        
        # Save
        save_path = os.path.join(save_dir, f'samples_epoch_{epoch:03d}.png')
        torchvision.utils.save_image(grid, save_path)
    
    generator.train()
    return save_path


# -----------------------------------------------------------------------------
# 3. Dataset Loading
# -----------------------------------------------------------------------------

def load_tiny_imagenet(data_root='./data', dataset_size=20000, batch_size=16, 
                        num_workers=2, quick_test=False):
    """
    Load Tiny ImageNet dataset (perfect for GANs, ~237MB download).
    
    Tiny ImageNet has 100k training images (200 classes, native 64×64 resolution).
    Auto-downloads and extracts on first run.
    
    Args:
        data_root: Root directory for dataset
        dataset_size: Number of samples to use (subset for speed)
        batch_size: Batch size
        num_workers: DataLoader workers
        quick_test: If True, use tiny subset for testing
    
    Returns:
        DataLoader instance
    """
    import urllib.request
    import zipfile
    from pathlib import Path
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Tiny ImageNet paths
    tiny_imagenet_dir = Path(data_root) / 'tiny-imagenet-200'
    train_dir = tiny_imagenet_dir / 'train'
    zip_path = Path(data_root) / 'tiny-imagenet-200.zip'
    
    # Download if needed
    if not tiny_imagenet_dir.exists():
        print(f"Downloading Tiny ImageNet (~237MB)...")
        Path(data_root).mkdir(parents=True, exist_ok=True)
        
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        urllib.request.urlretrieve(url, zip_path)
        
        print(f"Extracting Tiny ImageNet...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_root)
        
        zip_path.unlink()  # Delete zip after extraction
        print(f"✅ Tiny ImageNet ready!")
    
    # Load dataset using ImageFolder
    print(f"Loading Tiny ImageNet dataset...")
    full_dataset = torchvision.datasets.ImageFolder(
        root=str(train_dir),
        transform=transform
    )
    
    # Use subset for speed
    if quick_test:
        subset_size = 100
    else:
        subset_size = min(dataset_size, len(full_dataset))
    
    indices = list(range(subset_size))
    dataset = Subset(full_dataset, indices)
    
    print(f"Using {subset_size} images from Tiny ImageNet (total: {len(full_dataset)} available)")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


# -----------------------------------------------------------------------------
# 4. Training Loop
# -----------------------------------------------------------------------------

def train_gan(config, device, dataloader, num_epochs=50, lr_g=0.0002, lr_d=0.0002,
             z_dim=100, save_dir='./results', seed=42):
    """
    Train GAN with specified optimizer configuration.
    
    Args:
        config: Optimizer config ('casmo_adam', 'adam_casmo', 'casmo_casmo', 'adam_adam')
        device: torch device
        dataloader: LSUN dataloader
        num_epochs: Number of training epochs
        lr_g, lr_d: Learning rates for G and D
        z_dim: Latent dimension
        save_dir: Directory to save results
        seed: Random seed
    
    Returns:
        Dictionary of training metrics
    """
    print(f"\n{'='*70}")
    print(f"Training Config: {config.upper()}")
    print(f"{'='*70}\n")
    
    set_seed(seed)
    
    # Initialize models
    G = Generator(z_dim=z_dim).to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)
    
    print(f"Generator parameters: {sum(p.numel() for p in G.parameters())/1e6:.2f}M")
    print(f"Discriminator parameters: {sum(p.numel() for p in D.parameters())/1e6:.2f}M")
    
    # Calculate total steps for tau_init_steps
    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * num_epochs
    # Use CASMO recommendation: max(500, int(0.05 * total_steps))
    tau_init_steps = max(500, int(0.05 * total_steps))
    print(f"Total steps: {total_steps}, tau_init_steps: {tau_init_steps}")
    
    # Create optimizers
    opt_g, opt_d = create_optimizers(G, D, config, lr_g, lr_d, tau_init_steps=tau_init_steps)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Results tracking
    results = {
        'config': config,
        'g_losses': [],
        'd_losses': [],
        'd_real_accs': [],
        'd_fake_accs': [],
        'g_grad_vars': [],
        'd_grad_vars': [],
        'steps': [],
        'epochs': [],
        'agar_g': [],
        'conf_g': [],
        'agar_d': [],
        'conf_d': [],
    }
    
    # Training
    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)
    start_time = time.time()
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        g_loss_epoch = 0
        d_loss_epoch = 0
        d_real_correct = 0
        d_fake_correct = 0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, (real_imgs, _) in enumerate(pbar):
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)
            total_samples += b_size
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            opt_d.zero_grad()
            
            # Real images (label smoothing: 0.9 instead of 1.0)
            label_real = torch.full((b_size,), 0.9, device=device)
            output_real = D(real_imgs)
            d_real_loss = criterion(output_real, label_real)
            
            # Fake images
            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            fake_imgs = G(noise)
            label_fake = torch.full((b_size,), 0.0, device=device)
            output_fake = D(fake_imgs.detach())
            d_fake_loss = criterion(output_fake, label_fake)
            
            # Combined D loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            opt_d.step()
            
            # Track D accuracy
            d_real_correct += (output_real > 0.5).sum().item()
            d_fake_correct += (output_fake < 0.5).sum().item()
            d_loss_epoch += d_loss.item()
            
            # -----------------
            #  Train Generator
            # -----------------
            opt_g.zero_grad()
            
            # Generate new fake images
            noise = torch.randn(b_size, z_dim, 1, 1, device=device)
            fake_imgs = G(noise)
            output = D(fake_imgs)
            
            # Generator loss (non-saturating)
            label_g = torch.full((b_size,), 1.0, device=device)
            g_loss = criterion(output, label_g)
            g_loss.backward()
            opt_g.step()
            
            g_loss_epoch += g_loss.item()
            
            # Track gradient variance every 50 steps
            if global_step % 50 == 0:
                g_grad_var = compute_gradient_variance(G)
                d_grad_var = compute_gradient_variance(D)
                results['g_grad_vars'].append(g_grad_var)
                results['d_grad_vars'].append(d_grad_var)
                results['steps'].append(global_step)
                
                # Track AGAR/Confidence for CASMO optimizers
                agar_g, conf_g = get_agar_confidence(opt_g)
                agar_d, conf_d = get_agar_confidence(opt_d)
                
                if agar_g is not None:
                    results['agar_g'].append(agar_g)
                    results['conf_g'].append(conf_g)
                if agar_d is not None:
                    results['agar_d'].append(agar_d)
                    results['conf_d'].append(conf_d)
            
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}',
                'D(x)': f'{output_real.mean().item():.4f}',
                'D(G(z))': f'{output_fake.mean().item():.4f}'
            })
        
        # Epoch statistics
        avg_g_loss = g_loss_epoch / len(dataloader)
        avg_d_loss = d_loss_epoch / len(dataloader)
        d_real_acc = 100. * d_real_correct / total_samples
        d_fake_acc = 100. * d_fake_correct / total_samples
        
        results['g_losses'].append(avg_g_loss)
        results['d_losses'].append(avg_d_loss)
        results['d_real_accs'].append(d_real_acc)
        results['d_fake_accs'].append(d_fake_acc)
        results['epochs'].append(epoch)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
              f"D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}, "
              f"D_real_acc: {d_real_acc:.1f}%, D_fake_acc: {d_fake_acc:.1f}%")
        
        # Save sample images every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            sample_path = save_sample_grid(G, epoch + 1, save_dir, device=device)
            print(f"  Saved samples: {sample_path}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_{config}_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'opt_g_state_dict': opt_g.state_dict(),
                'opt_d_state_dict': opt_d.state_dict(),
            }, checkpoint_path)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes")
    
    # Save final models
    final_path = os.path.join(save_dir, f'final_{config}.pth')
    torch.save({
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'config': config,
        'results': results
    }, final_path)
    
    return results


# -----------------------------------------------------------------------------
# 5. Plotting
# -----------------------------------------------------------------------------

def plot_results(all_results, save_dir):
    """Create comprehensive comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Colors for different configs
    colors = {
        'adam_adam': 'orange',
        'casmo_adam': 'green',
        'adam_casmo': 'blue',
        'casmo_casmo': 'purple'
    }
    
    # 1. Generator Loss
    ax = axes[0, 0]
    for config, results in all_results.items():
        epochs = results['epochs']
        ax.plot(epochs, results['g_losses'], label=config.replace('_', '+').upper(), 
                color=colors[config], linewidth=2, alpha=0.8)
    ax.set_title('Generator Loss', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Discriminator Loss
    ax = axes[0, 1]
    for config, results in all_results.items():
        epochs = results['epochs']
        ax.plot(epochs, results['d_losses'], label=config.replace('_', '+').upper(),
                color=colors[config], linewidth=2, alpha=0.8)
    ax.set_title('Discriminator Loss', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Gradient Variance (Generator)
    ax = axes[1, 0]
    for config, results in all_results.items():
        if len(results['g_grad_vars']) > 0:
            # Smooth gradient variance for better visualization
            steps = results['steps']
            g_vars = results['g_grad_vars']
            # Rolling average
            window = 20
            if len(g_vars) >= window:
                smoothed = np.convolve(g_vars, np.ones(window)/window, mode='valid')
                steps_smoothed = steps[:len(smoothed)]
                ax.plot(steps_smoothed, smoothed, label=config.replace('_', '+').upper(),
                       color=colors[config], linewidth=2, alpha=0.8)
    ax.set_title('Generator Gradient Variance (Smoothed)', fontsize=14)
    ax.set_xlabel('Step')
    ax.set_ylabel('Variance')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. AGAR/Confidence (CASMO only)
    ax = axes[1, 1]
    for config, results in all_results.items():
        if len(results['agar_g']) > 0:
            steps = results['steps'][:len(results['agar_g'])]
            ax.plot(steps, results['agar_g'], label=f'{config} G-AGAR',
                   color=colors[config], linewidth=2, alpha=0.6, linestyle='-')
        if len(results['agar_d']) > 0:
            steps = results['steps'][:len(results['agar_d'])]
            ax.plot(steps, results['agar_d'], label=f'{config} D-AGAR',
                   color=colors[config], linewidth=2, alpha=0.6, linestyle='--')
    ax.set_title('AGAR Evolution (CASMO Optimizers)', fontsize=14)
    ax.set_xlabel('Step')
    ax.set_ylabel('AGAR')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'comparison_plots.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Comparison plots saved to {save_path}")
    plt.close()


# -----------------------------------------------------------------------------
# 6. Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='B7: GAN Stability Benchmark')
    parser.add_argument('--config', type=str, default='adam_adam',
                       choices=['adam_adam', 'casmo_adam', 'adam_casmo', 'casmo_casmo'],
                       help='Optimizer configuration')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--dataset_size', type=int, default=20000, help='Number of Tiny ImageNet samples to use')
    parser.add_argument('--lr_g', type=float, default=0.0002, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=0.0002, help='Discriminator learning rate')
    parser.add_argument('--z_dim', type=int, default=100, help='Latent dimension')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--quick_test', action='store_true', help='Quick test with 100 samples')
    parser.add_argument('--data_root', type=str, default='./data', help='Data directory')
    args = parser.parse_args()
    
    print("="*70)
    print("B7: GAN Stability Benchmark - Tiny ImageNet 64×64")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available. This benchmark requires a GPU.")
        print("Exiting...")
        return
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load dataset
    dataloader = load_tiny_imagenet(
        data_root=args.data_root,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        quick_test=args.quick_test
    )
    
    # Train
    results = train_gan(
        config=args.config,
        device=device,
        dataloader=dataloader,
        num_epochs=args.epochs,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        z_dim=args.z_dim,
        save_dir=results_dir
    )
    
    # Save results
    results_path = os.path.join(results_dir, f'metrics_{args.config}.json')
    # Convert numpy types to native Python types for JSON serialization
    results_json = {}
    for key, value in results.items():
        if isinstance(value, list):
            results_json[key] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in value]
        else:
            results_json[key] = value
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n✅ Metrics saved to {results_path}")
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70)


if __name__ == '__main__':
    main()
