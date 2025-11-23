"""
Helper script to run all optimizer configurations sequentially.
Simplifies running the full benchmark battery.
"""

import subprocess
import sys
import os
import argparse
import time

def run_experiment(config, batch_size, epochs, dataset_size, quick_test=False):
    """Run a single experiment configuration."""
    cmd = [
        sys.executable, 'train.py',
        '--config', config,
        '--batch_size', str(batch_size),
        '--epochs', str(epochs),
        '--dataset_size', str(dataset_size)
    ]
    
    if quick_test:
        cmd.append('--quick_test')
    
    print(f"\n{'='*70}")
    print(f"Starting: {config} (batch_size={batch_size})")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ Completed {config} in {elapsed/60:.1f} minutes")
        return True
    else:
        print(f"\n❌ Failed {config}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run all GAN benchmark experiments')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[16],
                       help='Batch sizes to test (default: 16)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs per experiment')
    parser.add_argument('--dataset_size', type=int, default=20000,
                       help='Tiny ImageNet dataset size')
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test mode (100 samples)')
    parser.add_argument('--configs', type=str, nargs='+',
                       default=['adam_adam', 'casmo_adam', 'adam_casmo', 'casmo_casmo'],
                       help='Configs to run')
    args = parser.parse_args()
    
    print("="*70)
    print("B7 GAN Benchmark - Running All Experiments")
    print("="*70)
    print(f"Configs: {args.configs}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Epochs: {args.epochs}")
    print(f"Dataset size: {args.dataset_size}")
    print("="*70)
    
    total_experiments = len(args.configs) * len(args.batch_sizes)
    print(f"\nTotal experiments: {total_experiments}")
    
    start_time = time.time()
    results = []
    
    for config in args.configs:
        for batch_size in args.batch_sizes:
            success = run_experiment(
                config, batch_size, args.epochs, args.dataset_size, args.quick_test
            )
            results.append((config, batch_size, success))
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\nResults:")
    for config, batch_size, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {config} (bs={batch_size}): {status}")
    
    # Create comparison plots if all experiments succeeded
    if all(success for _, _, success in results):
        print("\nGenerating comparison plots...")
        plot_script = os.path.join(os.path.dirname(__file__), 'plot_comparison.py')
        if os.path.exists(plot_script):
            subprocess.run([sys.executable, plot_script])
        else:
            print("⚠️  plot_comparison.py not found")


if __name__ == '__main__':
    main()
