"""
Benchmark B8: Multi-Asset Portfolio Optimization (DeepDow)
Benchmark ID: B8

Tests CASMO's ability to act as a "Risk-Aware Portfolio Manager" in a 
high-volatility environment containing Cryptocurrencies.

Task:
    Dynamic Asset Allocation (Stock, Bond, Gold, Real Estate, Crypto).
    Objective: Maximize Sharpe Ratio (Risk-Adjusted Return) Net of Costs.

Domain Specifics:
    - Transaction Costs: 10bps (0.10%) per turnover.
    - Risk-Free Rate: 2.0% Annualized hurdle.
    - Constraint: Long-only, Fully invested (Weights sum to 1.0).

Hypothesis:
    - AdamW will "churn" the portfolio (High Turnover) chasing noise.
    - CASMO will maintain stable allocations (Low Turnover) via AGAR filtering. 
"""

import sys
import os
import argparse
import logging
import random
import time

# Add parent directory to path to import casmo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from casmo import CASMO

# -----------------------------------------------------------------------------
# 1. Dataset: Multi-Asset Universe
# -----------------------------------------------------------------------------

class FinancialPortfolioDataset(Dataset):
    """
    Real-world financial data for portfolio optimization.
    Assets: SPY, TLT, GLD, VNQ, BTC-USD
    """
    def __init__(self, tickers=None, seq_len=60, split='train', split_date="2020-01-01"):
        if tickers is None:
            tickers = ['SPY', 'TLT', 'GLD', 'VNQ', 'BTC-USD']
        
        self.tickers = tickers
        self.seq_len = seq_len
        self.split = split
        
        # 1. Download Data
        print(f"[{split.upper()}] Loading data for: {', '.join(tickers)}")
        # Download generous range
        data = yf.download(tickers, start="2015-01-01", end="2024-01-01", progress=False)
        
        # Handle yfinance API structure
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            prices = data
            
        prices = prices.dropna()
        self.dates = prices.index
        
        # 2. Compute Log Returns
        # log(P_t / P_{t-1})
        self.returns = np.log(prices / prices.shift(1)).dropna()
        
        # 3. Train/Test Split
        if split == 'train':
            mask = self.returns.index < split_date
        else:
            mask = self.returns.index >= split_date
            
        self.data_slice = self.returns.loc[mask]
        self.dates_slice = self.returns.index[mask]
        
        # 4. Prepare Sequences
        # X: Past [seq_len] returns
        # y: Next [1] return (the target to optimize)
        self.X = []
        self.y_next = []
        self.sample_dates = []
        
        values = self.data_slice.values
        
        for i in range(len(values) - seq_len):
            self.X.append(values[i : i+seq_len])
            self.y_next.append(values[i+seq_len])
            self.sample_dates.append(self.dates_slice[i+seq_len])
            
        self.X = torch.FloatTensor(np.array(self.X))
        self.y_next = torch.FloatTensor(np.array(self.y_next))
        
        print(f"[{split.upper()}] Samples: {len(self.X)} | Risk Regime: {{'Crisis (2020-2023)' if split=='test' else 'Bull (2015-2019)'}}")

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_next[idx]


# -----------------------------------------------------------------------------
# 2. Model: LSTM Allocator
# -----------------------------------------------------------------------------

class PortfolioAllocator(nn.Module):
    """
    Policy Network: Maps past market state -> Portfolio Weights
    """
    def __init__(self, num_assets, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_assets, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, num_assets),
            nn.Softmax(dim=-1) # Enforces Sum(weights) = 1.0 (Long Only)
        )
        
    def forward(self, x):
        # x: [batch, seq, assets]
        out, _ = self.lstm(x)
        # Decision based on most recent hidden state
        weights = self.head(out[:, -1, :])
        return weights


# -----------------------------------------------------------------------------
# 3. Utilities & Loss
# -----------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sharpe_loss(weights, next_returns, risk_free_annual=0.02):
    """
    Differentiable Negative Sharpe Ratio.
    """
    # Daily Risk Free Rate approx
    rf_daily = risk_free_annual / 252.0
    
    # Portfolio Return: w * r
    port_ret = torch.sum(weights * next_returns, dim=1)
    
    # Excess Return
    excess_ret = port_ret - rf_daily
    
    # Statistics
    mean = torch.mean(excess_ret)
    std = torch.std(excess_ret) + 1e-6
    
    # Annualized Sharpe
    sharpe = (mean / std) * np.sqrt(252)
    
    return -sharpe


# -----------------------------------------------------------------------------
# 4. Training Loop
# -----------------------------------------------------------------------------

def run_benchmark(optimizer_name, device, train_loader, test_loader, num_assets, epochs=50):
    
    print(f"\n{ '='*60}")
    print(f"Running Strategy: {optimizer_name.upper()}")
    print(f"{ '='*60}")
    
    model = PortfolioAllocator(num_assets).to(device)
    
    # CASMO Configuration
    if optimizer_name == "CASMO":
        # 5% of Total Steps for Calibration (User Request)
        total_steps = len(train_loader) * epochs
        tau_init_steps = int(0.05 * total_steps)
        tau_init_steps = max(50, tau_init_steps) # Safety floor
        
        print(f"CASMO Configuration: Total Steps={total_steps}, Tau Init={tau_init_steps} (5%)")
        
        optimizer = CASMO(
            model.parameters(), 
            lr=1e-3, 
            weight_decay=0.0, # No L2, we want pure Sharpe optimization
            granularity='group',
            tau_init_steps=tau_init_steps,
            c_min=0.1
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # --- Training Phase ---
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for X, y_next in train_loader:
            X, y_next = X.to(device), y_next.to(device)
            
            optimizer.zero_grad()
            weights = model(X)
            loss = sharpe_loss(weights, y_next)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Sharpe Loss: {epoch_loss/len(train_loader):.4f}")
            
    train_time = time.time() - start_time
    print(f"Training finished in {train_time:.1f}s")
    
    # --- Backtesting Phase (Walk-Forward) ---
    print("\nRunning Walk-Forward Backtest (2020-2023)...")
    model.eval()
    
    portfolio_val = [1.0] # Equity Curve
    weights_history = []
    
    # Transaction Cost (10bps)
    TX_COST = 0.0010 
    prev_weights = torch.ones(num_assets).to(device) / num_assets
    
    accumulated_turnover = 0.0
    
    with torch.no_grad():
        # Sequential processing is mandatory for Backtesting
        for X, y_next in test_loader:
            X, y_next = X.to(device), y_next.to(device)
            
            # 1. Decide Weights
            curr_weights = model(X) 
            
            # 2. Calculate Turnover & Cost
            turnover = torch.sum(torch.abs(curr_weights - prev_weights)).item()
            cost = turnover * TX_COST
            accumulated_turnover += turnover
            
            # 3. Calculate Return
            # Return = (Weights * Asset_Returns) - Cost
            raw_ret = torch.sum(curr_weights * y_next).item()
            net_ret = raw_ret - cost
            
            # 4. Update Equity
            new_val = portfolio_val[-1] * (1 + net_ret)
            portfolio_val.append(new_val)
            
            # Store State
            weights_history.append(curr_weights.cpu().numpy().flatten())
            prev_weights = curr_weights
            
    return {
        'equity': np.array(portfolio_val),
        'weights': np.array(weights_history),
        'turnover': accumulated_turnover / len(test_loader) # Avg daily turnover
    }


# -----------------------------------------------------------------------------
# 5. Analysis & Plotting
# -----------------------------------------------------------------------------

def calculate_metrics(equity_curve, turnover_avg, name):
    """Compute financial KPIs."""
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    # CAGR
    total_ret = equity_curve[-1] - 1
    years = len(equity_curve) / 252.0
    cagr = (equity_curve[-1])**(1/years) - 1
    
    # Sharpe
    mean = np.mean(returns)
    std = np.std(returns) + 1e-9
    sharpe = (mean / std) * np.sqrt(252)
    
    # Sortino
    downside = returns[returns < 0]
    down_std = np.std(downside) + 1e-9
    sortino = (mean / down_std) * np.sqrt(252)
    
    # Max DD
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / peak
    max_dd = np.max(dd)
    
    return {
        'Name': name,
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'MaxDD': max_dd,
        'Turnover': turnover_avg,
        'TotalRet': total_ret
    }

def main():
    parser = argparse.ArgumentParser(description='B8 Portfolio Optimization')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Setup Data
    train_ds = FinancialPortfolioDataset(split='train')
    test_ds = FinancialPortfolioDataset(split='test')
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False) # Seq size 1 for backtest
    
    num_assets = len(train_ds.tickers)
    
    # 2. Run Benchmarks
    res_casmo = run_benchmark("CASMO", device, train_loader, test_loader, num_assets, args.epochs)
    res_adam = run_benchmark("AdamW", device, train_loader, test_loader, num_assets, args.epochs)
    
    # 3. Simulate 1/N Benchmark
    print("\nSimulating 1/N Benchmark...")
    eq_1n = [1.0]
    w_1n = np.ones(num_assets) / num_assets
    for _, y in test_loader:
        ret = np.sum(w_1n * y.numpy())
        eq_1n.append(eq_1n[-1] * (1 + ret))
    res_1n = {'equity': np.array(eq_1n), 'turnover': 0.0}
    
    # 4. Generate Report
    metrics = []
    metrics.append(calculate_metrics(res_1n['equity'], 0.0, "1/N Bench"))
    metrics.append(calculate_metrics(res_adam['equity'], res_adam['turnover'], "AdamW"))
    metrics.append(calculate_metrics(res_casmo['equity'], res_casmo['turnover'], "CASMO"))
    
    print("\n" + "="*95)
    print(f"{ 'Strategy':<15} | {'CAGR':<8} | {'Sharpe':<6} | {'Sortino':<7} | {'MaxDD':<7} | {'Turnover':<8}")
    print("-" * 95)
    for m in metrics:
        print(f"{m['Name']:<15} | {m['CAGR']*100:>7.1f}% | {m['Sharpe']:>6.2f} | {m['Sortino']:>7.2f} | {m['MaxDD']*100:>6.1f}% | {m['Turnover']*100:>7.2f}%")
    print("="*95)
    
    # 5. Plotting
    print("\nPlotting results...")
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))
    dates = test_ds.sample_dates
    
    # Equity Curve
    ax1 = axes[0]
    ax1.plot(dates, res_1n['equity'][1:], label='1/N (Equal Weight)', color='gray', linestyle=':')
    ax1.plot(dates, res_adam['equity'][1:], label='AdamW', color='orange')
    ax1.plot(dates, res_casmo['equity'][1:], label='CASMO', color='green', linewidth=2)
    ax1.set_title(f"Cumulative Wealth (2020-2023) [Initial: $1.0]")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Weights - CASMO
    ax2 = axes[1]
    ax2.stackplot(dates, res_casmo['weights'].T, labels=train_ds.tickers, alpha=0.8)
    ax2.set_title("CASMO: Asset Allocation")
    ax2.set_ylabel("Weight %")
    ax2.legend(loc='upper left')
    ax2.margins(0, 0)
    
    # Weights - AdamW
    ax3 = axes[2]
    ax3.stackplot(dates, res_adam['weights'].T, labels=train_ds.tickers, alpha=0.8)
    ax3.set_title("AdamW: Asset Allocation")
    ax3.set_ylabel("Weight %")
    ax3.margins(0, 0)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'portfolio_benchmark.png')
    plt.savefig(save_path)
    print(f"✅ Results saved to {save_path}")

if __name__ == "__main__":
    main()

