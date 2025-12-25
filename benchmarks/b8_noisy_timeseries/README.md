# B8: Financial Regime Change (Bitcoin)

## Overview
This benchmark evaluates CASMO's ability to adapt to **Real-World Concept Drift** using Bitcoin (BTC-USD) market data.

## The Challenge
Financial markets are non-stationary. Strategies that work in a "Bull Market" often fail catastrophically in a "Bear Market".
*   **Regime Change:** The transition from the 2020-2021 Crypto Bubble to the 2022 "Crypto Winter" represents a massive distributional shift in volatility and trend.
*   **High Noise:** Daily returns are dominated by stochastic noise, making gradient signals unreliable.

## Experiment Setup
*   **Dataset:** BTC-USD Daily Log Returns (via `yfinance`).
    *   **Training Period:** 2017-01-01 to 2021-12-31 (High Volatility, mostly upward/mixed).
    *   **Testing Period:** 2022-01-01 to 2023-01-01 (The 2022 Crash - Distinct "Bear" Regime).
*   **Task:** Predict next day's return given the past 30 days.
*   **Model:** LSTM (Hidden=64).
*   **Comparison:** CASMO vs. AdamW.

## Hypothesis
CASMO's **AGAR (Adaptive Gradient Alignment Ratio)** will identify the low signal-to-noise ratio inherent in financial returns.
*   **AdamW** is expected to overfit to the noise of the training set, leading to poor generalization during the regime shift (2022).
*   **CASMO** is expected to dampen learning when gradients are noisy (low AGAR), resulting in a more robust model that adapts better to the unseen market regime.