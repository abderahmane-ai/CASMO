# Changelog

All notable changes to CASMO will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-01-09

### Added
- **Documentation**: Comprehensive documentation suite in `docs/` directory
  - Getting Started guide with installation and quick start
  - Complete API Reference with all parameters documented
  - FAQ with 30+ common questions and answers
  - Migration Guide for moving from Adam/AdamW to CASMO
- **Examples**: 5 working examples in `examples/` directory
  - Basic usage example (minimal working code)
  - Image classification with CIFAR-10 and ResNet
  - LLM fine-tuning example with Transformers
  - Optimizer comparison with visualization
- **GitHub Infrastructure**: CI/CD and contribution templates
  - Automated testing workflow (multi-OS, multi-Python)
  - Code quality checks (Black, flake8, mypy)
  - Automated PyPI publishing workflow
  - Issue templates (bug report, feature request)
  - Pull request template with checklist
- **Contributing Guide**: Complete CONTRIBUTING.md with development guidelines
- **Benchmark Overview**: Comprehensive benchmarks/README.md documenting all 5 benchmarks

### Changed
- **README**: Updated benchmark table to match actual benchmarks (b1-b5)
- **Benchmarks**: Removed hardcoded 5% tau_init_steps calculations, now uses automatic calculation via `total_steps` parameter
- **Benchmarks**: Fixed all benchmarks to use current CASMO interface (removed obsolete `log_level` and `tau_dead_zone` parameters)
- **Benchmarks**: Renumbered benchmarks sequentially (b1-b5) after removing incomplete ones

### Removed
- **Benchmarks**: Removed incomplete benchmarks (b1_noisy_cifar10, b5_gan_stability) that had no reports
- **Cache**: Cleaned up all `__pycache__` directories

## [0.2.0] - 2025-11-28

### Fixed
- **Critical: DDEA Integration Bug** - Fixed a critical bug where the DDEAdapter's adaptive `tau` threshold was not being used in the confidence calculation. The optimizer was updating `tau` internally via `DDEAdapter.update()`, but the confidence mapping was incorrectly using the static `agar_mean` from calibration instead of the dynamically adapted `tau`. This rendered the DDEA's noise-robust adaptation mechanism non-functional.
  - **Impact**: After the initial calibration phase, CASMO was effectively using a fixed threshold rather than adapting to changing gradient dynamics.
  - **Fix**: Updated confidence mapping in both group-level and parameter-level granularity modes to correctly use `group_state['tau_adapter'].tau` as the center (`mu`) of the sigmoid function.
  - **Verification**: Added reproduction test (`tests/reproduce_ddea_fix.py`) and validated the fix with the B2 Grokking benchmark, which showed CASMO achieving 90% validation accuracy vs AdamW's 24% on 30% label noise.

- **Debug Logging** - Fixed inconsistency in log level 3 (debug) output where the logged `μ` value was incorrectly fetching the static calibrated mean instead of the current adaptive `tau`. Logs now accurately reflect the optimizer's internal state.

### Changed
- **Benchmark B4 (DP-SGD)** - Standardized hyperparameters across all optimizers (CASMO, DP-AdamW, DP-SGD) to use the same learning rate for fair comparison. Previously, SGD was hardcoded to use `lr=0.1` while others used `lr=1e-3`.

### Verified
- All existing tests pass (100% pass rate on pytest suite)
- `verify_installation.py` passes with no regressions
- B2 Grokking benchmark validates DDEA fix with clear 66% generalization improvement over AdamW

## [0.1.0] - 2025-11-22

### Added
- Initial release of CASMO optimizer
- AGAR (Adaptive Gradient Alignment Ratio) metric for gradient quality assessment
- DDEAdapter for dynamic tau threshold adjustment
- Universal sigmoid-based confidence mapping
- Group-level and parameter-level granularity modes
- Comprehensive test suite
- Benchmark suite (B1-B7) demonstrating CASMO's capabilities
- Production-ready PyTorch optimizer implementation
