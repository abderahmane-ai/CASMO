"""
CASMO: Confidence-Adjusted Signal-to-noise Momentum Optimizer.

Core idea (AGAR — Adaptive Gradient Alignment Ratio)
----------------------------------------------------
For every parameter coordinate CASMO estimates a *signal fraction*

    AGAR = signal / (signal + noise) = E[g]^2 / (E[g]^2 + Var[g])   in [0, 1]

where the signal is the squared first moment and the noise is the (belief-style)
variance of the gradient. AGAR is 1 for a perfectly consistent gradient and 0 for
pure noise. CASMO turns this per-coordinate signal fraction into a confidence that
scales an AdamW update along two complementary axes:

  * trust   (absolute)  -- slows the whole tensor when its gradients are noise
                           dominated. This is what buys robustness to label noise.
  * focus   (relative)  -- reweights coordinates toward the reliable directions
                           without changing the overall pace. This preserves clean
                           convergence speed and adds expressivity.

A single ``robustness`` dial in ``[0, 1]`` interpolates between "behave like AdamW"
(``0``) and "maximally noise robust" (``1``). The default of ``0.5`` improves
generalization under label noise while matching AdamW's clean-data test accuracy.

The estimator needs only two EMAs per parameter (the first moment ``m`` and the
belief variance ``s``) — the same optimizer-state memory as Adam — because the
Adam denominator ``sqrt(E[g^2])`` is reconstructed from ``m^2 + s``.
"""

from typing import Tuple, Optional, Callable, Dict, Any
import warnings
import torch

__all__ = ["CASMO"]

# Constructor keywords from the pre-0.4 calibration-based design. They no longer
# have any effect and are accepted only so old call sites keep running.
_DEPRECATED_KWARGS = (
    "tau_init_steps",
    "tau_clip_range",
    "granularity",
    "agar_clamp_factor",
    "total_steps",
)


class CASMO(torch.optim.Optimizer):
    """Confidence-Adjusted Signal-to-noise Momentum Optimizer.

    A drop-in replacement for :class:`torch.optim.AdamW` that scales each update
    by a per-coordinate confidence derived from the gradient signal-to-noise ratio.

    Args:
        params: Iterable of parameters or parameter-group dicts.
        lr (float): Learning rate. Default: ``1e-3``.
        betas (Tuple[float, float]): EMA coefficients for the first moment and the
            belief variance. Default: ``(0.9, 0.999)``.
        eps (float): Numerical-stability term. Default: ``1e-8``.
        weight_decay (float): Decoupled (AdamW-style) weight decay. Default: ``0.0``.
        c_min (float): Floor of the absolute ``trust`` factor, in ``[0, 1]``. Keeps
            the effective learning rate from collapsing under sustained noise.
            Default: ``0.1``.
        robustness (float): Strength of the absolute noise-suppression axis, ``>= 0``
            (``0`` disables it → AdamW-like speed; ``1`` is maximally robust).
            Default: ``0.5``.
        rel_floor (float): Floor of the relative ``focus`` factor, in ``[0, 1]``.
            Bounds how strongly a single coordinate can be down-weighted. Default: ``0.1``.
        nan_guard (bool): If ``True``, raise on non-finite gradients instead of
            stepping. Off by default (it forces a host sync every step). Default: ``False``.

    Raises:
        ValueError: If a hyper-parameter is outside its valid range.
        NotImplementedError: If a sparse gradient is encountered.
        RuntimeError: If ``nan_guard`` is set and a NaN/Inf gradient is seen.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        c_min: float = 0.1,
        robustness: float = 0.5,
        rel_floor: float = 0.1,
        nan_guard: bool = False,
        **deprecated: Any,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= c_min <= 1.0:
            raise ValueError(f"Invalid c_min: {c_min} (must be in [0, 1])")
        if not 0.0 <= robustness:
            raise ValueError(f"Invalid robustness: {robustness} (must be >= 0)")
        if not 0.0 <= rel_floor <= 1.0:
            raise ValueError(f"Invalid rel_floor: {rel_floor} (must be in [0, 1])")

        for key in deprecated:
            if key in _DEPRECATED_KWARGS:
                warnings.warn(
                    f"CASMO: '{key}' is deprecated and ignored since v0.4.0 — CASMO no "
                    "longer uses a calibration phase. See the migration guide.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                raise TypeError(f"CASMO got an unexpected keyword argument '{key}'")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            c_min=c_min,
            robustness=robustness,
            rel_floor=rel_floor,
            nan_guard=nan_guard,
        )
        super().__init__(params, defaults)

        # Transient per-group monitoring, recomputed every step. Not part of the
        # checkpoint (the parent Optimizer serializes the m/s/step state).
        self._group_states: Dict[int, Dict[str, Any]] = {
            idx: {"current_agar": None, "current_confidence": None}
            for idx in range(len(self.param_groups))
        }

    def _check_grad(self, grad: torch.Tensor, group_idx: int, nan_guard: bool) -> None:
        if grad.is_sparse:
            raise NotImplementedError(
                "CASMO does not support sparse gradients. Use torch.optim.SparseAdam "
                "for sparse scenarios, or densify with grad.to_dense()."
            )
        if nan_guard and not torch.isfinite(grad).all():
            raise RuntimeError(
                f"Non-finite (NaN/Inf) gradient detected in parameter group {group_idx}. "
                "Clip gradients (torch.nn.utils.clip_grad_norm_) or check the loss."
            )

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            c_min = group["c_min"]
            robustness = group["robustness"]
            rel_floor = group["rel_floor"]
            nan_guard = group["nan_guard"]

            agar_sum: Optional[torch.Tensor] = None
            conf_sum: Optional[torch.Tensor] = None
            count = 0

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                self._check_grad(grad, group_idx, nan_guard)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["s"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m, s = state["m"], state["s"]
                state["step"] += 1
                step = state["step"]

                # First moment (signal) and belief variance (noise).
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                diff = grad - m
                s.mul_(beta2).addcmul_(diff, diff, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                m_hat = m / bias_correction1
                s_hat = s / bias_correction2

                signal = m_hat * m_hat
                total = signal + s_hat  # ~= E[g^2]
                agar = signal / (total + eps)  # per-coord signal fraction

                agar_mean = agar.mean()
                trust = c_min + (1 - c_min) * agar_mean  # absolute pace (scalar)
                focus = (agar / (agar_mean + eps)).clamp_(rel_floor, 1.0)  # relative
                confidence = trust.pow(robustness) * focus

                denom = total.sqrt().add_(eps)  # reconstructed Adam denom
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                p.addcdiv_(m_hat * confidence, denom, value=-lr)

                n = agar.numel()
                agar_contrib = agar.sum()
                conf_contrib = confidence.sum()
                agar_sum = agar_contrib if agar_sum is None else agar_sum + agar_contrib
                conf_sum = conf_contrib if conf_sum is None else conf_sum + conf_contrib
                count += n

            gs = self._group_states.setdefault(
                group_idx, {"current_agar": None, "current_confidence": None}
            )
            if count > 0:
                gs["current_agar"] = (agar_sum / count).item()
                gs["current_confidence"] = (conf_sum / count).item()

        return loss

    def group_metrics(self, group_idx: int = 0) -> Dict[str, Optional[float]]:
        """Return the most recent mean AGAR and confidence for a parameter group.

        Useful for logging the optimizer's live view of gradient signal quality.
        Values are ``None`` before the first :meth:`step`.
        """
        gs = self._group_states.get(group_idx, {})
        return {
            "agar": gs.get("current_agar"),
            "confidence": gs.get("current_confidence"),
        }
