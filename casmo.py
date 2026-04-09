"""
CASMO: Confident Adaptive Selective Momentum Optimizer

Core Innovation: AGAR (Adaptive Gradient Alignment Ratio)
    AGAR = ||E[g]||² / (||E[g]||² + Var[g])

    Measures signal (consistent gradient direction) vs noise (random fluctuations).
    Naturally ranges from 0 (pure noise) to 1 (pure signal).
"""

from typing import Tuple, Optional, Callable, Dict, Any
import torch
import numpy as np
from collections import deque


class CASMO(torch.optim.Optimizer):
    """
    Confident Adaptive Selective Momentum Optimizer.

    Extends Adam with confidence-based learning rate scaling using AGAR metrics.
    Automatically adapts to gradient signal-to-noise ratio for improved convergence.

    Args:
        params: Iterable of parameters or dicts defining parameter groups.
        lr (float): Learning rate. Default: 1e-3
        betas (Tuple[float, float]): Coefficients for computing running averages
            of gradient and its square. Default: (0.9, 0.999)
        eps (float): Term added to denominator for numerical stability. Default: 1e-8
        weight_decay (float): Decoupled weight decay coefficient. Default: 0.0
        tau_init_steps (int): Steps to collect AGAR samples for tau calibration.
            Must be >= 50. Default: 500
        tau_clip_range (Tuple[float, float]): Min/max bounds for tau. Default: (0.01, 0.5)
        c_min (float): Minimum confidence scaling factor. Must be in [0, 1]. Default: 0.1
        granularity (str): AGAR computation granularity.
            - 'parameter': Per-parameter confidence scaling.
            - 'group': Per-group confidence scaling. Default: 'group'
        agar_clamp_factor (float): Outlier clamping factor for AGAR computation.
            Set to None to disable. Default: 10.0

    Raises:
        ValueError: If any parameter is outside its valid range.
        RuntimeError: If NaN or Inf gradients are detected.
        NotImplementedError: If sparse gradients are encountered.
    """

    MIN_CALIBRATION_SAMPLES = 50
    MIN_STD_THRESHOLD = 0.01

    CV_HIGH_THRESHOLD = 0.5
    CV_MEDIUM_THRESHOLD = 0.3

    C_MIN_HIGH_VARIANCE = 0.1
    C_MIN_MEDIUM_VARIANCE = 0.3
    C_MIN_LOW_VARIANCE = 0.5

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        tau_init_steps: Optional[int] = None,
        tau_clip_range: Tuple[float, float] = (0.01, 0.5),
        c_min: float = 0.1,
        granularity: str = 'group',
        agar_clamp_factor: Optional[float] = 10.0,
        total_steps: Optional[int] = None,
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
            raise ValueError(f"Invalid c_min: {c_min}")
        if granularity not in ['parameter', 'group']:
            raise ValueError(f"Invalid granularity: {granularity} (must be 'parameter' or 'group')")

        if tau_init_steps is None:
            tau_init_steps = max(500, int(50 / (1 - betas[0])))
            if total_steps is not None:
                tau_init_steps = min(tau_init_steps, total_steps // 5)

        if tau_init_steps < 50:
            raise ValueError(f"tau_init_steps too small: {tau_init_steps} (minimum: 50)")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            tau_init_steps=tau_init_steps,
            tau_clip_range=tau_clip_range,
            c_min=c_min,
            granularity=granularity,
            agar_clamp_factor=agar_clamp_factor,
            total_steps=total_steps,
        )

        super().__init__(params, defaults)

        self._step_count = 0

        self._group_states: Dict[int, Dict[str, Any]] = {}
        for idx, group in enumerate(self.param_groups):
            group_tau_init_steps = group.get('tau_init_steps', tau_init_steps)
            self._group_states[idx] = {
                'tau': None,
                'tau_initialized': False,
                'agar_buffer': deque(maxlen=group_tau_init_steps),
                'reuse_buffer_exp_avg': None,
                'reuse_buffer_exp_avg_sq': None,
                'current_agar': None,
                'current_confidence': None,
                'agar_mean': None,
                'agar_std': None,
                'agar_median': None,
                'agar_p10': None,
                'agar_p90': None,
                'c_min': c_min,
            }

    def _validate_gradient(self, grad: torch.Tensor, group_idx: int) -> None:
        if torch.isnan(grad).any():
            raise RuntimeError(
                f"NaN gradient detected in parameter group {group_idx}. "
                "Consider using gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)"
            )
        if torch.isinf(grad).any():
            raise RuntimeError(
                f"Inf gradient detected in parameter group {group_idx}. "
                "Check for numerical overflow in loss computation or model outputs."
            )
        if grad.is_sparse:
            raise NotImplementedError(
                "CASMO does not support sparse gradients. "
                "Use torch.optim.SparseAdam for sparse gradient scenarios, "
                "or convert gradients to dense format with grad.to_dense()."
            )

    def _init_param_state(self, p: torch.Tensor) -> Dict[str, Any]:
        state = self.state[p]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        return state

    def _update_moments(self, state: Dict[str, Any], grad: torch.Tensor, beta1: float, beta2: float) -> None:
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        state['step'] += 1

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    def _apply_weight_update(self, p: torch.Tensor, state: Dict[str, Any], lr: float,
                             weight_decay: float, eps: float, confidence: torch.Tensor,
                             beta1: float, beta2: float) -> None:
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        step = state['step']

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        m_hat = exp_avg / bias_correction1
        v_hat = exp_avg_sq / bias_correction2

        if weight_decay != 0:
            p.mul_(1 - lr * weight_decay)

        denom = v_hat.sqrt().add_(eps)
        step_size = lr * confidence
        p.addcdiv_(m_hat, denom, value=-step_size)

    def _compute_agar(
        self,
        exp_avg: torch.Tensor,
        exp_avg_sq: torch.Tensor,
        eps: float,
        clamp_factor: Optional[float],
    ) -> torch.Tensor:
        if clamp_factor is not None:
            m_scale = exp_avg.abs().mean() + eps
            v_scale = exp_avg_sq.mean() + eps
            m_clamped = torch.clamp(exp_avg, min=-m_scale * clamp_factor, max=m_scale * clamp_factor)
            v_clamped = torch.clamp(exp_avg_sq, min=0.0, max=v_scale * clamp_factor)
        else:
            m_clamped = exp_avg
            v_clamped = exp_avg_sq

        signal_per_elem = m_clamped.pow(2)
        noise_per_elem = torch.clamp(v_clamped - signal_per_elem, min=eps)
        agar_per_elem = signal_per_elem / (signal_per_elem + noise_per_elem + eps)
        agar = agar_per_elem.mean()

        return torch.clamp(agar, min=0.0, max=1.0)

    def _calibrate_tau(self, agar_buffer: deque, tau_clip_range: Tuple[float, float], group_idx: int) -> float:
        if len(agar_buffer) < self.MIN_CALIBRATION_SAMPLES:
            return tau_clip_range[1]

        samples = np.array(agar_buffer)

        mu = np.mean(samples)
        sigma = np.std(samples)
        median = np.median(samples)
        p10 = np.percentile(samples, 10)
        p90 = np.percentile(samples, 90)

        group_state = self._group_states[group_idx]
        group_state['agar_mean'] = float(mu)
        group_state['agar_std'] = float(max(sigma, self.MIN_STD_THRESHOLD))
        group_state['agar_median'] = float(median)
        group_state['agar_p10'] = float(p10)
        group_state['agar_p90'] = float(p90)

        cv = sigma / (mu + 1e-8)
        if cv > self.CV_HIGH_THRESHOLD:
            c_min_adaptive = self.C_MIN_HIGH_VARIANCE
        elif cv > self.CV_MEDIUM_THRESHOLD:
            c_min_adaptive = self.C_MIN_MEDIUM_VARIANCE
        else:
            c_min_adaptive = self.C_MIN_LOW_VARIANCE

        group_state['c_min'] = float(c_min_adaptive)

        return float(np.clip(median, tau_clip_range[0], tau_clip_range[1]))

    def _compute_confidence(self, agar_value: float, group_state: Dict[str, Any], c_min: float) -> float:
        if group_state['tau_initialized']:
            mu = group_state['tau']
            sigma = group_state.get('agar_std', 0.1)
            c_min_adaptive = group_state.get('c_min', c_min)

            z_score = (agar_value - mu) / sigma
            sigmoid = 1.0 / (1.0 + np.exp(-z_score))
            confidence_value = c_min_adaptive + (1.0 - c_min_adaptive) * sigmoid

            return float(np.clip(confidence_value, c_min_adaptive, 1.0))
        else:
            return float(np.clip(agar_value, c_min, 1.0))

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group_idx, group in enumerate(self.param_groups):
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']
            c_min = group['c_min']
            tau_init_steps = group['tau_init_steps']
            tau_clip_range = group['tau_clip_range']
            granularity = group['granularity']
            agar_clamp_factor = group['agar_clamp_factor']

            group_state = self._group_states[group_idx]

            if granularity == 'group':
                valid_params = [p for p in group['params'] if p.grad is not None]
                if not valid_params:
                    continue

                all_exp_avg = []
                all_exp_avg_sq = []

                for p in group['params']:
                    if p.grad is None:
                        continue

                    self._validate_gradient(p.grad, group_idx)
                    state = self._init_param_state(p)
                    self._update_moments(state, p.grad, beta1, beta2)

                    all_exp_avg.append(state['exp_avg'].flatten())
                    all_exp_avg_sq.append(state['exp_avg_sq'].flatten())

                if all_exp_avg:
                    if group_state['reuse_buffer_exp_avg'] is None:
                        total_params = sum(m.numel() for m in all_exp_avg)
                        device = all_exp_avg[0].device
                        dtype = all_exp_avg[0].dtype
                        group_state['reuse_buffer_exp_avg'] = torch.zeros(total_params, device=device, dtype=dtype)
                        group_state['reuse_buffer_exp_avg_sq'] = torch.zeros(total_params, device=device, dtype=dtype)

                    offset = 0
                    reuse_buffer_exp_avg = group_state['reuse_buffer_exp_avg']
                    reuse_buffer_exp_avg_sq = group_state['reuse_buffer_exp_avg_sq']

                    for m, v in zip(all_exp_avg, all_exp_avg_sq):
                        numel = m.numel()
                        reuse_buffer_exp_avg[offset:offset+numel].copy_(m)
                        reuse_buffer_exp_avg_sq[offset:offset+numel].copy_(v)
                        offset += numel

                    agar = self._compute_agar(
                        reuse_buffer_exp_avg[:offset],
                        reuse_buffer_exp_avg_sq[:offset],
                        eps,
                        agar_clamp_factor,
                    )

                    agar_value = agar.item()
                    group_state['current_agar'] = agar_value

                    if not group_state['tau_initialized']:
                        group_state['agar_buffer'].append(agar_value)

                        if len(group_state['agar_buffer']) >= tau_init_steps:
                            tau = self._calibrate_tau(group_state['agar_buffer'], tau_clip_range, group_idx)
                            group_state['tau'] = tau
                            group_state['tau_initialized'] = True
                            group_state['agar_buffer'].clear()

                    confidence_value = self._compute_confidence(agar_value, group_state, c_min)
                    group_state['current_confidence'] = confidence_value

                    confidence_tensor = torch.tensor(confidence_value, device=all_exp_avg[0].device, dtype=all_exp_avg[0].dtype)
                else:
                    confidence_tensor = torch.tensor(c_min)

                for p in group['params']:
                    if p.grad is None:
                        continue

                    self._apply_weight_update(p, self.state[p], lr, weight_decay,
                                              eps, confidence_tensor, beta1, beta2)

            else:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    self._validate_gradient(p.grad, group_idx)
                    state = self._init_param_state(p)
                    self._update_moments(state, p.grad, beta1, beta2)

                    agar = self._compute_agar(state['exp_avg'], state['exp_avg_sq'], eps, agar_clamp_factor)
                    agar_value = agar.item()

                    if group_state['current_agar'] is None:
                        group_state['current_agar'] = agar_value

                    if not group_state['tau_initialized']:
                        group_state['agar_buffer'].append(agar_value)

                        if len(group_state['agar_buffer']) >= tau_init_steps:
                            tau = self._calibrate_tau(group_state['agar_buffer'], tau_clip_range, group_idx)
                            group_state['tau'] = tau
                            group_state['tau_initialized'] = True
                            group_state['agar_buffer'].clear()

                    confidence_value = self._compute_confidence(agar_value, group_state, c_min)
                    confidence_tensor = torch.tensor(confidence_value, device=p.device, dtype=p.dtype)

                    if group_state['current_confidence'] is None:
                        group_state['current_confidence'] = confidence_value

                    self._apply_weight_update(p, state, lr, weight_decay,
                                              eps, confidence_tensor, beta1, beta2)

        return loss

    def state_dict(self):
        state_dict = super().state_dict()

        serializable_group_states = {}
        for idx, gs in self._group_states.items():
            serializable_group_states[idx] = {
                'tau': gs['tau'],
                'tau_initialized': gs['tau_initialized'],
                'agar_buffer': list(gs['agar_buffer']),
                'agar_buffer_maxlen': gs['agar_buffer'].maxlen,
                'agar_mean': gs.get('agar_mean'),
                'agar_std': gs.get('agar_std'),
                'agar_median': gs.get('agar_median'),
                'agar_p10': gs.get('agar_p10'),
                'agar_p90': gs.get('agar_p90'),
                'c_min': gs.get('c_min'),
            }

        state_dict['_group_states'] = serializable_group_states
        state_dict['_step_count'] = self._step_count

        return state_dict

    def load_state_dict(self, state_dict):
        if '_group_states' in state_dict:
            loaded_states = state_dict.pop('_group_states')
            self._group_states = {}
            for idx, gs in loaded_states.items():
                idx_int = int(idx) if isinstance(idx, str) else idx

                maxlen = gs.pop('agar_buffer_maxlen', None)
                buffer_list = gs.pop('agar_buffer', [])
                gs['agar_buffer'] = deque(buffer_list, maxlen=maxlen)

                gs.setdefault('reuse_buffer_exp_avg', None)
                gs.setdefault('reuse_buffer_exp_avg_sq', None)
                gs.setdefault('current_agar', None)
                gs.setdefault('current_confidence', None)
                gs.setdefault('agar_mean', None)
                gs.setdefault('agar_std', None)
                gs.setdefault('agar_median', None)
                gs.setdefault('agar_p10', None)
                gs.setdefault('agar_p90', None)
                gs.setdefault('c_min', self.param_groups[idx_int].get('c_min', 0.1))

                self._group_states[idx_int] = gs

        if '_step_count' in state_dict:
            self._step_count = state_dict.pop('_step_count')

        super().load_state_dict(state_dict)