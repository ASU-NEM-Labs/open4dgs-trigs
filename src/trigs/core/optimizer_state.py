"""Shared optimizer-state row operations for primitive remapping.

These helpers centralize the two operations used by both trainer-level pruning
and strategy-level relocation:

1) Remap a parameter object while preserving/slicing row-wise optimizer state.
2) Zero row-wise optimizer state for selected primitive indices.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


OwnerEntry = Tuple[torch.optim.Optimizer, Dict[str, Any], int]


def build_param_owners(
    optimizers: Dict[str, torch.optim.Optimizer],
) -> Dict[nn.Parameter, List[OwnerEntry]]:
    """Map each parameter object to all optimizer/group/index owners."""
    owners: Dict[nn.Parameter, List[OwnerEntry]] = {}
    for opt in optimizers.values():
        for group in opt.param_groups:
            for idx, param in enumerate(group["params"]):
                owners.setdefault(param, []).append((opt, group, idx))
    return owners


def remap_param_with_sliced_state(
    owners: Dict[nn.Parameter, List[OwnerEntry]],
    old_param: nn.Parameter,
    new_param: nn.Parameter,
    keep_idx: torch.Tensor,
    n_old: int,
) -> None:
    """Swap optimizer ownership from old->new param and slice row states."""
    for opt, group, idx in owners.get(old_param, []):
        old_state = opt.state.pop(old_param, None)
        if old_state is not None:
            new_state: Dict[str, Any] = {}
            for key, value in old_state.items():
                if (
                    torch.is_tensor(value)
                    and value.ndim >= 1
                    and int(value.shape[0]) == n_old
                ):
                    new_state[key] = value[keep_idx].contiguous()
                else:
                    new_state[key] = value
            opt.state[new_param] = new_state
        group["params"][idx] = new_param


def export_param_state(
    optimizers: Dict[str, torch.optim.Optimizer],
    params: Dict[str, nn.Parameter],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Adam moments per parameter name, copied to the CPU."""
    name_of = {id(p): name for name, p in params.items()}
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for opt in optimizers.values():
        for group in opt.param_groups:
            for p in group["params"]:
                name = name_of.get(id(p))
                state = opt.state.get(p)
                if name is None or not state or "exp_avg" not in state:
                    continue
                out[name] = {
                    key: state[key].detach().to("cpu", copy=True)
                    for key in ("exp_avg", "exp_avg_sq")
                }
    return out


def load_param_state(
    optimizers: Dict[str, torch.optim.Optimizer],
    params: Dict[str, nn.Parameter],
    saved: Dict[str, Dict[str, torch.Tensor]],
    step: int,
) -> None:
    """Install exported Adam moments on matching parameters.

    ``step`` is the number of updates the moments have already absorbed; it
    sets Adam's bias correction (near 1 once a run is under way), so restored
    moments are used as-is instead of being re-amplified like a fresh start.
    """
    if not saved:
        return
    name_of = {id(p): name for name, p in params.items()}
    for opt in optimizers.values():
        for group in opt.param_groups:
            for p in group["params"]:
                entry = saved.get(name_of.get(id(p), ""))
                if entry is None or tuple(entry["exp_avg"].shape) != tuple(p.shape):
                    continue
                opt.state[p] = {
                    "step": torch.tensor(float(step)),
                    "exp_avg": entry["exp_avg"].to(p.device, p.dtype).clone(),
                    "exp_avg_sq": entry["exp_avg_sq"].to(p.device, p.dtype).clone(),
                }


def zero_param_state_rows(
    owners: Dict[nn.Parameter, List[OwnerEntry]],
    param: nn.Parameter,
    row_idx: torch.Tensor,
    n_rows: int,
) -> None:
    """Zero row-wise optimizer state tensors for selected primitive rows."""
    for opt, _, _ in owners.get(param, []):
        state = opt.state.get(param)
        if state is None:
            continue
        for value in state.values():
            if (
                torch.is_tensor(value)
                and value.ndim >= 1
                and int(value.shape[0]) == n_rows
            ):
                value[row_idx] = 0.0
