from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MoEStats:
    importance: torch.Tensor
    load: torch.Tensor
    extras: dict[str, torch.Tensor] | None = None

    @classmethod
    def zeros(cls, num_experts: int, device: torch.device) -> MoEStats:
        return cls(
            importance=torch.zeros(num_experts, device=device, dtype=torch.float32),
            load=torch.zeros(num_experts, device=device, dtype=torch.float32),
        )

    def add_values_(
        self, importance: torch.Tensor, load: torch.Tensor, extra: dict[str, torch.Tensor] | None = None
    ):
        self.importance += importance
        self.load += load
        if extra:
            if self.extras is None:
                self.extras = {}
            for k, v in extra.items():
                if k in self.extras:
                    self.extras[k] += v
                else:
                    self.extras[k] = v.clone() if v.is_leaf else v
        return self
