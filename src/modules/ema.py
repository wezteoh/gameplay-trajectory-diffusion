"""Exponential moving average of model weights (LitEma-style)."""

from __future__ import annotations

import torch
from torch import nn


class LitEma(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        use_num_upates: bool = True,
    ):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.m_name2s_name: dict[str, str] = {}
        self.register_buffer("decay", torch.tensor(decay, dtype=torch.float32))
        self.register_buffer(
            "num_updates",
            (
                torch.tensor(0, dtype=torch.int)
                if use_num_upates
                else torch.tensor(-1, dtype=torch.int)
            ),
        )

        for name, p in model.named_parameters():
            if p.requires_grad:
                s_name = name.replace(".", "")
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params: list[torch.Tensor] = []

    def forward(self, model: nn.Module) -> None:
        if int(self.num_updates.item()) >= 0:
            self.num_updates += 1
            nu = float(self.num_updates.item())
            decay_val = min(float(self.decay.item()), (1.0 + nu) / (10.0 + nu))
        else:
            decay_val = float(self.decay.item())

        one_minus_decay = 1.0 - decay_val

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    cur = shadow_params[sname]
                    shadow_params[sname] = cur.type_as(m_param[key])
                    diff = shadow_params[sname] - m_param[key]
                    shadow_params[sname].sub_(one_minus_decay * diff)
                else:
                    assert key not in self.m_name2s_name

    def copy_to(self, model: nn.Module) -> None:
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                sname = self.m_name2s_name[key]
                m_param[key].data.copy_(shadow_params[sname].data)
            else:
                assert key not in self.m_name2s_name

    def store(
        self,
        parameters: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> None:
        self.collected_params = [param.clone() for param in parameters]

    def restore(
        self, parameters: list[torch.Tensor] | tuple[torch.Tensor, ...]
    ) -> None:
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
