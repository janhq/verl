# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Muon optimizer for 2D parameters in neural networks.

Muon (MomentUm Orthogonalized by Newton-Schulz) is designed for matrix parameters
in neural networks. It should be combined with AdamW for non-matrix parameters
(embeddings, biases, gains, etc.).

Reference: https://github.com/KellerJordan/Muon
    https://kellerjordan.github.io/posts/muon/
"""

from typing import Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-8) -> Tensor:
    """Compute the zeroth power of G via Newton-Schulz iteration.

    This orthogonalizes the matrix G, making it suitable for weight updates.
    The iteration converges to G / ||G||_2 when steps -> infinity.

    Args:
        G: Input tensor of shape [..., m, n] where m, n >= 2
        steps: Number of Newton-Schulz iterations (default: 5)
        eps: Small epsilon for numerical stability

    Returns:
        Orthogonalized tensor of same shape as G
    """
    assert G.ndim >= 2, f"G must have at least 2 dimensions, got {G.ndim}"
    *batch, m, n = G.shape

    # Handle special cases
    if m < 2 or n < 2:
        return G.float()

    G = G.bfloat16()

    # Normalize to prevent overflow
    G_norm = G.norm(dim=(-2, -1), keepdim=True)
    G = G / (G_norm + eps)

    # Ensure m >= n for the iteration
    need_transpose = m < n
    if need_transpose:
        G = G.transpose(-2, -1)
        m, n = n, m

    # Newton-Schulz iteration
    # Coefficients for 5-step iteration
    for _ in range(steps):
        G = G @ G.transpose(-2, -1)
        G = 0.5 * G + 0.5 * G @ G.transpose(-2, -1) @ G

    # Transpose back if we transposed
    if need_transpose:
        G = G.transpose(-2, -1)

    return G.float()


class Muon(Optimizer):
    """Muon optimizer for 2D matrix parameters.

    Muon applies orthogonalized SGD-momentum updates to weight matrices.
    It should be used alongside AdamW for non-matrix parameters.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 0.02)
        momentum: Momentum factor (default: 0.95)
        nesterov: Use Nesterov momentum (default: True)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        weight_decay: Weight decay coefficient (default: 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Optional loss from closure
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if wd > 0:
                    grad = grad.add(p, alpha=wd)

                # Get momentum buffer
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]

                # SGD with momentum
                buf.mul_(momentum).add_(grad)
                if nesterov:
                    g = grad + momentum * buf
                else:
                    g = buf

                # Orthogonalize via Newton-Schulz
                # Reshape for matrix operations if needed
                original_shape = g.shape
                if g.ndim > 2:
                    g = g.view(-1, g.shape[-1])

                g_ortho = zeropower_via_newtonschulz5(g, steps=ns_steps)

                if g_ortho.ndim > 2:
                    g_ortho = g_ortho.view(original_shape)

                # Scale by learning rate and apply update
                p.add_(g_ortho, alpha=-lr)

        return loss

    def offload_to_cpu(self):
        """Offload optimizer state to CPU for ZeRO offload."""
        for state in self.state.values():
            if "momentum_buffer" in state:
                buf = state["momentum_buffer"]
                if isinstance(buf, Tensor):
                    state["momentum_buffer"] = buf.to("cpu", non_blocking=True)

    def load_from_gpu(self, device_id: Optional[int] = None):
        """Load optimizer state back to GPU."""
        device = torch.device(f"cuda:{device_id}" if device_id is not None else "cuda")
        for state in self.state.values():
            if "momentum_buffer" in state:
                buf = state["momentum_buffer"]
                if isinstance(buf, Tensor) and buf.device != device:
                    state["momentum_buffer"] = buf.to(device, non_blocking=True)


class MuonWithAdamW(Optimizer):
    """Combined optimizer: Muon for 2D matrices, AdamW for everything else.

    This is the recommended way to use Muon in practice. It automatically
    separates parameters into matrix (Muon) and non-matrix (AdamW) groups.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate for Muon (2D parameters) (default: 0.02)
        lr_embedding: Learning rate for embeddings/non-matrix parameters (default: 0.2)
        momentum: Momentum factor for Muon (default: 0.95)
        nesterov: Use Nesterov momentum for Muon (default: True)
        ns_steps: Number of Newton-Schulz iterations for Muon (default: 5)
        adamw_lr: Learning rate for AdamW (default: 3e-4)
        adamw_betas: AdamW betas (default: (0.9, 0.95))
        adamw_weight_decay: AdamW weight decay (default: 0.1)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        lr_embedding: float = 0.2,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_weight_decay: float = 0.1,
    ):
        # Separate parameters into matrix and non-matrix groups
        matrix_params = []
        embedding_params = []  # Includes embeddings, LayerNorm gains, biases

        for p in params:
            if p.ndim >= 2:
                matrix_params.append(p)
            else:
                embedding_params.append(p)

        # Create sub-optimizers
        self.muon_optimizer = Muon(
            matrix_params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=0.0,  # We'll handle weight decay separately
        )

        self.adamw_optimizer = torch.optim.AdamW(
            embedding_params,
            lr=adamw_lr,
            betas=adamw_betas,
            weight_decay=adamw_weight_decay,
        )

        # Store config for reference
        self.muon_lr = lr
        self.embedding_lr = lr_embedding
        self.adamw_lr = adamw_lr

        # Combine param_groups for compatibility
        super().__init__(params)  # Dummy init, we manage our own

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Optional loss from closure
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Muon step with weight decay
        for group in self.muon_optimizer.param_groups:
            wd = group.get("weight_decay", 0.0)
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if wd > 0:
                    grad = grad.add(p, alpha=wd)

                state = self.muon_optimizer.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]

                buf.mul_(momentum).add_(grad)
                if nesterov:
                    g = grad + momentum * buf
                else:
                    g = buf

                original_shape = g.shape
                if g.ndim > 2:
                    g = g.view(-1, g.shape[-1])

                g_ortho = zeropower_via_newtonschulz5(g, steps=ns_steps)

                if g_ortho.ndim > 2:
                    g_ortho = g_ortho.view(original_shape)

                p.add_(g_ortho, alpha=-lr)

        # AdamW step
        self.adamw_optimizer.step(closure)

        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients of all parameter groups."""
        self.muon_optimizer.zero_grad(set_to_none)
        self.adamw_optimizer.zero_grad(set_to_none)

    def offload_to_cpu(self):
        """Offload optimizer states to CPU for ZeRO offload."""
        self.muon_optimizer.offload_to_cpu()

    def load_from_gpu(self, device_id: Optional[int] = None):
        """Load optimizer states back to GPU."""
        self.muon_optimizer.load_from_gpu(device_id)

    def state_dict(self):
        """Return optimizer state dict for checkpointing."""
        return {
            "muon_state": self.muon_optimizer.state_dict(),
            "adamw_state": self.adamw_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state dict from checkpoint."""
        self.muon_optimizer.load_state_dict(state_dict["muon_state"])
        self.adamw_optimizer.load_state_dict(state_dict["adamw_state"])