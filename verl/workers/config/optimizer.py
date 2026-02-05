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
import warnings
from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig

__all__ = [
    "OptimizerConfig",
    "FSDPOptimizerConfig",
    "McoreOptimizerConfig",
    "MuonOptimizerConfig",
    "build_optimizer",
    "VeOmniOptimizerConfig",
]


@dataclass
class OptimizerConfig(BaseConfig):
    """Base optimizer configuration.

    Args:
        lr (float): learning rate. Must be specified.
        lr_warmup_steps_ratio (float): Warmup steps ratio; total steps will be injected at runtime.
        total_training_steps (int): Total training steps (must be overridden at runtime).
        weight_decay (float): Weight decay factor.
        lr_warmup_steps (Optional[int]): Number of warmup steps; None delegates to lr_warmup_steps_ratio.
    """

    _mutable_fields = {"clip_grad", "total_training_steps", "lr_warmup_steps"}

    lr: float = 1e-3
    lr_warmup_steps_ratio: float = 0.0
    total_training_steps: int = -1
    weight_decay: float = 0.01
    lr_warmup_steps: Optional[int] = -1
    betas: tuple[float, float] = (0.9, 0.999)
    clip_grad: float = 1.0
    # deprecate grad_clip
    grad_clip: Optional[float] = None

    def __post_init__(self):
        assert self.lr != MISSING
        if self.grad_clip is not None:
            warnings.warn("`grad_clip` is deprecated, use `clip_grad` instead.", DeprecationWarning, stacklevel=2)
            self.clip_grad = self.grad_clip


@dataclass
class VeOmniOptimizerConfig(OptimizerConfig):
    """VeOmni optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer name; default is "adamw".
        lr (float): Learning rate.
        lr_min (float): Minimum learning rate.
        lr_start (float): Starting learning rate for warmup.
        lr_decay_ratio (float): LR decay ratio.
        lr_scheduler_type (str): LR scheduler type: "constant" or "cosine".
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()

    optimizer: str = "adamw"
    lr_min: float = 0.0
    lr_start: float = 0.0
    lr_decay_ratio: float = 1.0
    lr_scheduler_type: str = "constant"
    override_optimizer_config: Optional[dict] = None


@dataclass
class FSDPOptimizerConfig(OptimizerConfig):
    """FSDP optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer class name (e.g., "AdamW", "AdamW8bit", "_AdamW").
        optimizer_impl (str): Module path to import optimizer from (e.g., "torch.optim", "torchao.optim",
            "bitsandbytes.optim").
        lr (float): Learning rate.
        min_lr_ratio (Optional[float]): Minimum LR ratio for cosine schedule.
        lr_scheduler_type (str): LR scheduler type: "constant" or "cosine".
        num_cycles (float): Number of cosine cycles in LR schedule.
    """

    _mutable_fields = OptimizerConfig._mutable_fields.copy()
    _mutable_fields.add("lr_scheduler_type")

    optimizer: str = "AdamW"
    optimizer_impl: str = "torch.optim"
    min_lr_ratio: Optional[float] = None
    # deprecate warmup_style
    warmup_style: Optional[str] = None
    lr_scheduler_type: str = "constant"
    num_cycles: float = 0.5
    override_optimizer_config: Optional[dict] = None

    def __post_init__(self):
        if self.warmup_style is not None:
            assert self.warmup_style in ["constant", "cosine"]
            warnings.warn(
                "`warmup_style` is deprecated, use `lr_scheduler_type` instead.", DeprecationWarning, stacklevel=2
            )
            self.lr_scheduler_type = self.warmup_style
        assert self.lr_scheduler_type in ["constant", "cosine"]
        return super().__post_init__()


@dataclass
class McoreOptimizerConfig(OptimizerConfig):
    """Mcore optimizer configuration extending base OptimizerConfig.

    Args:
        optimizer (str): Optimizer name; default is "adam".
        lr (float): Learning rate.
        clip_grad (float): Gradient clipping norm.
        lr_warmup_init (float): Initial learning rate for warmup; defaults to 0.0.
        lr_decay_steps (Optional[int]): Number of decay steps.
        lr_decay_style (str): LR decay style: "constant", "linear", "cosine", or "inverse_square_root".
        min_lr (float): Minimum learning rate.
        weight_decay_incr_style (str): Weight decay increment style: "constant" or "cosine".
        lr_wsd_decay_style (str): Weight-standard-deviation decay style: "constant", "exponential", or "cosine".
        lr_wsd_decay_steps (Optional[int]): Number of steps for weight-standard-deviation decay.
        use_checkpoint_opt_param_scheduler (bool): Whether to use checkpoint optimizer parameter scheduler.
    """

    optimizer: str = "adam"
    lr_warmup_init: float = 0.0
    lr_decay_steps: Optional[int] = None
    lr_decay_style: str = "linear"
    min_lr: float = 0.0
    weight_decay_incr_style: str = "constant"
    lr_wsd_decay_style: str = "exponential"
    lr_wsd_decay_steps: Optional[int] = None
    use_checkpoint_opt_param_scheduler: bool = False
    override_optimizer_config: Optional[dict] = None


@dataclass
class MuonOptimizerConfig(FSDPOptimizerConfig):
    """Muon optimizer configuration extending FSDPOptimizerConfig.

    Muon (MomentUm Orthogonalized by Newton-Schulz) is designed for 2D matrix parameters
    in neural networks. It should be combined with AdamW for non-matrix parameters.

    Reference: https://github.com/KellerJordan/Muon

    Args:
        optimizer (str): Optimizer class name; default is "MuonWithAdamW".
        optimizer_impl (str): Module path to import optimizer from; default is "verl.optim.muon".
        lr (float): Learning rate for Muon (2D parameters); default is 0.02.
        lr_embedding (float): Learning rate for embeddings/non-matrix parameters; default is 0.2.
        momentum (float): Momentum factor for Muon; default is 0.95.
        nesterov (bool): Use Nesterov momentum for Muon; default is True.
        ns_steps (int): Number of Newton-Schulz iterations for Muon; default is 5.
        adamw_lr (float): Learning rate for AdamW fallback; default is 3e-4.
        adamw_betas (tuple[float, float]): AdamW betas; default is (0.9, 0.95).
        adamw_weight_decay (float): AdamW weight decay; default is 0.1.
    """

    optimizer: str = "MuonWithAdamW"
    optimizer_impl: str = "verl.optim.muon"
    lr_embedding: Optional[float] = None
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    adamw_lr: float = 3e-4
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_weight_decay: float = 0.1


def build_optimizer(parameters, config: FSDPOptimizerConfig):
    """Build an optimizer based on the configuration.

    Dynamically imports and instantiates an optimizer class from the specified module.

    Args:
        parameters: Model parameters to optimize
        config: FSDPOptimizerConfig with optimizer settings

    Returns:
        Optimizer instance

    Examples:
        # PyTorch AdamW
        config.optimizer_impl = "torch.optim"
        config.optimizer = "AdamW"

        # TorchAO AdamW with bf16 stochastic rounding
        config.optimizer_impl = "torchao.optim"
        config.optimizer = "_AdamW"
        config.override_optimizer_config = {"bf16_stochastic_round": True}

        # BitsAndBytes AdamW 8bit
        config.optimizer_impl = "bitsandbytes.optim"
        config.optimizer = "AdamW8bit"

        # Muon optimizer (for 2D params with AdamW fallback)
        from verl.workers.config import MuonOptimizerConfig
        config = MuonOptimizerConfig(
            optimizer="MuonWithAdamW",
            optimizer_impl="verl.optim.muon",
            lr=0.02,  # Muon lr
            lr_embedding=0.2,  # embeddings lr
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adamw_lr=3e-4,
            adamw_weight_decay=0.1,
        )
    """
    import importlib

    # Handle Muon optimizer specially (has different arg names)
    optimizer_name_lower = config.optimizer.lower()
    if "muon" in optimizer_name_lower:
        # Muon-specific arguments
        import verl.optim.muon as muon_module
        optimizer_cls = getattr(muon_module, config.optimizer)

        muon_args = {
            "lr": config.lr,
        }

        if hasattr(config, "lr_embedding") and config.lr_embedding is not None:
            muon_args["lr_embedding"] = config.lr_embedding
        if hasattr(config, "momentum"):
            muon_args["momentum"] = config.momentum
        if hasattr(config, "nesterov"):
            muon_args["nesterov"] = config.nesterov
        if hasattr(config, "ns_steps"):
            muon_args["ns_steps"] = config.ns_steps
        if hasattr(config, "adamw_lr"):
            muon_args["adamw_lr"] = config.adamw_lr
        if hasattr(config, "adamw_betas"):
            muon_args["adamw_betas"] = config.adamw_betas
        if hasattr(config, "adamw_weight_decay"):
            muon_args["adamw_weight_decay"] = config.adamw_weight_decay

        return optimizer_cls(parameters, **muon_args)

    # Standard optimizer arguments
    optimizer_args = {
        "lr": config.lr,
        "weight_decay": config.weight_decay,
    }

    if "adam" in optimizer_name_lower or "ademamix" in optimizer_name_lower:
        optimizer_args["betas"] = config.betas

    if config.override_optimizer_config is not None:
        optimizer_args.update(config.override_optimizer_config)

    try:
        module = importlib.import_module(config.optimizer_impl)
        optimizer_cls = getattr(module, config.optimizer)
    except ImportError as e:
        raise ImportError(
            f"Failed to import module '{config.optimizer_impl}'. Make sure the package is installed. Error: {e}"
        ) from e
    except AttributeError as e:
        raise AttributeError(
            f"Optimizer '{config.optimizer}' not found in module '{config.optimizer_impl}'. "
            f"Available optimizers: {dir(module)}"
        ) from e

    return optimizer_cls(parameters, **optimizer_args)
