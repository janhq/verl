# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
from __future__ import annotations

import logging
import multiprocessing as mp
import os
from typing import Generator

import ray
import sglang.srt.entrypoints.engine
import torch
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    assert_pkg_version,
    is_cuda,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights
from torch.distributed.device_mesh import DeviceMesh

from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.sglang_rollout.http_server_engine import AsyncHttpServerAdapter
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets
from verl.workers.rollout.utils import is_valid_ipv6_address

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# patch to avoid issue https://github.com/sgl-project/sglang/issues/6723
def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)


sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config


# because chatCompletion is an async method, it makes the whole ray actor be an async actor
# which can not call loop.run_until_complete. So we need to make the engine to be an async class
class ServerAdapter(BaseRollout):
    """SGLang server adapter used in native http server mode, serve as http client to request SGLang server
    to resume/release/update weights and kv_cache.

    - hybrid mode: reside in each hybrid worker to sync weights between training engine and SGLang server.
    - standalone/colocated mode: just a dummy placeholder to occupy the GPU to prevent ray scheduling new GPU actor.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        if config.get("quantization", None) == "fp8":
            import sglang

            assert sglang.__version__ >= "0.5.5", "sglang>=0.5.5 is required for FP8 quantization"
            FP8_BLOCK_QUANT_KWARGS = {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "ignored_layers": [
                        "lm_head",
                        "model.visual.merger.linear_fc1",
                        "model.visual.merger.linear_fc2",
                        "model.visual.merger.norm",
                        "model.visual.patch_embed.proj",
                        "model.visual.pos_embed",
                        "visual.merger.linear_fc1",
                        "visual.merger.linear_fc2",
                        "visual.merger.norm",
                        "visual.patch_embed.proj",
                        "visual.pos_embed",
                        "model.visual.blocks.0.attn.proj",
                        "model.visual.blocks.0.attn.qkv",
                        "model.visual.blocks.0.mlp.linear_fc1",
                        "model.visual.blocks.0.mlp.linear_fc2",
                        "visual.blocks.0.attn.proj",
                        "visual.blocks.0.attn.qkv_proj",
                        "visual.blocks.0.mlp.linear_fc1",
                        "visual.blocks.0.mlp.linear_fc2",
                        "model.visual.blocks.1.attn.proj",
                        "model.visual.blocks.1.attn.qkv",
                        "model.visual.blocks.1.mlp.linear_fc1",
                        "model.visual.blocks.1.mlp.linear_fc2",
                        "visual.blocks.1.attn.proj",
                        "visual.blocks.1.attn.qkv_proj",
                        "visual.blocks.1.mlp.linear_fc1",
                        "visual.blocks.1.mlp.linear_fc2",
                        "model.visual.blocks.2.attn.proj",
                        "model.visual.blocks.2.attn.qkv",
                        "model.visual.blocks.2.mlp.linear_fc1",
                        "model.visual.blocks.2.mlp.linear_fc2",
                        "visual.blocks.2.attn.proj",
                        "visual.blocks.2.attn.qkv_proj",
                        "visual.blocks.2.mlp.linear_fc1",
                        "visual.blocks.2.mlp.linear_fc2",
                        "model.visual.blocks.3.attn.proj",
                        "model.visual.blocks.3.attn.qkv",
                        "model.visual.blocks.3.mlp.linear_fc1",
                        "model.visual.blocks.3.mlp.linear_fc2",
                        "visual.blocks.3.attn.proj",
                        "visual.blocks.3.attn.qkv_proj",
                        "visual.blocks.3.mlp.linear_fc1",
                        "visual.blocks.3.mlp.linear_fc2",
                        "model.visual.blocks.4.attn.proj",
                        "model.visual.blocks.4.attn.qkv",
                        "model.visual.blocks.4.mlp.linear_fc1",
                        "model.visual.blocks.4.mlp.linear_fc2",
                        "visual.blocks.4.attn.proj",
                        "visual.blocks.4.attn.qkv_proj",
                        "visual.blocks.4.mlp.linear_fc1",
                        "visual.blocks.4.mlp.linear_fc2",
                        "model.visual.blocks.5.attn.proj",
                        "model.visual.blocks.5.attn.qkv",
                        "model.visual.blocks.5.mlp.linear_fc1",
                        "model.visual.blocks.5.mlp.linear_fc2",
                        "visual.blocks.5.attn.proj",
                        "visual.blocks.5.attn.qkv_proj",
                        "visual.blocks.5.mlp.linear_fc1",
                        "visual.blocks.5.mlp.linear_fc2",
                        "model.visual.blocks.6.attn.proj",
                        "model.visual.blocks.6.attn.qkv",
                        "model.visual.blocks.6.mlp.linear_fc1",
                        "model.visual.blocks.6.mlp.linear_fc2",
                        "visual.blocks.6.attn.proj",
                        "visual.blocks.6.attn.qkv_proj",
                        "visual.blocks.6.mlp.linear_fc1",
                        "visual.blocks.6.mlp.linear_fc2",
                        "model.visual.blocks.7.attn.proj",
                        "model.visual.blocks.7.attn.qkv",
                        "model.visual.blocks.7.mlp.linear_fc1",
                        "model.visual.blocks.7.mlp.linear_fc2",
                        "visual.blocks.7.attn.proj",
                        "visual.blocks.7.attn.qkv_proj",
                        "visual.blocks.7.mlp.linear_fc1",
                        "visual.blocks.7.mlp.linear_fc2",
                        "model.visual.blocks.8.attn.proj",
                        "model.visual.blocks.8.attn.qkv",
                        "model.visual.blocks.8.mlp.linear_fc1",
                        "model.visual.blocks.8.mlp.linear_fc2",
                        "visual.blocks.8.attn.proj",
                        "visual.blocks.8.attn.qkv_proj",
                        "visual.blocks.8.mlp.linear_fc1",
                        "visual.blocks.8.mlp.linear_fc2",
                        "model.visual.blocks.9.attn.proj",
                        "model.visual.blocks.9.attn.qkv",
                        "model.visual.blocks.9.mlp.linear_fc1",
                        "model.visual.blocks.9.mlp.linear_fc2",
                        "visual.blocks.9.attn.proj",
                        "visual.blocks.9.attn.qkv_proj",
                        "visual.blocks.9.mlp.linear_fc1",
                        "visual.blocks.9.mlp.linear_fc2",
                        "model.visual.blocks.10.attn.proj",
                        "model.visual.blocks.10.attn.qkv",
                        "model.visual.blocks.10.mlp.linear_fc1",
                        "model.visual.blocks.10.mlp.linear_fc2",
                        "visual.blocks.10.attn.proj",
                        "visual.blocks.10.attn.qkv_proj",
                        "visual.blocks.10.mlp.linear_fc1",
                        "visual.blocks.10.mlp.linear_fc2",
                        "model.visual.blocks.11.attn.proj",
                        "model.visual.blocks.11.attn.qkv",
                        "model.visual.blocks.11.mlp.linear_fc1",
                        "model.visual.blocks.11.mlp.linear_fc2",
                        "visual.blocks.11.attn.proj",
                        "visual.blocks.11.attn.qkv_proj",
                        "visual.blocks.11.mlp.linear_fc1",
                        "visual.blocks.11.mlp.linear_fc2",
                        "model.visual.blocks.12.attn.proj",
                        "model.visual.blocks.12.attn.qkv",
                        "model.visual.blocks.12.mlp.linear_fc1",
                        "model.visual.blocks.12.mlp.linear_fc2",
                        "visual.blocks.12.attn.proj",
                        "visual.blocks.12.attn.qkv_proj",
                        "visual.blocks.12.mlp.linear_fc1",
                        "visual.blocks.12.mlp.linear_fc2",
                        "model.visual.blocks.13.attn.proj",
                        "model.visual.blocks.13.attn.qkv",
                        "model.visual.blocks.13.mlp.linear_fc1",
                        "model.visual.blocks.13.mlp.linear_fc2",
                        "visual.blocks.13.attn.proj",
                        "visual.blocks.13.attn.qkv_proj",
                        "visual.blocks.13.mlp.linear_fc1",
                        "visual.blocks.13.mlp.linear_fc2",
                        "model.visual.blocks.14.attn.proj",
                        "model.visual.blocks.14.attn.qkv",
                        "model.visual.blocks.14.mlp.linear_fc1",
                        "model.visual.blocks.14.mlp.linear_fc2",
                        "visual.blocks.14.attn.proj",
                        "visual.blocks.14.attn.qkv_proj",
                        "visual.blocks.14.mlp.linear_fc1",
                        "visual.blocks.14.mlp.linear_fc2",
                        "model.visual.blocks.15.attn.proj",
                        "model.visual.blocks.15.attn.qkv",
                        "model.visual.blocks.15.mlp.linear_fc1",
                        "model.visual.blocks.15.mlp.linear_fc2",
                        "visual.blocks.15.attn.proj",
                        "visual.blocks.15.attn.qkv_proj",
                        "visual.blocks.15.mlp.linear_fc1",
                        "visual.blocks.15.mlp.linear_fc2",
                        "model.visual.blocks.16.attn.proj",
                        "model.visual.blocks.16.attn.qkv",
                        "model.visual.blocks.16.mlp.linear_fc1",
                        "model.visual.blocks.16.mlp.linear_fc2",
                        "visual.blocks.16.attn.proj",
                        "visual.blocks.16.attn.qkv_proj",
                        "visual.blocks.16.mlp.linear_fc1",
                        "visual.blocks.16.mlp.linear_fc2",
                        "model.visual.blocks.17.attn.proj",
                        "model.visual.blocks.17.attn.qkv",
                        "model.visual.blocks.17.mlp.linear_fc1",
                        "model.visual.blocks.17.mlp.linear_fc2",
                        "visual.blocks.17.attn.proj",
                        "visual.blocks.17.attn.qkv_proj",
                        "visual.blocks.17.mlp.linear_fc1",
                        "visual.blocks.17.mlp.linear_fc2",
                        "model.visual.blocks.18.attn.proj",
                        "model.visual.blocks.18.attn.qkv",
                        "model.visual.blocks.18.mlp.linear_fc1",
                        "model.visual.blocks.18.mlp.linear_fc2",
                        "visual.blocks.18.attn.proj",
                        "visual.blocks.18.attn.qkv_proj",
                        "visual.blocks.18.mlp.linear_fc1",
                        "visual.blocks.18.mlp.linear_fc2",
                        "model.visual.blocks.19.attn.proj",
                        "model.visual.blocks.19.attn.qkv",
                        "model.visual.blocks.19.mlp.linear_fc1",
                        "model.visual.blocks.19.mlp.linear_fc2",
                        "visual.blocks.19.attn.proj",
                        "visual.blocks.19.attn.qkv_proj",
                        "visual.blocks.19.mlp.linear_fc1",
                        "visual.blocks.19.mlp.linear_fc2",
                        "model.visual.blocks.20.attn.proj",
                        "model.visual.blocks.20.attn.qkv",
                        "model.visual.blocks.20.mlp.linear_fc1",
                        "model.visual.blocks.20.mlp.linear_fc2",
                        "visual.blocks.20.attn.proj",
                        "visual.blocks.20.attn.qkv_proj",
                        "visual.blocks.20.mlp.linear_fc1",
                        "visual.blocks.20.mlp.linear_fc2",
                        "model.visual.blocks.21.attn.proj",
                        "model.visual.blocks.21.attn.qkv",
                        "model.visual.blocks.21.mlp.linear_fc1",
                        "model.visual.blocks.21.mlp.linear_fc2",
                        "visual.blocks.21.attn.proj",
                        "visual.blocks.21.attn.qkv_proj",
                        "visual.blocks.21.mlp.linear_fc1",
                        "visual.blocks.21.mlp.linear_fc2",
                        "model.visual.blocks.22.attn.proj",
                        "model.visual.blocks.22.attn.qkv",
                        "model.visual.blocks.22.mlp.linear_fc1",
                        "model.visual.blocks.22.mlp.linear_fc2",
                        "visual.blocks.22.attn.proj",
                        "visual.blocks.22.attn.qkv_proj",
                        "visual.blocks.22.mlp.linear_fc1",
                        "visual.blocks.22.mlp.linear_fc2",
                        "model.visual.blocks.23.attn.proj",
                        "model.visual.blocks.23.attn.qkv",
                        "model.visual.blocks.23.mlp.linear_fc1",
                        "model.visual.blocks.23.mlp.linear_fc2",
                        "visual.blocks.23.attn.proj",
                        "visual.blocks.23.attn.qkv_proj",
                        "visual.blocks.23.mlp.linear_fc1",
                        "visual.blocks.23.mlp.linear_fc2",
                        "model.visual.blocks.24.attn.proj",
                        "model.visual.blocks.24.attn.qkv",
                        "model.visual.blocks.24.mlp.linear_fc1",
                        "model.visual.blocks.24.mlp.linear_fc2",
                        "visual.blocks.24.attn.proj",
                        "visual.blocks.24.attn.qkv_proj",
                        "visual.blocks.24.mlp.linear_fc1",
                        "visual.blocks.24.mlp.linear_fc2",
                        "model.visual.blocks.25.attn.proj",
                        "model.visual.blocks.25.attn.qkv",
                        "model.visual.blocks.25.mlp.linear_fc1",
                        "model.visual.blocks.25.mlp.linear_fc2",
                        "visual.blocks.25.attn.proj",
                        "visual.blocks.25.attn.qkv_proj",
                        "visual.blocks.25.mlp.linear_fc1",
                        "visual.blocks.25.mlp.linear_fc2",
                        "model.visual.blocks.26.attn.proj",
                        "model.visual.blocks.26.attn.qkv",
                        "model.visual.blocks.26.mlp.linear_fc1",
                        "model.visual.blocks.26.mlp.linear_fc2",
                        "visual.blocks.26.attn.proj",
                        "visual.blocks.26.attn.qkv_proj",
                        "visual.blocks.26.mlp.linear_fc1",
                        "visual.blocks.26.mlp.linear_fc2",
                        "model.visual.deepstack_merger_list.0.linear_fc1",
                        "model.visual.deepstack_merger_list.0.linear_fc2",
                        "model.visual.deepstack_merger_list.0.norm",
                        "visual.deepstack_merger_list.0.linear_fc1",
                        "visual.deepstack_merger_list.0.linear_fc2",
                        "visual.deepstack_merger_list.0.norm",
                        "model.visual.deepstack_merger_list.1.linear_fc1",
                        "model.visual.deepstack_merger_list.1.linear_fc2",
                        "model.visual.deepstack_merger_list.1.norm",
                        "visual.deepstack_merger_list.1.linear_fc1",
                        "visual.deepstack_merger_list.1.linear_fc2",
                        "visual.deepstack_merger_list.1.norm",
                        "model.visual.deepstack_merger_list.2.linear_fc1",
                        "model.visual.deepstack_merger_list.2.linear_fc2",
                        "model.visual.deepstack_merger_list.2.norm",
                        "visual.deepstack_merger_list.2.linear_fc1",
                        "visual.deepstack_merger_list.2.linear_fc2",
                        "visual.deepstack_merger_list.2.norm",
                        "model.language_model.layers.0.mlp.gate",
                        "model.language_model.layers.1.mlp.gate",
                        "model.language_model.layers.2.mlp.gate",
                        "model.language_model.layers.3.mlp.gate",
                        "model.language_model.layers.4.mlp.gate",
                        "model.language_model.layers.5.mlp.gate",
                        "model.language_model.layers.6.mlp.gate",
                        "model.language_model.layers.7.mlp.gate",
                        "model.language_model.layers.8.mlp.gate",
                        "model.language_model.layers.9.mlp.gate",
                        "model.language_model.layers.10.mlp.gate",
                        "model.language_model.layers.11.mlp.gate",
                        "model.language_model.layers.12.mlp.gate",
                        "model.language_model.layers.13.mlp.gate",
                        "model.language_model.layers.14.mlp.gate",
                        "model.language_model.layers.15.mlp.gate",
                        "model.language_model.layers.16.mlp.gate",
                        "model.language_model.layers.17.mlp.gate",
                        "model.language_model.layers.18.mlp.gate",
                        "model.language_model.layers.19.mlp.gate",
                        "model.language_model.layers.20.mlp.gate",
                        "model.language_model.layers.21.mlp.gate",
                        "model.language_model.layers.22.mlp.gate",
                        "model.language_model.layers.23.mlp.gate",
                        "model.language_model.layers.24.mlp.gate",
                        "model.language_model.layers.25.mlp.gate",
                        "model.language_model.layers.26.mlp.gate",
                        "model.language_model.layers.27.mlp.gate",
                        "model.language_model.layers.28.mlp.gate",
                        "model.language_model.layers.29.mlp.gate",
                        "model.language_model.layers.30.mlp.gate",
                        "model.language_model.layers.31.mlp.gate",
                        "model.language_model.layers.32.mlp.gate",
                        "model.language_model.layers.33.mlp.gate",
                        "model.language_model.layers.34.mlp.gate",
                        "model.language_model.layers.35.mlp.gate",
                        "model.language_model.layers.36.mlp.gate",
                        "model.language_model.layers.37.mlp.gate",
                        "model.language_model.layers.38.mlp.gate",
                        "model.language_model.layers.39.mlp.gate",
                        "model.language_model.layers.40.mlp.gate",
                        "model.language_model.layers.41.mlp.gate",
                        "model.language_model.layers.42.mlp.gate",
                        "model.language_model.layers.43.mlp.gate",
                        "model.language_model.layers.44.mlp.gate",
                        "model.language_model.layers.45.mlp.gate",
                        "model.language_model.layers.46.mlp.gate",
                        "model.language_model.layers.47.mlp.gate"
                        ],
            }
            fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
            model_config.hf_config.quantization_config = fp8_block_quant_kwargs
        super().__init__(config, model_config, device_mesh)
        self._engine: AsyncHttpServerAdapter = None

        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        rollout_world_size = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        self.replica_rank = rank // rollout_world_size
        self.rollout_rank = rank % rollout_world_size
        self.node_rank = self.rollout_rank // local_world_size
        self.local_rank = self.rollout_rank % local_world_size

    async def _init_server_adapter(self):
        if self._engine is not None:
            return

        # Lazy init http server adapter because http server is launched after hybrid engine.
        self.server_actor = ray.get_actor(f"sglang_server_{self.replica_rank}_{self.node_rank}")
        server_address, server_port = await self.server_actor.get_server_address.remote()
        logger.debug(
            f"replica_rank={self.replica_rank} node_rank={self.node_rank}, "
            f"server address: {server_address}, port: {server_port}"
        )
        host = f"[{server_address}]" if is_valid_ipv6_address(server_address) else server_address
        self._engine = AsyncHttpServerAdapter(
            model_path=self.model_config.local_path, host=host, port=server_port, launch_server=False
        )

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tag: weights or kv_cache.
        """
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._init_server_adapter()
            await self._engine.resume_memory_occupation(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.config.free_cache_engine:
            await self._init_server_adapter()
            await self._engine.release_memory_occupation(tags=["kv_cache", "weights"])

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """
        Update model weights using tensor buckets, similar to THUDM/slime's implementation.

        Notes:
          - For the best performance of `rebuild_cuda_tensor`, it is recommended to:
              1. Enable `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`.
              2. Manually set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
            when using Tensor Parallelism (TP >= 8).
          - See reference implementations in SLIME:
            - Main logic: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L452
            - runtime envs: https://github.com/THUDM/slime/blob/fb7605cc5fb09af0f9369d37f7192f12bddee577/slime/ray/ppo_actor.py#L39
        """
        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self._init_server_adapter()

        update_weights_bucket_bytes = int(self.config.update_weights_bucket_megabytes) << 20
        if self.config.get("quantization", None) == "fp8":
            from verl.utils.sglang.sglang_fp8_utils import quant_weights_by_name

            logger.info("Convert bf16 weights to fp8 format before loading")
            weights = quant_weights_by_name(
                weights,
                self.model_config.hf_config.quantization_config,
                dtype=self.model_config.hf_config.dtype,
            )
        else:
            weights = weights

        for params_batch in get_named_tensor_buckets(weights, update_weights_bucket_bytes):
            await sgl_update_weights(
                engine=self._engine,
                params_batch=params_batch,
                device_mesh_key="infer_tp",
                device_mesh=self.device_mesh,
            )

        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self._engine.flush_cache()