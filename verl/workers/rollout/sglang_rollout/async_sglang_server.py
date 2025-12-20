# Copyright 2023-2024 SGLang Team
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import dataclasses
import json
import logging
import os
from typing import Any, Optional

import ray
import sglang
import sglang.srt.entrypoints.engine
import torch
from ray.actor import ActorHandle
from sglang.srt.entrypoints.http_server import (
    ServerArgs,
    _GlobalState,
    _launch_subprocesses,
    app,
    set_global_state,
)
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.tokenizer_manager import ServerStatus

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter, _set_envs_and_config
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address, run_unvicorn

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


@ray.remote(num_cpus=1)
class SGLangHttpServer:
    """SGLang http server in single node, this is equivalent to launch server with command line:
    ```
    python -m sglang.launch_server --node-rank 0 --nnode 1 ...
    ```

    Args:
        config (DictConfig): full config.
        rollout_mode (RolloutMode): rollout mode.
        replica_rank (int): replica rank, a replica may contain multiple nodes.
        node_rank (int): node rank.
        nnodes (int): number of nodes.
        cuda_visible_devices (str): cuda visible devices.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        print(f"SGLang http server: {rollout_mode=}, {replica_rank=}, {node_rank=}, {nnodes=}, {cuda_visible_devices=}")
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        assert torch.cuda.is_available(), "SGLang http server should run on GPU node"

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.config.max_model_len = self.config.prompt_length + self.config.response_length
        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.nnodes = nnodes

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for NCCL process group
        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, self._master_sock = get_free_port(self._server_address)
            logger.info(
                f"SGLangHttpServer, replica_rank: {self.replica_rank}, "
                f"master address: {self._master_address}, port: {self._master_port}"
            )
        else:
            self._master_address = None
            self._master_port = None

    def get_master_address(self):
        """Get master address and port for init NCCL process group."""
        return self._master_address, self._master_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port, "non-master node should provide master address and port"
            self._master_address = master_address
            self._master_port = master_port

        engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
        attention_backend = engine_kwargs.pop("attention_backend", None)
        quantization = self.config.get("quantization", None)
        if quantization is not None:
            if quantization == "fp8":
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
            else:
                raise ValueError(f"Currently only support fp8 quantization, got: {quantization}")
        dist_init_addr = (
            f"[{self._master_address}]:{self._master_port}"
            if is_valid_ipv6_address(self._master_address)
            else f"{self._master_address}:{self._master_port}"
        )

        args = {
            "model_path": self.model_config.local_path,
            "dtype": self.config.dtype,
            "mem_fraction_static": self.config.gpu_memory_utilization,
            "disable_cuda_graph": self.config.enforce_eager,
            "enable_memory_saver": True,
            "base_gpu_id": 0,
            "gpu_id_step": 1,
            "tp_size": self.config.tensor_model_parallel_size,
            "dp_size": self.config.data_parallel_size,
            "ep_size": self.config.expert_parallel_size,
            "node_rank": self.node_rank,
            "load_format": self.config.load_format,
            "dist_init_addr": dist_init_addr,
            "nnodes": self.nnodes,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_running_requests": self.config.get("max_num_seqs", None),
            "log_level": "error",
            "mm_attention_backend": "triton_attn",
            "attention_backend": attention_backend if attention_backend is not None else "triton",
            "cuda_graph_max_bs" : 512,
            "skip_tokenizer_init": self.config.skip_tokenizer_init,
            "skip_server_warmup": True,
            "quantization": quantization,
            # "disable_cuda_graph": True,
            "json_model_override_args": json.dumps({"quantization_config": fp8_block_quant_kwargs})
            if quantization == "fp8"
            else json.dumps({}),
            **engine_kwargs,
        }

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

            # start sglang metrics
            args["enable_metrics"] = True

        # enable_weights_cpu_backup is supported in sglang>=0.5.3
        if "enable_weights_cpu_backup" in [f.name for f in dataclasses.fields(ServerArgs)]:
            enable_weights_cpu_backup = True if self.rollout_mode == RolloutMode.COLOCATED else False
            args["enable_weights_cpu_backup"] = enable_weights_cpu_backup

        # NOTE: We can't directly call SGLang's launch_server since it's not an async function.
        # https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py
        sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        server_args = ServerArgs(**args)
        self.tokenizer_manager, self.template_manager, self.scheduler_info, *_ = _launch_subprocesses(
            server_args=server_args
        )

        # In multi-node cases, non-zero rank nodes should not launch http server.
        if self.node_rank > 0:
            return

        set_global_state(
            _GlobalState(
                tokenizer_manager=self.tokenizer_manager,
                template_manager=self.template_manager,
                scheduler_info=self.scheduler_info,
            )
        )
        app.is_single_tokenizer_mode = True

        # Set warmup_thread_args to avoid AttributeError in lifespan function
        app.warmup_thread_args = (
            server_args,
            None,
            None,
        )

        # Manually add Prometheus middleware before starting server
        # This ensures /metrics endpoint is available immediately
        if server_args.enable_metrics:
            from sglang.srt.utils.common import add_prometheus_middleware

            add_prometheus_middleware(app)

        self._server_port, self._server_task = await run_unvicorn(app, server_args, self._server_address)
        self.tokenizer_manager.server_status = ServerStatus.Up

    async def wake_up(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            # Call all workers to switch between trainer mode and rollout mode.
            await asyncio.gather(*[worker.wake_up.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            obj = ResumeMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.resume_memory_occupation(obj, None)
            await self.tokenizer_manager.flush_cache()
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if self.rollout_mode == RolloutMode.HYBRID:
            await asyncio.gather(*[worker.sleep.remote() for worker in self.workers])
        elif self.rollout_mode == RolloutMode.COLOCATED:
            obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.release_memory_occupation(obj, None)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def clear_kv_cache(self):
        obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
        await self.tokenizer_manager.release_memory_occupation(obj, None)

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        # TODO(@wuxibin): switch to `/generate` http endpoint once multi-modal support ready.
        max_new_tokens = min(self.config.response_length, self.config.max_model_len - len(prompt_ids) - 1)
        sampling_params["max_new_tokens"] = max_new_tokens
        return_logprob = sampling_params.pop("logprobs", False)

        request = GenerateReqInput(
            rid=request_id,
            input_ids=prompt_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            image_data=image_data,
        )
        output = await self.tokenizer_manager.generate_request(request, None).__anext__()
        print("########## OUTPUT", output)
        if return_logprob:
            output_token_logprobs = output["meta_info"]["output_token_logprobs"]
            log_probs, token_ids = zip(
                *[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs], strict=True
            )
        else:
            token_ids = output["output_ids"]
            log_probs = None
        return TokenOutput(token_ids=token_ids, log_probs=log_probs)


_rollout_worker_actor_cls = ray.remote(ServerAdapter)


class SGLangReplica(RolloutReplica):
    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        worker_dict_cls = RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )
        return worker_dict_cls

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (ray.get_runtime_context().get_node_id(), os.environ["CUDA_VISIBLE_DEVICES"])
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]

        # create server actor in each node with node affinity and cuda visible devices
        for node_rank in range(self.nnodes):
            workers = self.workers[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]
            node_cuda_visible_devices = ",".join(
                worker_cuda_visible_devices[node_rank * self.gpus_per_node : (node_rank + 1) * self.gpus_per_node]
            )
            node_id = worker_node_ids[node_rank * self.gpus_per_node]
            name = (
                f"sglang_server_{self.replica_rank}_{node_rank}"
                if not self.is_reward_model
                else f"sglang_server_reward_{self.replica_rank}_{node_rank}"
            )
            server = SGLangHttpServer.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}},
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                nnodes=self.nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )
