# Ported from /Users/abhishekmishra/Documents/infinite/RL2/RL2/workers/rollout.py
# Original implementation from RL2 repository

from omegaconf import OmegaConf
import os
import json
import asyncio
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from tqdm.asyncio import tqdm
import wandb
from train.workers.base import Worker
from env.wrapper import VerifiersEnvWrapper
from train.utils.comm import split_and_scatter_list, gather_and_concat_list
from train.utils.logging import time_logger, gather_and_log


class Rollout(Worker):
    def __init__(self, config):
        super().__init__(config, train=None)
        self.env_wrapper: VerifiersEnvWrapper

        self.prepare_environment_variables()

        if self.device_mesh["tp"].get_local_rank() == 0:
            sglang_port = 30000 + dist.get_rank()
            sglang_endpoint = f"http://localhost:{sglang_port}/v1"

            self.env_wrapper = VerifiersEnvWrapper(
                env_configs=config.verifiers_envs,
                tokenizer=self.tokenizer,
                sglang_endpoint_url=sglang_endpoint,
            )

            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self.llm = Engine(
                model_path=config.model_name,
                dtype="bfloat16",
                tp_size=self.device_mesh["tp"].size(),
                mem_fraction_static=config.gpu_memory_utilization,
                enable_memory_saver=True,
                port=sglang_port,
            )

            self.train_sampling_params = OmegaConf.to_container(
                config.train_sampling_params
            )
            self.test_sampling_params = OmegaConf.to_container(
                config.test_sampling_params
            )

        dist.barrier()

    def prepare_device_mesh(self):
        world_size = dist.get_world_size()
        assert world_size % self.config.tp_size == 0, (
            f"World_size {world_size} must be divisible by tp_size {self.config.tp_size}."
        )
        self.dp_size = world_size // self.config.tp_size
        self.device_mesh = dist.device_mesh.init_device_mesh(
            "cpu",
            mesh_dim_names=("dp", "tp"),
            mesh_shape=(self.dp_size, self.config.tp_size),
        )

    def prepare_environment_variables(self):
        if "TORCHELASTIC_USE_AGENT_STORE" in os.environ:
            del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
        monkey_patch_torch_reductions()

        cuda_visible_devices_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices_str:
            cuda_devices = cuda_visible_devices_str.split(",")
            local_rank_device = cuda_devices[int(os.environ["LOCAL_RANK"])]
        else:
            local_rank_device = os.environ["LOCAL_RANK"]

        all_tp_devices = self.device_mesh["tp"].size() * [None]
        dist.all_gather_object(
            all_tp_devices, local_rank_device, self.device_mesh["tp"].get_group()
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(all_tp_devices)

    async def rollout(self, ex: dict, train: bool):
        tensor_dict, final_messages, reward = await self.env_wrapper.rollout(
            ex=ex,
            model_name=self.config.model_name,
            sampling_params=(
                self.train_sampling_params if train else self.test_sampling_params
            ),
            apply_chat_template=self.config.apply_chat_template,
        )

        metric = defaultdict(list)
        metric["n_turns"].append(len(final_messages) // 2)
        metric["rewards"].append(reward)
        metric["trajectory_length"].append(len(tensor_dict["states"]))

        return tensor_dict, final_messages, metric

    @time_logger("rollout")
    def __call__(self, data_list, train: bool, step: int):
        if self.device_mesh["tp"].get_local_rank() == 0:
            data_list = split_and_scatter_list(data_list, self.device_mesh["dp"])
            loop = asyncio.get_event_loop()
            outputs = loop.run_until_complete(
                tqdm.gather(
                    *(self.rollout(ex, train) for ex in data_list),
                    desc="Rollout",
                    position=1,
                    leave=False,
                    disable=(dist.get_rank() != 0),
                )
            )
            if train:
                self.llm.release_memory_occupation()

        dist.barrier()

        if self.device_mesh["tp"].get_local_rank() == 0:
            tensor_dicts, all_messages, metrics_list = map(list, zip(*outputs))

            if dist.get_rank() == 0:
                tqdm.write(json.dumps(all_messages[0], indent=2))
            
            suffix = "train" if train else "test"
            aggregated_metrics = {
                f"{key}/{suffix}": sum([m[key] for m in metrics_list], [])
                for key in metrics_list[0].keys()
            }
            gather_and_log(aggregated_metrics, self.device_mesh["dp"], step)

            if not train:
                return
            
            tensor_dicts = gather_and_concat_list(tensor_dicts, self.device_mesh["dp"])

            if dist.get_rank() == 0 and self.config.dynamic_filtering:
                rewards = torch.FloatTensor(
                    [td["rewards"].sum() for td in tensor_dicts]
                ).view(-1, self.config.responses_per_prompt)
                is_filtered = (rewards.std(-1) == 0).tolist()
                wandb.log(
                    {"dynamic_filtering_ratio": sum(is_filtered) / len(is_filtered)},
                     step=step
                )
                return sum(
                    [
                        tensor_dicts[
                            i * self.config.responses_per_prompt : 
                            (i + 1) * self.config.responses_per_prompt
                        ]
                        for i, filtered in enumerate(is_filtered) if not filtered
                    ],
                    [],
                )
            return tensor_dicts
        return None

    @time_logger("update_rollout")
    def update(self, state_dict: dict, step: int):
        torch.cuda.empty_cache()
        if self.device_mesh["tp"].get_local_rank() == 0:
            self.llm.resume_memory_occupation()
        
        for idx, (name, tensor) in enumerate(state_dict.items()):
            tensor = tensor.to(torch.cuda.current_device())
            serialized_tensor = MultiprocessingSerializer.serialize(
                tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
            )
            serialized_tensors = (
                [None for _ in range(self.device_mesh["tp"].size())]
                if self.device_mesh["tp"].get_local_rank() == 0
                else None
            )
            dist.gather_object(
                serialized_tensor,
                serialized_tensors,
                group_dst=0,
                group=self.device_mesh["tp"].get_group(),
            )
            if self.device_mesh["tp"].get_local_rank() == 0:
                self.llm.update_weights_from_tensor(
                    named_tensors=[(name, LocalSerializedTensor(values=serialized_tensors))],
                    flush_cache=(idx == len(state_dict) - 1),
                )
        dist.barrier()