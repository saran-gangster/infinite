# infinite/env/wrapper.py
from typing import Any, Dict, List, Tuple

import torch
import verifiers as vf
from openai import AsyncOpenAI
from transformers import PreTrainedTokenizerBase

from config.types import VerifiersEnvConfig
from train.datasets.base import tokenize_messages

class VerifiersEnvWrapper:
    """An adapter to use verifiers environments within the infinite framework."""
    env: vf.Environment

    def __init__(
        self,
        env_configs: List[VerifiersEnvConfig],
        tokenizer: PreTrainedTokenizerBase,
        sglang_endpoint_url: str,
    ):
        if not env_configs:
            raise ValueError("At least one environment configuration must be provided.")
        self.tokenizer = tokenizer
        
        envs = [vf.load_environment(config.id, **config.kwargs) for config in env_configs]
        self.env = vf.EnvGroup(envs=envs, env_names=[c.id for c in env_configs]) if len(envs) > 1 else envs[0]
        
        self.client = AsyncOpenAI(base_url=sglang_endpoint_url, api_key="DUMMY_KEY")

    async def rollout(
        self, ex: Dict[str, Any], model_name: str, sampling_params: Dict[str, Any], apply_chat_template: bool
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]], float]:
        """Performs a rollout using the verifiers environment and translates the output."""
        prompt = ex["prompt"]
        completion, state = await self.env.rollout(
            client=self.client, model=model_name, prompt=prompt,
            answer=ex.get("answer"), info=ex.get("info"), sampling_args=sampling_params
        )
        final_messages = list(prompt) + completion
        reward = state.get("reward", 0.0)

        tensor_dict = tokenize_messages(
            tokenizer=self.tokenizer, messages=final_messages, apply_chat_template=apply_chat_template
        )
        tensor_dict["rewards"] = torch.FloatTensor((len(tensor_dict["states"]) - 1) * [0] + [reward])
        tensor_dict["domain_id"] = state.get("task", "default_domain")

        return tensor_dict, final_messages, reward