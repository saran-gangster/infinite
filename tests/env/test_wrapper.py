import pytest
import torch
from unittest.mock import AsyncMock, MagicMock
from infinite.config.types import VerifiersEnvConfig
from infinite.env.wrapper import VerifiersEnvWrapper

@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    def mock_encode(text, add_special_tokens=False): return [ord(c) for c in text]
    tokenizer.encode.side_effect = mock_encode
    return tokenizer

@pytest.fixture
def mock_verifiers_env():
    env = AsyncMock()
    env.rollout.return_value = ([{"role": "assistant", "content": "42"}], {"reward": 1.0, "task": "math"})
    return env

@pytest.mark.asyncio
async def test_verifiers_env_wrapper_rollout(mock_tokenizer, mock_verifiers_env):
    import verifiers as vf
    vf.load_environment = MagicMock(return_value=mock_verifiers_env)
    
    wrapper = VerifiersEnvWrapper(
        env_configs=[VerifiersEnvConfig(id="mock-env")], tokenizer=mock_tokenizer,
        sglang_endpoint_url="http://mock.url/v1"
    )
    ex = {"prompt": [{"role": "user", "content": "Q"}], "answer": "42"}
    
    tensor_dict, final_messages, reward = await wrapper.rollout(ex, "m", {}, False)

    assert reward == 1.0
    assert tensor_dict["domain_id"] == "math"
    assert "states" in tensor_dict and "rewards" in tensor_dict
    assert torch.all(tensor_dict["rewards"][-1] == 1.0)
    assert final_messages == ex["prompt"] + [{"role": "assistant", "content": "42"}]