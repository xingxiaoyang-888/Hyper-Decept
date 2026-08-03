# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import os
import logging
from time import sleep

from camel.models import BaseModelBackend, ModelFactory
from camel.types import ModelPlatformType
from openai import OpenAI

thread_log = logging.getLogger(name="inference.thread")
thread_log.setLevel("DEBUG")


def get_env_variable(var_name):
    """Fetch environment variable or return None if not set."""
    return os.getenv(var_name)

class SharedMemory:
    Message_ID = 0
    Message = None
    Agent_ID = 0
    Response = None
    Busy = False
    Working = False
    Done = False


class InferenceThread:

    def __init__(
        self,
        model_path: str = "models/Meta-Llama-3-8B-Instruct",  # noqa
        server_url: str = "http://10.140.0.144:8000/v1",
        stop_tokens: list[str] = None,
        model_platform_type: ModelPlatformType = ModelPlatformType.VLLM,
        model_type: str = "llama-3",
        temperature: float = 0.5,
        shared_memory: SharedMemory = None,
        max_tokens: int | None = None,
    ):
        self.alive = True
        self.count = 0
        self.server_url = server_url
        # print('model_type in InferenceThread:', model_type)
        self.model_type = model_type

        # print('server_url:', server_url)
        # print("self.model_type:", self.model_type)
        if model_path in ["deepinfra", "api"]:
            model_platform_type = ModelPlatformType.OPENAI_COMPATIBILITY_MODEL
            deepinfra_api_key = get_env_variable('DEEPINFRA_API_KEY')
            deepinfra_base_url = get_env_variable('BASE_URL_DEEPINFRA')
            self.model_backend: BaseModelBackend = ModelFactory.create(
                model_platform=model_platform_type,
                model_type=self.model_type,
                model_config_dict={
                    "temperature": temperature,
                },
                url=deepinfra_base_url,
                api_key=deepinfra_api_key,
            )
        elif model_path == "openai":
            model_platform_type = ModelPlatformType.OPENAI_COMPATIBILITY_MODEL
            is_local = "localhost" in server_url or "127.0.0.1" in server_url
            is_deepseek = (
                "deepseek" in str(self.model_type).lower()
                or "api.deepseek.com" in server_url.lower()
            )
            api_key = (
                get_env_variable("DEEPSEEK_API_KEY") if is_deepseek else None
            ) or (
                get_env_variable("OPENAI_API_KEY")
                or get_env_variable("ZHIPUAI_API_KEY")
                or ("ollama" if is_local else None)
            )
            if not api_key:
                raise ValueError(
                    "Set DEEPSEEK_API_KEY for DeepSeek or OPENAI_API_KEY for "
                    "a remote OpenAI-compatible endpoint"
                )
            model_config = {
                "temperature": temperature,
            }
            if max_tokens is not None:
                if max_tokens <= 0:
                    raise ValueError("max_tokens must be positive")
                model_config["max_tokens"] = int(max_tokens)
            if is_local:
                # Ollama thinking models otherwise spend most of the local
                # smoke-test runtime producing reasoning tokens.
                model_config.update({
                    "reasoning_effort": "none",
                    "max_tokens": int(max_tokens or 768),
                    "response_format": {"type": "json_object"},
                })
            self.model_backend: BaseModelBackend = ModelFactory.create(
                model_platform=model_platform_type,
                model_type=self.model_type,
                model_config_dict=model_config,
                url=server_url,
                api_key=api_key,
            )
            if is_local:
                # One local Ollama worker serves all agents serially. Avoid
                # multiplying a slow request with automatic retries.
                self.model_backend._client = OpenAI(
                    timeout=180,
                    max_retries=0,
                    base_url=server_url,
                    api_key=api_key,
                )
            elif hasattr(self.model_backend, "_client"):
                # Disable hidden SDK retries: the simulation already records and
                # bounds failed activations, and retry amplification is budgeted.
                self.model_backend._client = OpenAI(
                    timeout=float(os.getenv("HYPERDECEPT_API_TIMEOUT", "180")),
                    max_retries=0,
                    base_url=server_url,
                    api_key=api_key,
                )
        else:
            self.model_backend: BaseModelBackend = ModelFactory.create(
                model_platform=model_platform_type,
                model_type=self.model_type,
                model_config_dict={
                    "temperature": temperature,
                    "stop": stop_tokens
                },
                url="vllm",
                api_key=server_url,
                # because of CAMEL bugs here, will fix when CAMEL upgrade.
            )
        # print('self.model_backend._url:', self.model_backend._url)
        if shared_memory is None:
            self.shared_memory = SharedMemory()
        else:
            self.shared_memory = shared_memory

    def run(self):
        while self.alive:
            if self.shared_memory.Busy and not self.shared_memory.Working:
                self.shared_memory.Working = True
                try:
                    response = self.model_backend.run(
                        self.shared_memory.Message)
                    self.shared_memory.Response = response.choices[
                        0].message.content
                except Exception as e:
                    thread_log.error(f"Received response exception: {e}")
                    self.shared_memory.Response = "No response."
                self.shared_memory.Done = True
                self.count += 1
                thread_log.debug(
                    f"Thread {self.server_url}: {self.count} finished.")

            sleep(0.01)

if __name__ == "__main__":    
    # for name, value in ModelPlatformType.__members__.items():
    #     print(f"{name}: {value}")
    import os
    del os.environ["http_proxy"]
    del os.environ["HTTP_PROXY"]    
    inference_thread = InferenceThread(
        model_type="meta-llama/Llama-3.3-70B-Instruct-Turbo",  # Customize as needed
        model_platform_type=ModelPlatformType.OPENAI_COMPATIBILITY_MODEL,  # Customize if needed
        temperature=0.7
    )
    # Call the run method to start inference 
    messages = [
        {"role": "user", "content": "hello world!"}
    ]
    print(inference_thread.model_backend.run(messages))
