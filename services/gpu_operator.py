from pydantic import BaseModel, PrivateAttr
from typing import Any
import os
import runpod
from runpod.error import QueryError
from time import sleep
import httpx

from services.models import GpuPodConfig

class GpuOperator(BaseModel):
    llm_url: str | None = None
    _pod_id: str | None = PrivateAttr(default=None)
    _config: GpuPodConfig = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self._config = self._config or GpuPodConfig()
        runpod_api_key = os.environ.get("RUNPOD_API_KEY")

        if not runpod_api_key:
            raise ValueError("RUNPOD_API_KEY environment variable is not set")

        runpod.api_key = runpod_api_key

    def _wait_until_ready(self) -> None:
        if not self.llm_url:
            raise RuntimeError("LLM URL not set")

        health_url = self.llm_url.removesuffix("/v1") + "/health"
        deadline = self._config.ready_timeout_seconds
        elapsed = 0.0

        while elapsed < deadline:
            try:
                response = httpx.get(health_url, timeout=10)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            sleep(self._config.ready_poll_interval_seconds)
            elapsed += self._config.ready_poll_interval_seconds

        raise TimeoutError(f"vLLM not ready at {health_url} after {deadline}s")

    def create_pod(self) -> dict:

        gpu_ids = sorted(
            gpu["id"]
            for gpu in runpod.get_gpus()
            if self._config.min_memory_gb <= gpu["memoryInGb"] <= self._config.max_memory_gb
        )

        errors: list[str] = []
        for gpu_id in gpu_ids:
            sleep(self._config.retry_delay_seconds)
            try:
                created_pod = runpod.create_pod(
                    name=self._config.name,
                    image_name=self._config.image_name,
                    gpu_type_id=gpu_id,
                    container_disk_in_gb=self._config.container_disk_in_gb,
                    volume_in_gb=self._config.volume_in_gb,
                    docker_args=self._config.docker_args,
                    ports=f"{self._config.vllm_port}/http",
                )
                self._pod_id = created_pod['id']
                self.llm_url = (
                    f"https://{self._pod_id}-{self._config.vllm_port}.proxy.runpod.net/v1"
                )

                print(f"Waiting for vLLM at {self.llm_url} ...")
                self._wait_until_ready()

                return self
            except QueryError as e:
                print(f"{gpu_id}: {e.message}")
                errors.append(f"{gpu_id}: {e.message}")

        raise RuntimeError(
            "Failed to create pod on any GPU in range "
            f"({self._config.min_memory_gb}-{self._config.max_memory_gb} GB). "
            + "; ".join(errors)
        )

    def terminate_pod(self) -> None:
        if not self._pod_id:
            return False
        try:
            runpod.terminate_pod(self._pod_id)
        except QueryError as e:
            print(f"Failed to terminate pod {self._pod_id}: {e.message}")
        return False

    def __enter__(self) -> "GpuOperator":
        return self.create_pod()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return self.terminate_pod()
