from __future__ import annotations

import os
from time import sleep
from typing import Any

import httpx
import runpod
from pydantic import BaseModel, PrivateAttr
from runpod.error import QueryError

from services.models import VLLM_API_KEY, GpuPodConfig


class GpuOperator(BaseModel):
    config: GpuPodConfig | None = None
    llm_url: str | None = None
    _pod_id: str | None = PrivateAttr(default=None)
    _config: GpuPodConfig = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self._config = self.config or GpuPodConfig()
        runpod_api_key = os.environ.get("RUNPOD_API_KEY")

        if not runpod_api_key:
            raise ValueError("RUNPOD_API_KEY environment variable is not set")

        runpod.api_key = runpod_api_key

    def _ready_urls(self) -> list[str]:
        """Probe both liveness (/health) and model-loaded (/v1/models)."""
        if not self.llm_url:
            raise RuntimeError("LLM URL not set")
        base = self.llm_url.removesuffix("/v1").rstrip("/")
        return [f"{base}/health", f"{base}/v1/models"]

    def _is_ready(self) -> tuple[bool, str]:
        headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
        last = "no response"
        for url in self._ready_urls():
            try:
                response = httpx.get(url, headers=headers, timeout=15)
                last = f"{url} -> {response.status_code}"
                if response.status_code == 200:
                    return True, last
            except httpx.HTTPError as e:
                last = f"{url} -> {type(e).__name__}: {e}"
        return False, last

    def _wait_until_ready(self) -> None:
        deadline = self._config.ready_timeout_seconds
        elapsed = 0.0
        urls = ", ".join(self._ready_urls())
        print(f"Waiting for vLLM ready ({urls}), timeout={deadline}s ...")
        detail = "not probed yet"

        while elapsed < deadline:
            ok, detail = self._is_ready()
            if ok:
                print(f"vLLM ready after {elapsed:.0f}s ({detail})")
                return
            if int(elapsed) % 30 == 0:
                print(f"  still waiting {elapsed:.0f}s — {detail}")
            sleep(self._config.ready_poll_interval_seconds)
            elapsed += self._config.ready_poll_interval_seconds

        raise TimeoutError(
            f"vLLM not ready after {deadline}s (last: {detail})"
        )

    def _cleanup_pod(self) -> None:
        if not self._pod_id:
            return
        pod_id = self._pod_id
        try:
            runpod.terminate_pod(pod_id)
            print(f"Terminated pod {pod_id}")
        except QueryError as e:
            print(f"Failed to terminate pod {pod_id}: {e.message}")
        finally:
            self._pod_id = None
            self.llm_url = None

    def create_pod(self) -> "GpuOperator":
        gpu_ids = sorted(
            gpu["id"]
            for gpu in runpod.get_gpus()
            if self._config.min_memory_gb
            <= gpu["memoryInGb"]
            <= self._config.max_memory_gb
        )
        if not gpu_ids:
            raise RuntimeError(
                "No GPUs in range "
                f"{self._config.min_memory_gb}-{self._config.max_memory_gb} GB"
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
                self._pod_id = created_pod["id"]
                self.llm_url = (
                    f"https://{self._pod_id}-{self._config.vllm_port}"
                    f".proxy.runpod.net/v1"
                )
                print(f"Created pod {self._pod_id} on {gpu_id}")
                print(f"Waiting for vLLM at {self.llm_url} ...")
                self._wait_until_ready()
                return self
            except QueryError as e:
                print(f"{gpu_id}: {e.message}")
                errors.append(f"{gpu_id}: {e.message}")
                self._cleanup_pod()
            except TimeoutError as e:
                print(f"{gpu_id}: {e}")
                errors.append(f"{gpu_id}: {e}")
                self._cleanup_pod()
            except Exception as e:
                print(f"{gpu_id}: unexpected {type(e).__name__}: {e}")
                errors.append(f"{gpu_id}: {e}")
                self._cleanup_pod()

        raise RuntimeError(
            "Failed to create pod on any GPU in range "
            f"({self._config.min_memory_gb}-{self._config.max_memory_gb} GB). "
            + "; ".join(errors)
        )

    def terminate_pod(self) -> bool:
        self._cleanup_pod()
        return False

    def __enter__(self) -> "GpuOperator":
        return self.create_pod()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return self.terminate_pod()
