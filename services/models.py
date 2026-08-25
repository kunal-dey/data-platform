from pydantic import BaseModel

VLLM_API_KEY = "my_secure_langgraph_secret_123"
VLLM_PORT = 8000

class GpuPodConfig(BaseModel):
    name: str = "qwen-pod"
    image_name: str = "vllm/vllm-openai:latest"
    container_disk_in_gb: int = 15
    volume_in_gb: int = 5
    docker_args: str = (
        "Qwen/Qwen3-8B-AWQ --quantization awq --host 0.0.0.0 --port 8000 "
        "--gpu-memory-utilization 0.85 --max-model-len 16384 --max-num-seqs 64 "
        "--enable-chunked-prefill --trust-remote-code "
        "--enable-auto-tool-choice --tool-call-parser hermes "
        f"--api-key {VLLM_API_KEY}"
    )
    min_memory_gb: int = 16
    max_memory_gb: int = 24
    retry_delay_seconds: float = 2
    vllm_port: int = VLLM_PORT
    ready_timeout_seconds: int = 600
    ready_poll_interval_seconds: float = 5