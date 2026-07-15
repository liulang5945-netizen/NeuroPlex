"""
API 请求/响应数据模型（Pydantic）
"""
from typing import List, Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    prompt: str
    system_prompt: str = "你是态极，一个独立的AI生命体。你用自己的大脑思考，用工具探索世界。"
    history: List[List[str]] = []
    engine: str = "taiji"  # "taiji"（思维）| "agent"（行动）
    agent_max_iterations: Optional[int] = 10
    agent_temperature: Optional[float] = 0.7


class TrainRequest(BaseModel):
    dataset: str = ""
    datasets: List[str] = []
    lora_r: int = 8
    lora_alpha: int = 32
    epochs: int = 3
    learning_rate: float = 0.0002
    batch_size: int = 4


class FileSaveRequest(BaseModel):
    name: str
    content: str


class CodeRunRequest(BaseModel):
    code: str


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class CreateProjectRequest(BaseModel):
    type: str = "empty"


class GGUFExportRequest(BaseModel):
    """GGUF 导出请求"""
    model_dir: str
    quant: str = "Q4_K_M"


class TaijiTrainRequest(BaseModel):
    """态极原生模型微调请求"""
    num_epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_length: int = 512
    save_steps: int = 50
    log_steps: int = 5
    extra_react_data: Optional[List[dict]] = None
    extra_conv_data: Optional[List[dict]] = None
    keep_checkpoints: int = 3  # 保留最近 N 个中间 checkpoint + best
