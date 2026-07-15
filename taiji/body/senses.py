"""
态极感知系统 (Senses)
====================

态极的感官 — 负责接收外部输入：API 请求、终端、前端。

态极原生实现，专门为态极服务。

注意：不再使用全局单例，由 BodyCore 统一管理。
"""
import asyncio
import logging
import time
from typing import Optional, Generator, AsyncGenerator

logger = logging.getLogger("Taiji.Senses")


class InputSensor:
    """
    态极的输入感知器

    接收外部请求，转化为态极可理解的输入格式。
    """

    def __init__(self):
        self._request_count = 0
        self._last_request_time = 0
        self._engine = None

    def set_engine(self, engine):
        """设置推理引擎"""
        self._engine = engine
        logger.info(f"推理引擎已设置: {type(engine).__name__}")

    def get_engine(self):
        """获取推理引擎"""
        return self._engine

    def has_engine(self) -> bool:
        """检查是否有推理引擎"""
        return self._engine is not None

    async def process_input(
        self,
        text: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict:
        """
        处理输入请求

        Args:
            text: 输入文本
            max_tokens: 最大生成 token 数
            temperature: 温度
            stream: 是否流式生成

        Returns:
            响应字典
        """
        self._request_count += 1
        self._last_request_time = time.time()

        if self._engine is None:
            return {"error": "推理引擎未设置", "status": "error"}

        try:
            if stream:
                return {
                    "status": "stream",
                    "generator": self._stream_generate(text, max_tokens, temperature),
                }
            else:
                result = await self._generate(text, max_tokens, temperature)
                return {"status": "ok", "content": result}
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return {"error": str(e), "status": "error"}

    async def _generate(self, text: str, max_tokens: int, temperature: float) -> str:
        """同步生成"""
        if hasattr(self._engine, 'generate'):
            return self._engine.generate(text, max_new_tokens=max_tokens, temperature=temperature)
        return "引擎不支持生成"

    async def _stream_generate(self, text: str, max_tokens: int, temperature: float) -> AsyncGenerator[str, None]:
        """流式生成"""
        if hasattr(self._engine, 'generate_stream'):
            async for chunk in self._engine.generate_stream(text, max_new_tokens=max_tokens, temperature=temperature):
                yield chunk
        else:
            yield "引擎不支持流式生成"

    def get_status(self) -> dict:
        """获取感知器状态"""
        return {
            "request_count": self._request_count,
            "last_request_time": self._last_request_time,
            "has_engine": self.has_engine(),
            "engine_type": type(self._engine).__name__ if self._engine else None,
        }


class TerminalSensor:
    """
    态极的终端感知器

    接收终端输入。
    """

    def __init__(self):
        self._input_count = 0

    def read_input(self, prompt: str = "> ") -> str:
        """
        读取终端输入

        Args:
            prompt: 提示符

        Returns:
            用户输入
        """
        self._input_count += 1
        try:
            return input(prompt)
        except EOFError:
            return ""

    def get_status(self) -> dict:
        """获取终端感知器状态"""
        return {
            "input_count": self._input_count,
        }