"""
态极 WebSocket 服务器
====================

提供实时通信，让前端和态极核心直接连接。
态极的"神经系统" — 让态极能够主动与外界交流。
"""
import asyncio
import json
import logging
import websockets
from typing import Set, Optional, Callable
from taiji.core.api import get_core

logger = logging.getLogger("Taiji.WebSocket")


class TaijiWebSocketServer:
    """
    态极 WebSocket 服务器

    特点：
    - 前端和态极核心直接连接
    - 态极可以主动推送消息
    - 支持语音流式传输
    - 状态实时同步
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.core = None
        self.server = None

    async def start(self):
        """启动 WebSocket 服务器"""
        self.core = get_core()
        self.core.initialize()

        # 注册 EventBus 广播回调 — 生命事件实时推送到前端
        try:
            from taiji.infra.events import get_event_bus
            event_bus = get_event_bus()
            event_bus.set_broadcast_callback(self._on_life_event)
            logger.info("EventBus broadcast callback registered")
        except Exception as e:
            logger.warning(f"Failed to register EventBus broadcast: {e}")

        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
        )
        logger.info(f"态极 WebSocket 服务器启动: ws://{self.host}:{self.port}")
        await self.server.wait_closed()

    def _on_life_event(self, message: dict):
        """
        生命事件广播回调 — EventBus 发布事件时调用。

        将事件异步推送到所有连接的 WebSocket 客户端。
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 已在异步上下文中，创建任务
                asyncio.ensure_future(self.broadcast(message))
            else:
                loop.run_until_complete(self.broadcast(message))
        except RuntimeError:
            # 没有事件循环，跳过
            pass

    async def stop(self):
        """停止 WebSocket 服务器"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("态极 WebSocket 服务器已停止")

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol):
        """处理客户端连接"""
        self.clients.add(websocket)
        logger.info(f"新客户端连接: {websocket.remote_address}")

        try:
            # 发送欢迎消息
            await self.send_to_client(websocket, {
                "type": "welcome",
                "message": "你好，我是态极！",
                "status": self.core.get_life_status() if self.core else None,
            })

            # 监听客户端消息
            async for message in websocket:
                await self.handle_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端断开: {websocket.remote_address}")
        finally:
            self.clients.discard(websocket)

    async def handle_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """处理客户端消息"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "chat":
                await self.handle_chat(websocket, data)
            elif msg_type == "feed":
                await self.handle_feed(websocket, data)
            elif msg_type == "train":
                await self.handle_train(websocket, data)
            elif msg_type == "sleep":
                await self.handle_sleep(websocket, data)
            elif msg_type == "play":
                await self.handle_play(websocket, data)
            elif msg_type == "voice":
                await self.handle_voice(websocket, data)
            elif msg_type == "status":
                await self.handle_status(websocket, data)
            else:
                await self.send_to_client(websocket, {
                    "type": "error",
                    "message": f"未知消息类型: {msg_type}",
                })

        except json.JSONDecodeError:
            await self.send_to_client(websocket, {
                "type": "error",
                "message": "无效的 JSON 格式",
            })
        except Exception as e:
            logger.error(f"处理消息失败: {e}")
            await self.send_to_client(websocket, {
                "type": "error",
                "message": f"处理失败: {e}",
            })

    async def handle_chat(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        """处理聊天消息"""
        message = data.get("message", "")
        if not message:
            return

        # 发送思考状态
        await self.send_to_client(websocket, {
            "type": "thinking",
            "message": "思考中...",
        })

        try:
            # 调用态极核心
            response = self.core.chat(message)

            # 发送回复
            await self.send_to_client(websocket, {
                "type": "chat_response",
                "message": response,
            })

            # 语音播放回复
            try:
                self.core.speak_text(response)
            except Exception:
                pass

        except Exception as e:
            await self.send_to_client(websocket, {
                "type": "error",
                "message": f"聊天失败: {e}",
            })

    async def handle_feed(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        """处理喂养消息"""
        content = data.get("content", "")
        content_type = data.get("content_type", "text")

        try:
            result = self.core.feed(content, content_type)
            await self.send_to_client(websocket, {
                "type": "feed_response",
                "result": result,
            })
        except Exception as e:
            await self.send_to_client(websocket, {
                "type": "error",
                "message": f"喂养失败: {e}",
            })

    async def handle_train(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        """处理训练消息"""
        epochs = data.get("epochs", 3)
        learning_rate = data.get("learning_rate", 5e-5)

        # 发送训练开始状态
        await self.send_to_client(websocket, {
            "type": "training_start",
            "message": "开始训练...",
        })

        try:
            result = self.core.train(epochs, learning_rate)
            await self.send_to_client(websocket, {
                "type": "training_complete",
                "result": result,
            })
        except Exception as e:
            await self.send_to_client(websocket, {
                "type": "error",
                "message": f"训练失败: {e}",
            })

    async def handle_sleep(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        """处理睡眠消息"""
        try:
            result = self.core.sleep()
            await self.send_to_client(websocket, {
                "type": "sleep_response",
                "result": result,
            })
        except Exception as e:
            await self.send_to_client(websocket, {
                "type": "error",
                "message": f"睡眠失败: {e}",
            })

    async def handle_play(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        """处理玩耍消息"""
        try:
            result = self.core.play()
            await self.send_to_client(websocket, {
                "type": "play_response",
                "result": result,
            })
        except Exception as e:
            await self.send_to_client(websocket, {
                "type": "error",
                "message": f"玩耍失败: {e}",
            })

    async def handle_voice(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        """处理语音消息"""
        action = data.get("action")

        if action == "listen":
            try:
                text = self.core.listen_voice()
                await self.send_to_client(websocket, {
                    "type": "voice_text",
                    "text": text,
                })
            except Exception as e:
                await self.send_to_client(websocket, {
                    "type": "error",
                    "message": f"语音识别失败: {e}",
                })

        elif action == "speak":
            text = data.get("text", "")
            try:
                self.core.speak_text(text)
                await self.send_to_client(websocket, {
                    "type": "voice_spoken",
                    "text": text,
                })
            except Exception as e:
                await self.send_to_client(websocket, {
                    "type": "error",
                    "message": f"语音合成失败: {e}",
                })

    async def handle_status(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        """处理状态查询"""
        try:
            status = self.core.get_life_status()
            await self.send_to_client(websocket, {
                "type": "status_response",
                "status": status,
            })
        except Exception as e:
            await self.send_to_client(websocket, {
                "type": "error",
                "message": f"获取状态失败: {e}",
            })

    async def send_to_client(self, websocket: websockets.WebSocketServerProtocol, data: dict):
        """发送消息给客户端"""
        try:
            await websocket.send(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logger.error(f"发送消息失败: {e}")

    async def broadcast(self, data: dict):
        """广播消息给所有客户端"""
        for client in self.clients.copy():
            try:
                await self.send_to_client(client, data)
            except Exception:
                self.clients.discard(client)


# 全局服务器实例
_server: Optional[TaijiWebSocketServer] = None


async def start_server(host: str = "localhost", port: int = 8765):
    """启动态极 WebSocket 服务器"""
    global _server
    _server = TaijiWebSocketServer(host, port)
    await _server.start()


async def stop_server():
    """停止态极 WebSocket 服务器"""
    global _server
    if _server:
        await _server.stop()
        _server = None


def get_server() -> Optional[TaijiWebSocketServer]:
    """获取服务器实例"""
    return _server


if __name__ == "__main__":
    # 直接运行服务器
    asyncio.run(start_server())
