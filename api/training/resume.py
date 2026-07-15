"""
原生态极 — 断点续训 API（精简版）

支持从检查点加载模型并继续训练。
"""
import asyncio
import json as _json
import logging
import os
import threading
import queue

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from taiji.core.app_state import app_state
from taiji.core.config import TrainingConfig
from taiji.core.utils import get_external_path

from .common import safe_put

logger = logging.getLogger("ApiServer.Training")
router = APIRouter()


@router.post("/api/training/resume_checkpoint")
async def resume_from_checkpoint():
    """从检查点恢复原生训练（SSE 流式进度）。"""
    if app_state.is_training:
        raise HTTPException(status_code=400, detail="训练正在进行中")

    app_state.is_training = True
    app_state.stop_training_requested = False
    app_state.pause_training_requested = False

    q = queue.Queue()

    def _train():
        try:
            safe_put(q, {"type": "status", "status": "loading_checkpoint", "message": "正在加载检查点..."})

            from taiji.train.data_loader import InstructionDataset
            from taiji.train.trainer import ModelSelfTrainer
            from taiji.tokenizer import ModelSelfTokenizer
            from taiji.loader import load_model

            ckpt_dir = get_external_path("checkpoints")
            model, tokenizer = load_model(ckpt_dir)
            if model is None:
                safe_put(q, {"type": "error", "message": "检查点加载失败"})
                return

            config = TrainingConfig()
            config.max_length = 512
            config.batch_size = 2
            config.num_epochs = 3
            config.learning_rate = 1e-4
            config.gradient_accumulation_steps = 4
            device = config.resolve_device()

            dataset_files = [os.path.join(get_external_path("data"), f) for f in os.listdir(
                get_external_path("data")) if f.endswith((".json", ".jsonl", ".txt"))]
            dataset = InstructionDataset(dataset_files, tokenizer, max_length=config.max_length)

            trainer = ModelSelfTrainer(
                model, tokenizer,
                learning_rate=config.learning_rate,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
            )

            app_state._trainer_ref = trainer
            save_dir = os.path.join(ckpt_dir, "training")
            os.makedirs(save_dir, exist_ok=True)

            for fraction, desc, loss_history, meta in trainer.pretrain(
                dataset,
                num_epochs=config.num_epochs,
                batch_size=config.batch_size,
                save_dir=save_dir,
                device=device,
            ):
                if app_state.stop_training_requested:
                    trainer.is_stopped = True
                    break
                safe_put(q, {"type": "progress", "fraction": round(fraction, 3), "description": desc})
            safe_put(q, {"type": "complete", "message": "训练完成"})
        except Exception as exc:
            safe_put(q, {"type": "error", "message": str(exc)})
        finally:
            app_state.is_training = False

    threading.Thread(target=_train, daemon=True).start()

    async def _stream():
        while True:
            try:
                item = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, lambda: q.get(timeout=0.1)),
                    timeout=0.5,
                )
                yield f"data: {_json.dumps(item, ensure_ascii=False)}\n\n"
                if item.get("type") in {"complete", "error"}:
                    break
            except asyncio.TimeoutError:
                yield f"data: {_json.dumps({'type': 'heartbeat'})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.get("/api/training/checkpoints")
def list_checkpoints():
    """列出检查点"""
    ckpt_dir = get_external_path("checkpoints")
    checkpoints = []
    if os.path.isdir(ckpt_dir):
        for name in sorted(os.listdir(ckpt_dir)):
            full = os.path.join(ckpt_dir, name)
            if os.path.isdir(full):
                config_path = os.path.join(full, "config.json")
                checkpoints.append({
                    "name": name,
                    "has_model": os.path.exists(config_path),
                })
    return {"checkpoints": checkpoints}
