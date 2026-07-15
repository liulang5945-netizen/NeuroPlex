"""
原生态极 — 训练流式 API

原生训练通过 ModelSelfTrainer 进行，支持暂停/停止/SSE 进度。
"""
import asyncio
import json as _json
import logging
import os
import threading
import queue

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from taiji.core.app_state import app_state
from taiji.core.config import TrainingConfig
from taiji.core.utils import get_external_path

from .common import safe_put

logger = logging.getLogger("ApiServer.Training")
router = APIRouter()


class TrainStartRequest(BaseModel):
    datasets: list[str] = []
    dataset: str = ""


@router.post("/api/training/start")
async def start_training_stream(req: TrainStartRequest):
    """启动原生训练（SSE 流式进度）。"""
    if app_state.is_training:
        raise HTTPException(status_code=400, detail="训练正在进行中")

    if req.datasets:
        dataset_files = [get_external_path(os.path.join("data", dp)) for dp in req.datasets]
    elif req.dataset:
        dataset_files = [get_external_path(os.path.join("data", req.dataset))]
    else:
        raise HTTPException(status_code=400, detail="未指定训练数据集")

    valid_files = [f for f in dataset_files if os.path.exists(f)]
    if not valid_files:
        raise HTTPException(status_code=400, detail="未找到有效的训练数据集")

    app_state.is_training = True
    app_state.stop_training_requested = False
    app_state.pause_training_requested = False

    q = queue.Queue()

    def _train():
        try:
            safe_put(q, {"type": "status", "status": "starting", "message": "正在准备原生训练..."})

            from taiji.train.data_loader import InstructionDataset, create_dataloader
            from taiji.train.trainer import ModelSelfTrainer
            from taiji.tokenizer import ModelSelfTokenizer

            tokenizer = app_state.tokenizer or ModelSelfTokenizer()
            model = app_state.model

            if model is None:
                safe_put(q, {"type": "error", "message": "模型未加载"})
                return

            config = TrainingConfig()
            config.max_length = 512
            config.batch_size = 2
            config.num_epochs = 3
            config.learning_rate = 1e-4
            config.gradient_accumulation_steps = 4
            device = config.resolve_device()

            dataset = InstructionDataset(valid_files, tokenizer, max_length=config.max_length, pre_tokenize=True)
            dataloader = create_dataloader(dataset, batch_size=config.batch_size, shuffle=True)

            trainer = ModelSelfTrainer(
                model, tokenizer,
                learning_rate=config.learning_rate,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
            )

            app_state._trainer_ref = trainer

            save_dir = os.path.join(get_external_path("checkpoints"), "training")
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
                    safe_put(q, {"type": "stopped", "message": "训练已停止"})
                    break

                while app_state.pause_training_requested and not app_state.stop_training_requested:
                    trainer.is_paused = True
                    safe_put(q, {"type": "paused"})
                    import time
                    time.sleep(0.5)

                trainer.is_paused = False

                safe_put(q, {
                    "type": "progress",
                    "fraction": round(fraction, 3),
                    "description": desc,
                    "loss": loss_history[-1] if loss_history else None,
                    "meta": meta,
                })

            if not app_state.stop_training_requested:
                from taiji.loader import save_model
                final_dir = os.path.join(save_dir, "final")
                os.makedirs(final_dir, exist_ok=True)
                save_model(model, tokenizer, final_dir)
                safe_put(q, {"type": "complete", "message": f"训练完成，模型已保存至 {final_dir}"})
        except Exception as exc:
            logger.error(f"训练异常: {exc}")
            safe_put(q, {"type": "error", "message": str(exc)})
        finally:
            app_state.is_training = False
            app_state._trainer_ref = None

    threading.Thread(target=_train, daemon=True).start()

    async def _stream():
        alive = True
        while alive:
            try:
                item = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, lambda: q.get(timeout=0.1)),
                    timeout=0.5,
                )
                yield f"data: {_json.dumps(item, ensure_ascii=False)}\n\n"
                if item.get("type") in {"complete", "error", "stopped"}:
                    alive = False
            except asyncio.TimeoutError:
                yield f"data: {_json.dumps({'type': 'heartbeat'})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.post("/api/training/stop")
def stop_training():
    if not app_state.is_training:
        raise HTTPException(status_code=400, detail="训练未在进行中")
    app_state.stop_training_requested = True
    return {"status": "ok", "message": "正在停止训练..."}


@router.post("/api/training/pause")
def pause_training():
    if not app_state.is_training:
        raise HTTPException(status_code=400, detail="训练未在进行中")
    app_state.pause_training_requested = True
    return {"status": "ok", "message": "训练已暂停"}


@router.post("/api/training/resume")
def resume_training():
    if not app_state.is_training:
        raise HTTPException(status_code=400, detail="训练未在进行中")
    app_state.pause_training_requested = False
    return {"status": "ok", "message": "训练已恢复"}


@router.get("/api/training/status")
def get_training_status():
    return {
        "status": "training" if app_state.is_training else "idle",
        "paused": app_state.pause_training_requested,
    }
