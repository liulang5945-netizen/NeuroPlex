"""
鎬佹瀬 Autodl 棰勮缁冭剼鏈?v2
==========================
淇鏁版嵁鏍煎紡闂锛屼娇鐢ㄦ€佹瀬姝ｇ‘鐨?[绯荤粺]/[鐢ㄦ埛]/[鍔╂墜] 鏍煎紡

鐢ㄦ硶:
  python taiji/train/autodl_pretrain_v2.py --size 350m --data taiji_data/training_data/pretrain_final.jsonl
"""
import os
import sys
import json
import time
import math
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("AutodlPretrain")


def get_args():
    parser = argparse.ArgumentParser(description="鎬佹瀬 Autodl 棰勮缁?v2")
    parser.add_argument("--size", type=str, default="350m", choices=["125m", "350m", "1b", "3b", "7b"])
    parser.add_argument("--data", type=str, required=True, help="璁粌鏁版嵁璺緞")
    parser.add_argument("--output_dir", type=str, default="taiji_data/autodl_checkpoints")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--target_loss", type=float, default=1.5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    return parser.parse_args()


def format_conversation(messages):
    """
    灏?messages 鏍煎紡杞崲涓烘€佹瀬璁粌鏍煎紡

    杈撳叆: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]
    杈撳嚭: "[绯荤粺] ...\n[鐢ㄦ埛] ...\n[鍔╂墜] ..."
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            parts.append(f"[绯荤粺] {content}")
        elif role == "user":
            parts.append(f"[鐢ㄦ埛] {content}")
        elif role == "assistant":
            parts.append(f"[鍔╂墜] {content}")

    return "\n".join(parts)


def format_react(task, steps):
    """
    灏?ReAct 鏍煎紡杞崲涓烘€佹瀬璁粌鏍煎紡

    杈撳叆: {"task": "...", "steps": [{"thought": "...", "action": "...", ...}, ...]}
    杈撳嚭: "[绯荤粺] ...\n[鐢ㄦ埛] {task}\n[鍔╂墜] <think>{thought}</think><tool_call>{action} {args}\n<tool_result>{result}</tool_result>\n..."
    """
    parts = [
        "[绯荤粺] 浣犳槸鎬佹瀬AI鍔╂墜锛屽彲浠ヤ娇鐢ㄥ伐鍏峰畬鎴愪换鍔°€?,
        f"[鐢ㄦ埛] {task}",
        "[鍔╂墜] "
    ]

    for step in steps:
        thought = step.get("thought", "")
        action = step.get("action")
        action_args = step.get("action_args", {})
        final_answer = step.get("final_answer", "")
        observation = step.get("observation", "")

        if thought:
            parts[2] += f"<think>{thought}</think>"

        if action:
            args_str = json.dumps(action_args, ensure_ascii=False)
            parts[2] += f"<tool_call>{action} {args_str}"
            if observation:
                parts[2] += f"\n<tool_result>{observation}</tool_result>\n"

        if final_answer:
            parts[2] += f"<final_answer>{final_answer}"

    return "\n".join(parts)


class TaijiDataset(torch.utils.data.Dataset):
    """鎬佹瀬璁粌鏁版嵁闆?- 浣跨敤姝ｇ‘鐨勬牸寮?""

    def __init__(self, data_file, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        logger.info(f"鍔犺浇鏁版嵁: {data_file}")
        count = 0
        skipped = 0

        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)

                    # 澶勭悊 messages 鏍煎紡锛堝璇濓級
                    if "messages" in item:
                        text = format_conversation(item["messages"])

                    # 澶勭悊 ReAct 鏍煎紡锛堝伐鍏疯皟鐢級
                    elif "task" in item and "steps" in item:
                        text = format_react(item["task"], item["steps"])

                    # 澶勭悊绾枃鏈牸寮?                    elif "text" in item:
                        text = item["text"]

                    else:
                        skipped += 1
                        continue

                    # 杩囨护澶煭鐨勬牱鏈?                    if len(text) < 50:
                        skipped += 1
                        continue

                    self.samples.append(text)
                    count += 1

                except json.JSONDecodeError:
                    skipped += 1
                    continue

        logger.info(f"鍔犺浇瀹屾垚: {count} 鏉℃牱鏈? 璺宠繃 {skipped} 鏉℃棤鏁堟暟鎹?)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoded = self.tokenizer.encode(text)

        # 鎴柇鍒版渶澶ч暱搴?        if len(encoded) > self.max_len:
            encoded = encoded[:self.max_len]

        input_ids = torch.tensor(encoded, dtype=torch.long)
        return {"input_ids": input_ids, "labels": input_ids}


def collate_fn(batch, pad_id=0, max_len=512):
    """鍔ㄦ€乸adding鐨刢ollate鍑芥暟"""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_len_batch = min(max(len(ids) for ids in input_ids), max_len)

    padded_input = torch.full((len(batch), max_len_batch), pad_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len_batch), -100, dtype=torch.long)

    for i, (ids, labs) in enumerate(zip(input_ids, labels)):
        length = min(len(ids), max_len_batch)
        padded_input[i, :length] = ids[:length]
        padded_labels[i, :length] = labs[:length]

    return {"input_ids": padded_input, "labels": padded_labels}


def setup_model(args):
    """鍒濆鍖栨ā鍨?""
    from taiji.config import ModelConfig
    from taiji.architecture import ModelSelf

    preset_map = {
        "125m": ModelConfig.size_125m,
        "350m": ModelConfig.size_350m,
        "1b": ModelConfig.size_1b,
        "3b": ModelConfig.size_3b,
        "7b": ModelConfig.size_7b,
    }
    config = preset_map[args.size]()
    config.max_position_embeddings = args.max_seq_len

    if args.gradient_checkpointing:
        config.use_cache = False

    model = ModelSelf(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"妯″瀷鍙傛暟: {total_params/1e6:.1f}M (鍙缁? {trainable_params/1e6:.1f}M)")

    return model, config


def get_lr(step, max_steps, warmup_steps, max_lr):
    """Cosine warmup 瀛︿範鐜囪皟搴?""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


def train(args):
    """涓昏缁冨惊鐜?""
    # 璁剧疆璁惧
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"浣跨敤璁惧: {device}")

    # 鍒濆鍖?tokenizer
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()

    # 鏌ユ壘 SentencePiece 妯″瀷
    sp_candidates = [
        os.path.join("taiji", "tokenizer", "sentencepiece.model"),
        os.path.join("taiji_data", "final", "tokenizer", "sentencepiece.model"),
        os.path.join("model", "tokenizer", "sentencepiece.model"),
    ]

    sp_loaded = False
    for sp_path in sp_candidates:
        if os.path.exists(sp_path):
            sp.Load(sp_path)
            logger.info(f"鍔犺浇 SentencePiece: {sp_path}")
            sp_loaded = True
            break

    if not sp_loaded:
        logger.error("鏈壘鍒?SentencePiece 妯″瀷鏂囦欢!")
        return

    class SimpleTokenizer:
        def __init__(self, sp_model):
            self.sp = sp_model
        def encode(self, text):
            return self.sp.EncodeAsIds(text)
        def decode(self, ids):
            return self.sp.DecodeIds(ids)
        @property
        def pad_id(self):
            return self.sp.pad_id() if self.sp.pad_id() >= 0 else 0

    tokenizer = SimpleTokenizer(sp)

    # 鍔犺浇鏁版嵁
    if not os.path.exists(args.data):
        logger.error(f"鏁版嵁鏂囦欢涓嶅瓨鍦? {args.data}")
        return

    dataset = TaijiDataset(args.data, tokenizer, args.max_seq_len)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=0, max_len=args.max_seq_len),
        num_workers=4,
        pin_memory=True,
    )

    # 鍒濆鍖栨ā鍨?    model, config = setup_model(args)

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("姊害妫€鏌ョ偣宸插惎鐢?)

    model = model.to(device)

    # 浼樺寲鍣?    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    # 鎭㈠璁粌
    start_step = 0
    best_loss = float("inf")
    no_improve = 0

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt.get("step", 0)
            best_loss = ckpt.get("best_loss", float("inf"))
        else:
            model.load_state_dict(ckpt)
        logger.info(f"浠?checkpoint 鎭㈠: step={start_step}, loss={best_loss:.4f}")

    # 璁粌寰幆
    model.train()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = start_step
    epoch = 0
    running_loss = 0.0
    log_loss = 0.0

    logger.info("=" * 60)
    logger.info(f"寮€濮嬭缁?| 妯″瀷: {args.size} | 姝ユ暟: {args.max_steps}")
    logger.info(f"鏁版嵁: {args.data} | 鏍锋湰鏁? {len(dataset)}")
    logger.info(f"Batch: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    logger.info(f"LR: {args.lr} | SeqLen: {args.max_seq_len}")
    logger.info("=" * 60)

    while global_step < args.max_steps:
        epoch += 1
        for batch in dataloader:
            if global_step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(tokens=input_ids, targets=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

            loss = loss / args.grad_accum
            loss.backward()

            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # 鏇存柊瀛︿範鐜?            lr = get_lr(global_step, args.max_steps, args.warmup_steps, args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            loss_val = loss.item() * args.grad_accum
            running_loss += loss_val
            log_loss += loss_val
            global_step += 1

            # 鏃ュ織
            if global_step % args.log_every == 0:
                avg_loss = log_loss / args.log_every
                logger.info(
                    f"Step {global_step}/{args.max_steps} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                    f"Best: {best_loss:.4f}"
                )
                log_loss = 0.0

            # 淇濆瓨 checkpoint
            if global_step % args.save_every == 0:
                save_path = output_dir / f"checkpoint-{global_step}.pt"
                save_obj = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                    "best_loss": best_loss,
                    "config": config.__dict__,
                }
                torch.save(save_obj, save_path)
                logger.info(f"Checkpoint 宸蹭繚瀛? {save_path}")

            # 璇勪及鍜屾棭鍋?            if global_step % args.eval_every == 0:
                avg_eval_loss = running_loss / args.eval_every
                running_loss = 0.0

                if avg_eval_loss < best_loss - 0.01:
                    best_loss = avg_eval_loss
                    no_improve = 0
                    best_path = output_dir / "best.pt"
                    save_obj = {
                        "model": model.state_dict(),
                        "config": config.__dict__,
                    }
                    torch.save(save_obj, best_path)
                    logger.info(f"鏂扮殑鏈€浣虫ā鍨? Loss: {best_loss:.4f}")
                else:
                    no_improve += 1
                    logger.info(f"鏈敼鍠?({no_improve}/{args.patience})")

                if avg_eval_loss <= args.target_loss:
                    logger.info(f"杈惧埌鐩爣 loss {args.target_loss}, 璁粌瀹屾垚!")
                    break

                if no_improve >= args.patience:
                    logger.info(f"杩炵画 {args.patience} 娆℃湭鏀瑰杽, 鏃╁仠!")
                    break

    # 淇濆瓨鏈€缁堟ā鍨?    final_path = output_dir / "final.pt"
    save_obj = {
        "model": model.state_dict(),
        "config": config.__dict__,
    }
    torch.save(save_obj, final_path)
    logger.info(f"鏈€缁堟ā鍨嬪凡淇濆瓨: {final_path}")
    logger.info(f"璁粌瀹屾垚! 鎬绘鏁? {global_step}, 鏈€浣矻oss: {best_loss:.4f}")


if __name__ == "__main__":
    args = get_args()
    train(args)

