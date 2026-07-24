"""两阶段训练 —— 阶段二：协作训练。

阶段一已完成：10 个神经元各自训练好（PPL~7, argmax~73%）
阶段二目标：冻结 backbone，只训练协作层（field_write/field_read），
让 10 个强神经元学习如何通过共振场配合。

训练时：
- 所有神经元参与 forward_train（残差模式）
- 只更新协作层参数：field_write, field_read_layers, field_pool_query, field_read_gate
- 冻结：embed_adapter, layers, norm, lm_head, shared_embedding
- 所有神经元看相同数据（共享核心），学习协作而非专精

用法：
    python -u scripts/training/train_collaboration.py --steps 4000
"""
import sys, os, torch, math, argparse, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.resonance import ResonanceNeuron, ResonanceField, ResonanceEnsemble
from taiji.resonance.translator import batch_align_and_embed
from scripts.training.train_neuron import (
    load_domain_tokenizer, load_general_tokenizer, load_or_create_shared_embedding,
    OUTPUT_DIR,
)
import torch.nn.functional as F

DATA_PATH = "data/distill/zh_texts.jsonl"


def load_shared_texts(data_path, max_texts=50000):
    """加载共享核心数据（所有神经元看相同数据学协作）。"""
    texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) >= 20:
                texts.append(line)
            if len(texts) >= max_texts:
                break
    print(f'  加载 {len(texts)} 条共享数据', flush=True)
    return texts


def freeze_backbone(neuron):
    """冻结 backbone，只保留协作层可训练。

    可训练：field_write, field_projector, field_pool_query, field_read_layers, field_read_gate
    冻结：embed_adapter, layers, norm, lm_head
    """
    # 冻结所有参数
    for param in neuron.parameters():
        param.requires_grad = False

    # 解冻协作层
    trainable = 0
    for name, param in neuron.named_parameters():
        if any(k in name for k in [
            'field_write', 'field_projector', 'field_pool_query',
            'field_read_layers', 'field_read_gate',
        ]):
            param.requires_grad = True
            trainable += param.numel()

    return trainable


def main():
    parser = argparse.ArgumentParser(description='阶段二：协作训练')
    parser.add_argument('--n_neurons', type=int, default=10)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3, help='协作层参数少，用更高 lr')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--fusion_mode', default='residual',
                        help='residual（族长+残差修正）或 soft（软加权）')
    parser.add_argument('--max_texts', type=int, default=50000)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=100)
    args = parser.parse_args()

    print('=' * 70, flush=True)
    print(f'两阶段训练 —— 阶段二：协作训练', flush=True)
    print(f'  {args.n_neurons} 个神经元, {args.steps} 步, fusion={args.fusion_mode}', flush=True)
    print(f'  lr={args.lr} (协作层参数少，用更高 lr)', flush=True)
    print('=' * 70, flush=True)

    # 1. 加载数据
    print(f'\n[1] 加载共享数据...', flush=True)
    texts = load_shared_texts(DATA_PATH, args.max_texts)

    # 2. 加载 tokenizers + embedding
    print(f'\n[2] 加载 tokenizers...', flush=True)
    domain_sp = load_domain_tokenizer('zh')
    general_sp = load_general_tokenizer()
    shared_emb = load_or_create_shared_embedding(args.device)
    shared_emb.requires_grad_(False)  # 冻结 embedding
    shared_emb.eval()

    # 3. 加载 10 个训练好的神经元
    print(f'\n[3] 加载 {args.n_neurons} 个训练好的神经元...', flush=True)
    neurons = {}
    total_trainable = 0
    for i in range(args.n_neurons):
        ckpt_path = os.path.join(OUTPUT_DIR, f'neuron_zh_j{i}.pt')
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        cfg = ckpt['neuron_config']
        cfg.dropout = 0.0  # 协作训练不需要 dropout（backbone 已冻结）
        neuron = ResonanceNeuron(cfg).to(args.device)
        neuron.load_state_dict(ckpt['state_dict'], strict=False)

        # 冻结 backbone，只保留协作层
        trainable = freeze_backbone(neuron)
        neuron.train()  # 协作层需要训练模式
        neurons[f'zh_j{i}'] = neuron
        print(f'  zh_j{i}: 协作层参数 {trainable/1e6:.2f}M', flush=True)
        total_trainable += trainable

    print(f'  总可训练参数: {total_trainable/1e6:.1f}M (backbone 已冻结)', flush=True)

    # 4. 创建 ensemble
    print(f'\n[4] 创建 ensemble...', flush=True)
    field_dim = next(iter(neurons.values())).config.field_dim
    field = ResonanceField(dim=field_dim)
    ensemble = ResonanceEnsemble(neurons, field, max_rounds=1)

    # 5. 优化器（只优化协作层参数）
    trainable_params = []
    for neuron in neurons.values():
        for param in neuron.parameters():
            if param.requires_grad:
                trainable_params.append(param)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # WSD 调度
    decay_start = max(args.warmup_steps + 1, int(args.steps * 0.8))
    def _wsd_lr(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        elif step < decay_start:
            return 1.0
        else:
            progress = (step - decay_start) / max(1, args.steps - decay_start)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _wsd_lr)

    # 6. 训练循环
    print(f'\n[5] 开始协作训练 ({args.steps} steps, fusion={args.fusion_mode})...', flush=True)
    n_texts = len(texts)

    def _sample_batch():
        idx = torch.randint(0, n_texts, (args.batch_size,))
        return [texts[int(i)] for i in idx]

    total_loss = 0.0
    total_balance = 0.0
    step, t_start = 0, time.time()
    best_loss = float('inf')
    best_step = 0
    best_states = None
    recent_losses = []

    for _ in range(args.steps):
        batch_texts = _sample_batch()
        shared_emb_batch, targets, mask = batch_align_and_embed(
            batch_texts, domain_sp, general_sp, shared_emb,
        )
        shared_emb_batch = shared_emb_batch.to(args.device)
        targets = targets.to(args.device)
        mask = mask.to(args.device)

        # 前向：所有神经元参与，残差模式
        result = ensemble.forward_train(
            shared_emb_batch, temperature=1.0, fusion_mode=args.fusion_mode,
        )
        fused_logits = result['fused_logits']
        balance_loss = result.get('balance_loss', torch.tensor(0.0))

        # CE loss
        shift_logits = fused_logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()
        shift_targets = shift_targets.clone()
        shift_targets[~shift_mask] = -100

        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            ignore_index=-100,
        )
        loss = ce_loss + 0.1 * balance_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += ce_loss.item()
        total_balance += balance_loss.item() if hasattr(balance_loss, 'item') else 0
        step += 1

        recent_losses.append(ce_loss.item())
        if len(recent_losses) > 100:
            recent_losses.pop(0)
        if len(recent_losses) >= 50:
            recent_avg = sum(recent_losses) / len(recent_losses)
            if recent_avg < best_loss:
                best_loss = recent_avg
                best_step = step
                best_states = {
                    nid: {k: v.detach().clone() for k, v in n.state_dict().items()}
                    for nid, n in neurons.items()
                }

        if step % args.log_every == 0:
            avg_loss = total_loss / step
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t_start
            current_lr = scheduler.get_last_lr()[0]
            print(
                f'  step {step}/{args.steps} '
                f'ce={ce_loss.item():.4f} avg={avg_loss:.4f} '
                f'PPL={ppl:.1f} balance={balance_loss.item():.4f} '
                f'lr={current_lr:.2e} best={best_loss:.4f}@{best_step} '
                f'elapsed={elapsed:.0f}s',
                flush=True,
            )

    avg_loss = total_loss / max(step, 1)
    ppl = math.exp(min(avg_loss, 20))
    elapsed = time.time() - t_start
    print(f'\n训练完成！{step} steps, avg_loss={avg_loss:.4f}, PPL={ppl:.1f}, '
          f'best={best_loss:.4f}@{best_step}, time={elapsed:.0f}s ({elapsed/60:.1f}min)', flush=True)

    # 7. 保存 best 状态
    print(f'\n[6] 保存 best 协作状态...', flush=True)
    if best_states is not None:
        for nid, state in best_states.items():
            ckpt_path = os.path.join(OUTPUT_DIR, f'neuron_{nid}.pt')
            # 加载原始 ckpt 获取 config
            orig_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            orig_ckpt['state_dict'] = state
            orig_ckpt['result']['collab_best_loss'] = best_loss
            orig_ckpt['result']['collab_best_step'] = best_step
            torch.save(orig_ckpt, ckpt_path)
        print(f'  已保存 best@step{best_step} (loss={best_loss:.4f})', flush=True)
    else:
        print(f'  ⚠️ 无 best state（步数太少），保存当前状态', flush=True)
        for nid, neuron in neurons.items():
            ckpt_path = os.path.join(OUTPUT_DIR, f'neuron_{nid}.pt')
            orig_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            orig_ckpt['state_dict'] = neuron.state_dict()
            torch.save(orig_ckpt, ckpt_path)

    print(f'\n✅ 阶段二完成，可运行 eval_collab.py 评估协作效果', flush=True)


if __name__ == '__main__':
    main()
