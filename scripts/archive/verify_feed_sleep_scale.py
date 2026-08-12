#!/usr/bin/env python3
"""培养期平台期定性实验（2026-08-11）。

承接 verify_feed_sleep_progressive（5 轮×8 条，PPL 384 平台/回升）——
定性 384 平台是**容量饱和**（51M 学到头）还是**样本量/主题多样性不足**。

实验设计（单轮对照 × 大样本混合喂养）：
- 评估集扩大至 16 条（与训练同分布列表式，降 8 条小样本噪声）
- 训练 3 轮 × 24 条**混合主题**（72 条，涵盖 progressive 40 条 + 新 32 条）
- 判断：若末轮显著突破 progressive 384 平台 → 样本量/多样性不足是主因；
  若 ~384-450 停滞 → 容量饱和（培养期阶段性上限）

运行：python -u scripts/training/verify_feed_sleep_scale.py
"""

from __future__ import annotations

import math
import os
import random
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from taiji.loader import assemble_cortex  # noqa: E402
from taiji.life.feed_engine import FeedEngine  # noqa: E402
from taiji.life.sleep_engine import SleepConfig, SleepEngine, SleepReport  # noqa: E402

passed = 0
failed = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}", flush=True)
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}", flush=True)


DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"


def knowledge(topic: str, desc: str, points: list) -> str:
    """构造列表式知识段落（与训练/评估同分布）。"""
    lines = [f"{topic}\n{desc}"]
    lines += [f"- {p}" for p in points]
    return "\n".join(lines)


# 训练池：progressive 5 轮 40 条（机器学习/NLP/编程/数学）+ 新增 16 条（AI 前沿）
# = 56 条；3 轮 × 16 条 = 48 条（每轮混合主题，样本量/多样性均超 progressive）
NEW_TOPICS = [
    ("智能体系统", "智能体是能感知环境并自主行动的 AI 系统。",
     ["感知模块收集环境信息", "决策模块规划动作序列", "工具调用扩展能力边界", "多智能体协作分工"]),
    ("强化学习", "强化学习通过试错与环境交互学习策略。",
     ["智能体与环境持续互动", "奖励信号引导行为方向", "价值函数评估状态优劣", "探索利用平衡困境"]),
    ("多模态学习", "多模态学习融合多种信息形式。",
     ["图像文本语音联合建模", "跨模态对齐语义空间", "模态缺失容错处理", "统一编码器共享表征"]),
    ("生成式模型", "生成式模型学习数据分布并采样新样本。",
     ["自回归逐步生成序列", "扩散模型去噪重建", "变分自编码器隐空间", "生成质量评估指标"]),
    ("自监督学习", "自监督学习从数据自身构造监督信号。",
     ["掩码重建预测缺失", "对比学习拉近正样本", "旋转预测学习表征", "无需人工标注"]),
    ("图神经网络", "图神经网络建模图结构数据。",
     ["节点特征聚合邻居信息", "消息传递多层迭代", "图池化压缩结构", "推荐分子社交应用"]),
    ("知识蒸馏", "知识蒸馏把大模型能力迁移给小模型。",
     ["教师模型输出软标签", "温度缩放软化分布", "学生模型匹配教师", "部署效率大幅提升"]),
    ("模型压缩", "模型压缩减小推理开销。",
     ["权重量化降低精度", "剪枝移除冗余参数", "低秩分解近似权重", "稀疏化加速计算"]),
    ("分布式训练", "分布式训练加速大规模模型训练。",
     ["数据并行切分样本", "模型并行拆分层", "梯度同步与异步", "通信开销优化"]),
    ("提示工程", "提示工程引导大模型完成任务。",
     ["指令描述清晰任务", "示例演示输入输出", "思维链引导推理", "输出格式约束"]),
    ("检索增强生成", "检索增强生成结合外部知识库。",
     ["查询编码检索文档", "上下文注入生成", "减少幻觉提升准确", "知识实时更新"]),
    ("迁移学习", "迁移学习复用源域知识到目标域。",
     ["预训练加微调范式", "特征迁移共享表征", "领域自适应对齐", "少样本快速适应"]),
    ("集成学习", "集成学习组合多个模型提升性能。",
     ["袋装减少方差", "提升串行修正误差", "堆叠学习元模型", "投票融合决策"]),
    ("可解释性", "可解释性让模型决策透明可信。",
     ["特征重要性分析", "梯度归因定位输入", "注意力可视化解释", "规则提取近似逻辑"]),
    ("联邦学习", "联邦学习在保护隐私前提下协作训练。",
     ["数据不出本地", "模型参数中心聚合", "差分隐私加噪保护", "非独立同分布挑战"]),
    ("边缘计算", "边缘计算把推理下沉到设备端。",
     ["低延迟本地响应", "带宽占用减少", "离线可用能力", "模型轻量化部署"]),
]
TRAIN_POOL = []
for t, d, pts in NEW_TOPICS:
    TRAIN_POOL.append(knowledge(t, d, pts))
# 并入 progressive 的 5 轮 40 条（机器学习/NLP/编程/数学主题）
from scripts.training.verify_feed_sleep_progressive import ROUND_BATCHES  # noqa: E402
for batch in ROUND_BATCHES:
    TRAIN_POOL.extend(batch)
TRAIN_POOL = list(dict.fromkeys(TRAIN_POOL))  # 去重（同 FeedEngine dedup 语义）
print(f"  训练池: {len(TRAIN_POOL)} 条（progressive 40 + 新增 {len(NEW_TOPICS)}）", flush=True)

# 评估集：16 条 held-out（与训练同分布列表式，内容/主题不重复）
EVAL_TOPICS = [
    ("语音识别", "语音识别把声音转为文本。",
     ["声学特征提取", "声学模型建模发音", "语言模型约束文本", "端到端联合优化"]),
    ("机器翻译", "机器翻译实现跨语言文本转换。",
     ["编码器理解源语言", "解码器生成目标语言", "注意力对齐词对", "低资源语言挑战"]),
    ("问答系统", "问答系统根据问题给出答案。",
     ["阅读理解抽取证据", "知识库检索事实", "生成式回答开放问题", "多跳推理复杂问题"]),
    ("知识图谱", "知识图谱以图结构组织知识。",
     ["实体与关系三元组", "图存储与查询", "链接预测补全缺失", "语义推理辅助问答"]),
    ("文本摘要", "文本摘要压缩长文保留要点。",
     ["抽取式选择原句", "生成式重写内容", "长度可控输出", "摘要忠实度评估"]),
    ("图像分类", "图像分类识别图片所属类别。",
     ["卷积特征提取", "全局池化汇总", "分类头输出概率", "数据增强提升泛化"]),
    ("目标检测", "目标检测定位并分类图中物体。",
     ["锚框候选区域", "回归框坐标", "类别置信度评分", "非极大值抑制去重"]),
    ("语音合成", "语音合成把文本转为自然语音。",
     ["文本前端分析", "声学模型预测特征", "声码器重建波形", "韵律控制表达"]),
    ("异常检测", "异常检测识别数据中的离群模式。",
     ["统计分布建模", "重构误差判异常", "隔离森林划分", "实时监控预警"]),
    ("时间序列预测", "时间序列预测推断未来趋势。",
     ["自回归历史窗口", "季节趋势分解", "长短期依赖建模", "多步滚动预测"]),
    ("推荐排序", "推荐排序决定物品展示顺序。",
     ["多路召回候选", "精排模型打分", "多样性与相关性权衡", "线上实时反馈"]),
    ("数据清洗", "数据清洗提升数据质量。",
     ["缺失值处理策略", "重复记录去重", "异常值检测修正", "格式标准化统一"]),
    ("特征选择", "特征选择筛除无关冗余特征。",
     ["过滤式统计评分", "包裹式模型评估", "嵌入式正则筛选", "降维协同工作"]),
    ("超参数调优", "超参数调优搜索最优训练配置。",
     ["网格搜索穷举", "随机搜索高效采样", "贝叶斯优化代理模型", "早停加速评估"]),
    ("模型评估", "模型评估全面衡量能力表现。",
     ["交叉验证稳定估计", "混淆矩阵细粒度分析", "校准度与置信度", "线上 A/B 验证"]),
    ("数据增强", "数据增强扩充训练样本多样性。",
     ["图像几何变换", "文本同义改写", "噪声注入鲁棒性", "对抗样本生成"]),
]
EVAL_TEXTS = [knowledge(t, d, pts) for t, d, pts in EVAL_TOPICS]
print(f"  评估集: {len(EVAL_TEXTS)} 条 held-out（混合主题）", flush=True)


def eval_zh_ppl(cortex, texts, max_tokens=256) -> float | None:
    """与 _train_single_neuron 同口径：tokenizer_hub → general 映射 → lm_head CE。"""
    neuron = cortex.neurons.get("zh")
    if neuron is None:
        return None
    tokenizer_hub = getattr(cortex, "_tokenizer_hub", None)
    general_sp = getattr(cortex, "_general_sp", None)
    shared_embedding = getattr(cortex, "_shared_embedding", None)
    if tokenizer_hub is None or general_sp is None or shared_embedding is None:
        return None
    device = next(neuron.parameters()).device
    domain_sp = tokenizer_hub.get_tokenizer("zh")
    neuron.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            domain_ids = tokenizer_hub.encode(text, domain="zh")
            if not domain_ids or len(domain_ids) < 3:
                continue
            domain_ids = domain_ids[:max_tokens]
            target_ids = torch.tensor([domain_ids], dtype=torch.long, device=device)
            general_ids = []
            for did in domain_ids:
                piece = domain_sp.id_to_piece(did)
                gen_ids = general_sp.EncodeAsIds(piece)
                general_ids.append(gen_ids[0] if gen_ids else 0)
            if len(general_ids) < 3:
                continue
            input_ids = torch.tensor([general_ids], dtype=torch.long, device=device)
            embeddings = shared_embedding(input_ids)
            result = neuron.forward(embeddings, field_state=None, round_num=1,
                                    return_logits=True)
            logits = result["logits"]
            min_len = logits.size(1) - 1
            if min_len < 1:
                continue
            shift_logits = logits[:, :min_len, :].contiguous()
            shift_targets = target_ids[:, 1:1 + min_len].contiguous()
            vocab_size = logits.size(-1)
            shift_targets = shift_targets.clamp(0, vocab_size - 1)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_targets.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += min_len
    if total_tokens == 0:
        return None
    return math.exp(min(total_loss / total_tokens, 20))


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("培养期平台期定性实验（3 轮 × 16 条混合主题）", flush=True)
    print("=" * 60, flush=True)

    tmp_dir = tempfile.mkdtemp(prefix="feed_sleep_scale_")
    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name=COLLAB_NAME,
        extra_neurons_dir=EXTRA_NEURONS_DIR,
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS,
    )
    print(f"  装配: {list(cortex.neurons.keys())}", flush=True)
    cortex.neurons_dir = tmp_dir

    feed_engine = FeedEngine(data_dir=os.path.join(tmp_dir, "feed_data"))
    sleep_engine = SleepEngine(SleepConfig(training_enabled=True))
    sleep_engine.set_brain_interfaces(
        cortex=cortex,
        feed_engine=feed_engine,
        lifecycle=modules.get("lifecycle"),
        sleep_consolidator=modules.get("sleep_consolidator"),
        stdp_tracker=modules.get("stdp_tracker"),
    )

    ppl0 = eval_zh_ppl(cortex, EVAL_TEXTS)
    print(f"\n[baseline] held-out zh PPL = {ppl0:.1f}", flush=True)
    check("baseline PPL 有效", ppl0 is not None and math.isfinite(ppl0), f"ppl={ppl0}")

    random.seed(7)
    used = set()
    records = []
    BATCH_SIZE = 16
    for r in range(3):
        # 每轮从池中抽 16 条（去重、混合主题）
        pool = [t for t in TRAIN_POOL if t not in used]
        random.shuffle(pool)
        batch = pool[:BATCH_SIZE]
        used.update(batch)
        print(f"\n[轮 {r + 1}] feed {len(batch)} 条（混合主题）→ sleep 训练", flush=True)
        fed = 0
        for text in batch:
            item = feed_engine.feed_text(text, source=f"scale_r{r}",
                                         category="knowledge", domain="zh")
            if item is not None and item.status == "digested":
                fed += 1
        check(f"轮{r + 1} 样本消化", fed == len(batch), f"{fed}/{len(batch)}")

        report = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                             duration_seconds=0)
        try:
            sleep_engine._sleep_phase_model_training(report)
        except Exception as e:
            import traceback
            traceback.print_exc()
            check(f"轮{r + 1} 训练无异常", False, f"{e}")
            return
        check(f"轮{r + 1} 样本被消费", report.training_samples_used > 0,
              f"used={report.training_samples_used}")
        from taiji.core.app_state import app_state
        check(f"轮{r + 1} 训练锁释放", not app_state.is_training)

        ppl = eval_zh_ppl(cortex, EVAL_TEXTS)
        records.append({"round": r + 1, "ppl": ppl, "loss": report.training_loss,
                        "n_fed": len(batch)})
        print(f"  轮{r + 1}: held-out PPL={ppl:.1f}  loss={report.training_loss:.4f}",
              flush=True)

    print(f"\n{'=' * 50}\n趋势汇总（progressive 参考平台：384）:", flush=True)
    print(f"  baseline PPL={ppl0:.1f}", flush=True)
    for rec in records:
        print(f"  轮{rec['round']}: PPL={rec['ppl']:.1f} (喂 {rec['n_fed']} 条)", flush=True)
    final = records[-1]
    check("末轮 < baseline（改善）", final["ppl"] < ppl0,
          f"final={final['ppl']:.1f} < baseline={ppl0:.1f}")
    if final["ppl"] < 380:
        print("  → 突破 384 平台：样本量/多样性不足是平台主因", flush=True)
    elif final["ppl"] <= 460:
        print("  → 平台区 380-460 徘徊：疑似容量饱和（51M 阶段性上限）", flush=True)
    else:
        print("  → 平台期判定见数据（无明确结论）", flush=True)

    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    print("\n" + "=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL  ({time.time() - t0:.1f}s)", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
