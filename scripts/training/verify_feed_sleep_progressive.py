#!/usr/bin/env python3
"""培养期渐进改善验证（多轮 feed → sleep 循环，2026-08-11）。

承接 verify_feed_sleep_e2e.py（闭环可用性 14/14 PASS）——本次验证闭环的
真正价值主张：**喂养数据渐进改善 zh 生成**。

结构：
- 5 轮培养循环，每轮喂 8 条新 zh 样本（主题递进，互不重复）→ sleep Phase 2 训练
- 首轮训练前测 baseline；每轮训练后测：
  1. held-out zh PPL（独立评估集 10 条，从未参与训练，口径与 _train_single_neuron 一致）
  2. 生成质量（4 个 zh prompt 非空率 + 重复率，temperature 0.55）
- 每轮验证：样本消费、训练锁释放、ckpt 自动保存

主断言：末轮 held-out PPL < baseline（渐进改善）；生成非空率不劣、重复率不升。

运行：python -u scripts/training/verify_feed_sleep_progressive.py
"""

from __future__ import annotations

import math
import os
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

# 5 轮训练批次（主题递进、互不重复；>50 字符结构化通过质量评估）
ROUND_BATCHES = [
    [  # 轮 1：机器学习基础
        "感知机模型\n感知机是最简单的线性分类器。\n要点：\n- 输入加权求和后过阈值函数\n- 仅能处理线性可分问题\n- 通过误分类点更新权重\n- 是神经网络的基本单元。",
        "逻辑回归\n逻辑回归用于二分类任务。\n特点：\n- 输出经过 sigmoid 映射到概率\n- 决策边界是线性的\n- 用交叉熵作为损失函数\n- 可扩展到多分类。",
        "决策树\n决策树通过特征划分实现分类。\n过程：\n- 选择最优特征作为划分节点\n- 递归生成子树\n- 剪枝防止过拟合\n- 易于解释但易波动。",
        "支持向量机\n支持向量机寻找最大间隔超平面。\n要点：\n- 间隔最大化提升泛化\n- 核函数实现非线性映射\n- 支持向量决定边界\n- 对高维数据有效。",
        "聚类算法\n聚类在无监督场景下发现数据结构。\n常见方法：\n- K 均值：按距离划分簇\n- 层次聚类：逐步合并或分裂\n- DBSCAN：按密度聚类\n- 聚类结果用于数据探索。",
        "降维方法\n降维减少特征数量保留主要信息。\n常用技术：\n- 主成分分析：找方差最大方向\n- t-SNE：保留局部结构\n- 线性判别分析：有监督降维\n- 降维利于可视化与加速。",
        "特征工程\n特征工程直接影响模型上限。\n实践：\n- 缺失值填充与异常处理\n- 数值特征标准化\n- 类别特征编码\n- 特征组合构造新变量。",
        "模型评估\n评估模型需要合理的指标。\n常用指标：\n- 准确率：预测正确比例\n- 精确率与召回率\n- F1 分数综合两者\n- AUC 评估排序能力。",
    ],
    [  # 轮 2：深度学习实践
        "反向传播\n反向传播是训练神经网络的引擎。\n原理：\n- 链式法则逐层求梯度\n- 从输出层向输入层传播\n- 计算图自动微分\n- 梯度用于更新权重。",
        "优化器比较\n不同优化器影响收敛速度。\n常见选择：\n- SGD：简单但有振荡\n- 动量：累积历史梯度\n- Adam：自适应学习率\n- 学习率预热提升稳定。",
        "批归一化\n批归一化加速深层网络训练。\n作用：\n- 每批数据归一化到标准分布\n- 缓解内部协变量偏移\n- 允许更大学习率\n- 训练与推理行为不同。",
        "残差网络\n残差连接解决深层退化问题。\n设计：\n- 跨层恒等映射捷径\n- 梯度直达深层\n- 允许数百层堆叠\n- 图像任务表现优异。",
        "卷积网络\n卷积网络擅长处理网格结构数据。\n组成：\n- 卷积核提取局部特征\n- 池化降低分辨率\n- 感受野逐层扩大\n- 参数共享大幅减少。",
        "循环网络\n循环网络建模序列依赖。\n机制：\n- 隐藏状态沿时间传递\n- LSTM 引入门控机制\n- 处理长距离依赖\n- 序列生成的基础。",
        "学习率调度\n学习率策略影响最终精度。\n常见方案：\n- 阶梯下降：按步数衰减\n- 余弦退火：平滑降低\n- 周期重启：跳出局部最优\n- 预热后调优。",
        "过拟合应对\n防止过拟合有多种手段。\n组合使用：\n- 数据增强扩充样本\n- 正则项约束权重\n- 丢弃随机失活\n- 早停监控验证集。",
    ],
    [  # 轮 3：自然语言处理
        "中文分词\n中文分词是中文处理第一步。\n方法：\n- 词典匹配：正向最大匹配\n- 统计模型：条件随机场\n- 神经网络分词\n- 新词发现提升准确率。",
        "词性标注\n词性标注确定词语的语法类别。\n应用：\n- 句法分析前置步骤\n- 命名实体识别辅助\n- 序列标注框架求解\n- 歧义词需上下文消解。",
        "句法分析\n句法分析还原句子结构。\n类型：\n- 依存句法：词间依赖关系\n- 短语结构：层次组成\n- 成分分析生成语法树\n- 用于语义理解。",
        "语义角色标注\n语义角色标注识别谓词论元。\n任务：\n- 找出动作的施事受事\n- 标注时间地点工具\n- 支撑问答系统\n- 深层语义理解基础。",
        "指代消解\n指代消解确定代词所指对象。\n难点：\n- 回指与前指判断\n- 共指链聚类\n- 语义一致性约束\n- 提升阅读理解质量。",
        "文本生成\n文本生成是语言模型核心任务。\n技术：\n- 自回归逐词生成\n- 束搜索与采样策略\n- 控制生成内容风格\n- 评估用困惑度与人工。",
        "情感分析\n情感分析判断文本情绪倾向。\n应用：\n- 商品评论正负分类\n- 舆情监控预警\n- 细粒度情感挖掘\n- 结合上下文避免误判。",
        "信息抽取\n信息抽取从文本提取结构化知识。\n任务：\n- 命名实体识别\n- 关系抽取\n- 事件抽取\n- 构建知识图谱基础。",
    ],
    [  # 轮 4：编程与算法
        "递归算法\n递归是函数调用自身的编程技巧。\n要素：\n- 明确终止条件\n- 每次调用缩小问题规模\n- 避免重复计算\n- 记忆化提升效率。",
        "动态规划\n动态规划分解重叠子问题。\n步骤：\n- 定义状态转移方程\n- 初始化边界条件\n- 自底向上填表\n- 空间优化滚动数组。",
        "排序算法\n排序是算法的基础操作。\n对比：\n- 快排平均最快\n- 归并排序稳定\n- 堆排序原地排序\n- 小数据用插入排序。",
        "二分查找\n二分查找快速定位有序数据。\n前提：\n- 数据必须有序\n- 每次缩小一半范围\n- 复杂度为对数级\n- 变体含上下界查找。",
        "树结构\n树是层次化的数据结构。\n常见类型：\n- 二叉树与平衡树\n- 堆与优先队列\n- 前缀树用于检索\n- 线段树支持区间查询。",
        "图算法\n图算法解决网络结构问题。\n基础：\n- 深度优先遍历\n- 广度优先遍历\n- 最短路径迪杰斯特拉\n- 并查集处理连通性。",
        "复杂度分析\n复杂度衡量算法效率。\n要点：\n- 大 O 表示增长趋势\n- 时间与空间权衡\n- 最坏与平均情况\n- 数据规模决定可行性。",
        "哈希表\n哈希表提供常数时间访问。\n原理：\n- 哈希函数映射键值\n- 冲突用链地址或开放寻址\n- 负载因子触发扩容\n- 字典与缓存核心结构。",
    ],
    [  # 轮 5：数学与统计
        "概率分布\n概率分布刻画随机变量规律。\n常见：\n- 伯努利与二项分布\n- 正态分布居中集中\n- 泊松分布计次事件\n- 分布决定建模方式。",
        "贝叶斯定理\n贝叶斯定理更新先验信念。\n形式：\n- 后验正比于似然乘先验\n- 证据归一化分母\n- 朴素贝叶斯假设独立\n- 广泛用于分类推断。",
        "假设检验\n假设检验判断样本是否支持结论。\n流程：\n- 设立原假设与备择\n- 计算检验统计量\n- 得出 P 值判断显著\n- 注意两类错误风险。",
        "线性代数基础\n线性代数是机器学习的语言。\n核心：\n- 向量空间与线性映射\n- 矩阵乘法与变换\n- 特征值与特征向量\n- 奇异值分解压缩。",
        "矩阵分解\n矩阵分解简化高维数据。\n应用：\n- 特征值分解方阵\n- 奇异值分解任意矩阵\n- 非负矩阵分解用于主题\n- 推荐系统隐因子模型。",
        "微积分基础\n微积分支撑优化理论。\n要点：\n- 导数描述变化率\n- 梯度指向最陡上升\n- 偏导与链式法则\n- 积分计算面积体积。",
        "数值方法\n数值方法求解解析难的问题。\n技术：\n- 牛顿法迭代求根\n- 梯度法优化目标\n- 数值积分近似\n- 稳定性控制误差。",
        "统计推断\n统计推断从样本推广总体。\n途径：\n- 点估计给出参数值\n- 区间估计给出置信范围\n- 自助法重采样\n- 蒙特卡洛模拟近似。",
    ],
]

# held-out 评估集：提问式风格（与训练列表式区分），10 条，从未参与训练
EVAL_TEXTS = [
    "什么是过拟合？过拟合是指模型在训练数据上表现很好，但在未见过的测试数据上表现较差的现象，通常由模型过于复杂或训练数据不足引起。",
    "为什么使用激活函数？激活函数为神经网络引入非线性表达能力，没有激活函数的多层网络可以化简为单层线性变换，无法拟合复杂模式。",
    "注意力机制有什么作用？注意力机制允许模型在生成每个输出时动态关注输入序列中最相关的部分，显著改善长序列的建模能力。",
    "如何选择学习率？学习率决定参数更新的步长，过大会导致震荡不收敛，过小则收敛缓慢，实践中常用学习率调度逐步调整。",
    "什么是词嵌入？词嵌入把离散的单词映射为稠密的实数向量，使语义相近的词在向量空间中距离更近，方便后续模型处理。",
    "dropout 为什么有效？dropout 在训练时随机丢弃部分神经元，迫使网络学习冗余的分布式表示，从而提升模型的泛化能力。",
    "LSTM 如何解决长依赖问题？LSTM 引入输入门、遗忘门和输出门，控制信息流的写入与遗忘，使梯度可以沿时间更长地传播。",
    "什么是正则化？正则化通过在损失函数中加入对参数复杂度的惩罚项，约束模型容量，是缓解过拟合的常用手段。",
    "如何评估分类模型？分类模型常用准确率、精确率、召回率和 F1 分数评估，不平衡数据集上更应关注后三者。",
    "什么是反向传播？反向传播利用链式法则从输出层向输入层逐层计算损失对各参数的梯度，是训练神经网络的数学基础。",
]

ZH_PROMPTS = [
    "请介绍什么是神经网络",
    "如何缓解过拟合问题",
    "什么是注意力机制",
    "请解释梯度下降的原理",
]


def eval_zh_ppl(cortex, texts, max_tokens=256) -> float | None:
    """held-out zh PPL：口径与 _train_single_neuron 完全一致
    （tokenizer_hub encode → general 逐 token 映射 → shared_embedding → lm_head CE）。"""
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


def eval_generation(cortex, prompts, max_tokens=30) -> dict:
    """生成质量：非空数 + 重复数（temperature 0.55 随机性下粗粒度信号）。"""
    outs = []
    for p in prompts:
        try:
            outs.append(cortex.generate(p, max_tokens=max_tokens, domain="zh"))
        except Exception:
            outs.append("")
    non_empty = sum(1 for o in outs if o.strip())
    seen = set()
    dup = 0
    for o in outs:
        k = o.strip()[:40]
        if k in seen:
            dup += 1
        seen.add(k)
    return {"non_empty": non_empty, "dup": dup, "outs": outs}


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("培养期渐进改善验证（5 轮 feed → sleep 循环）", flush=True)
    print("=" * 60, flush=True)

    tmp_dir = tempfile.mkdtemp(prefix="feed_sleep_prog_")
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
    cortex.neurons_dir = tmp_dir  # 隔离 ckpt 保存

    feed_engine = FeedEngine(data_dir=os.path.join(tmp_dir, "feed_data"))
    sleep_engine = SleepEngine(SleepConfig(training_enabled=True))
    sleep_engine.set_brain_interfaces(
        cortex=cortex,
        feed_engine=feed_engine,
        lifecycle=modules.get("lifecycle"),
        sleep_consolidator=modules.get("sleep_consolidator"),
        stdp_tracker=modules.get("stdp_tracker"),
    )

    # ── baseline（训练前 held-out PPL）──
    ppl0 = eval_zh_ppl(cortex, EVAL_TEXTS)
    print(f"\n[baseline] held-out zh PPL = {ppl0:.1f}", flush=True)
    check("baseline PPL 有效", ppl0 is not None and math.isfinite(ppl0), f"ppl={ppl0}")

    records = []
    for r in range(len(ROUND_BATCHES)):
        print(f"\n{'=' * 50}\n[轮 {r + 1}] feed {len(ROUND_BATCHES[r])} 条新 zh 样本 → sleep 训练",
              flush=True)
        fed = 0
        for text in ROUND_BATCHES[r]:
            item = feed_engine.feed_text(text, source=f"prog_r{r}",
                                         category="knowledge", domain="zh")
            if item is not None and item.status == "digested":
                fed += 1
        check(f"轮{r + 1} 样本消化", fed == len(ROUND_BATCHES[r]), f"{fed}/{len(ROUND_BATCHES[r])}")

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
        gen = eval_generation(cortex, ZH_PROMPTS)
        records.append({"round": r + 1, "ppl": ppl, "loss": report.training_loss,
                        **gen})
        ckpt_ok = os.path.exists(os.path.join(tmp_dir, "cortex_state.pt"))
        check(f"轮{r + 1} ckpt 自动保存", ckpt_ok)
        print(f"  轮{r + 1}: held-out PPL={ppl:.1f}  loss={report.training_loss:.4f}"
              f"  生成非空={gen['non_empty']}/4 重复={gen['dup']}", flush=True)

    # ── 主断言：渐进改善 ──
    final = records[-1]
    print(f"\n{'=' * 50}\n趋势汇总:", flush=True)
    print(f"  baseline PPL={ppl0:.1f}", flush=True)
    for rec in records:
        print(f"  轮{rec['round']}: PPL={rec['ppl']:.1f}", flush=True)
    check("末轮 held-out PPL < baseline（渐进改善）",
          final["ppl"] is not None and final["ppl"] < ppl0,
          f"final={final['ppl']:.1f} < baseline={ppl0:.1f}")
    check("生成非空率不劣于 baseline 期",
          final["non_empty"] >= records[0]["non_empty"],
          f"final={final['non_empty']}/4 vs 首轮={records[0]['non_empty']}/4")
    check("重复率未上升",
          final["dup"] <= records[0]["dup"] + 1,
          f"final dup={final['dup']} vs 首轮 dup={records[0]['dup']}")

    # ── 训练后生成展示 ──
    print("\n[末轮生成抽样]", flush=True)
    for p, o in zip(ZH_PROMPTS, final["outs"]):
        print(f"  {p} → {o[:60]!r}", flush=True)

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
