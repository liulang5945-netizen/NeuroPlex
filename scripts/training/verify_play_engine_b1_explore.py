#!/usr/bin/env python3
"""自举门槛 B1 探索自主性：play 引擎常态运行下它自己选主题方向（2026-08-20）。

背景：
    A5 完整已证 100 步 × 10 批固定 prompt 注入下 3 组 mean 上升 + 曲线自然饱和。
    但 A5 是"脚本固定每 10 步给同主题"——不是"它自己选方向"。

    B1 要回答：**play 引擎常态运行时，它能不能自己选下一个主题？**
    - 给 6 个主题池（每池 24 条，主题差异大：哲学/法律/医学/艺术/历史/工程）
    - 每 50 步决策一次：用 cortex.judge NLL 评估每池 1 条代表 prompt
    - **选 judge NLL 最高（短板）的池**——这是它"自己选方向"
    - 然后从该池 24 条全部注入 replay buffer
    - 1000 步 = 20 次决策

判据（B1）：
    B1.a new_experiences 中"它自己选的主题"次数 / 总决策次数 ≥ 30%
        （≥ 30% 表示：6 个池中它至少重复关注 1-2 个池方向，不是均匀选 6 个）
    B1.b 选中次数最高池 / 总决策次数 ≥ 15%（向某个方向集中）
    B1.c 0 崩溃 / 0 NaN
    B1.d 1000 步 ≤ 60 min（forward_replay 累积限制；预计 dt 60s/10 步 → 6000s = 100 min，会超！需限制
        micro-sleep 类型：前 800 步只 phase 1.5，后 200 步 phase 1.5+1.6，避免 forward_replay 爆时间）

约束：
    - 冻结 9 成员 production weights（不动 body）
    - 复用 A3 衰减 0.9 + SleepConsolidator
    - 1000 步预算 ≤ 60 min，必要时减到 500 步

运行：python -u scripts/training/verify_play_engine_b1_explore.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
from neuroplex.loader import assemble_cortex  # noqa: E402
from neuroplex.life.sleep_engine import SleepEngine, SleepConfig, SleepReport  # noqa: E402
from neuroplex.resonance.neuro_modulation import SleepConsolidator  # noqa: E402

from scripts.training.verify_a1_judge_signal_real import (  # noqa: E402
    DIALOGUE_IDS, COLLAB_NAME, EXTRA_NEURONS_DIR,
    DIALOGUE_PROMPTS, KNOWLEDGE_PROMPTS, UNFAMILIAR_PROMPTS,
)
from scripts.training.verify_a3_with_decay import (  # noqa: E402
    field_state_of, lora_l2_norm,
)
from scripts.training.verify_a4_post_sleep_judge_signal import (  # noqa: E402
    measure_group_stds,
)

passed = 0
failed = 0
N_MICRO = int(os.environ.get("B1_MICRO_N", "1000"))
DECISION_EVERY = int(os.environ.get("B1_DECISION_EVERY", "50"))
DECAY = float(os.environ.get("B1_DECAY", "0.9"))

TOPIC_POOLS = {
    "philosophy": [
        "什么是存在主义的核心主张？",
        "请解释康德的先验范畴理论。",
        "庄子齐物论的主要思想是什么？",
        "维特根斯坦的语言游戏理论。",
        "苏格拉底的诘问法如何运作？",
        "请解释黑格尔的辩证法。",
        "海德格尔此在（Dasein）概念。",
        "尼采的永恒轮回思想。",
        "罗尔斯正义论的差异原则。",
        "休谟的归纳问题。",
        "请解释柏拉图的理念论。",
        "福柯的规训社会理论。",
        "萨特自由与责任的关系。",
        "德里达解构主义方法。",
        "请解释斯宾诺莎的神即自然。",
        "卢梭公意与私意区分。",
        "霍布斯自然状态假说。",
        "请解释洛克的第一性质第二性质。",
        "边沁的功利主义原则。",
        "密尔对自由的理解。",
        "请解释哈贝马斯的公共领域。",
        "伽达默尔解释学循环。",
        "莱维纳斯他者哲学。",
        "请解释阿伦特积极生活。",
    ],
    "law": [
        "什么是普通法系与大陆法系的区别？",
        "请解释英美法系的陪审团制度。",
        "什么是正当程序原则？",
        "请解释合同法的对价制度。",
        "知识产权中的合理使用原则。",
        "请解释侵权法中的注意义务。",
        "什么是公司法中的信义义务？",
        "请解释国际公法中的主权豁免。",
        "什么是反垄断法的相关市场界定？",
        "请解释刑法的罪行法定原则。",
        "什么是无罪推定？",
        "请解释证据法中的传闻规则。",
        "什么是合理怀疑标准？",
        "请解释民事诉讼的举证责任。",
        "什么是国际私法中的冲突规范？",
        "请解释仲裁协议的法律效力。",
        "什么是宪法的基本权利条款？",
        "请解释司法审查权的来源。",
        "什么是刑法中的共同犯罪？",
        "请解释民法中的善意取得。",
        "什么是专利法中的新颖性？",
        "请解释商标法中的混淆可能性。",
        "什么是版权法中的思想表达二分？",
        "请解释国际人权法的域外适用。",
    ],
    "medicine": [
        "什么是药物的首过效应？",
        "请解释心肌梗死的病理机制。",
        "糖尿病酮症酸中毒如何处理？",
        "请解释抗生素后效应。",
        "什么是肺栓塞的 Wells 评分？",
        "请解释肾小球滤过率的临床意义。",
        "什么是 GCS 评分？",
        "请解释哮喘的阶梯治疗。",
        "什么是脓毒症 SIRS 标准？",
        "请解释青光眼的视野缺损。",
        "什么是胰岛素抵抗的机制？",
        "请解释胃食管反流病。",
        "什么是甲亢的甲状腺危象？",
        "请解释肝硬化的 Child 分级。",
        "什么是急性胰腺炎 Ranson 标准？",
        "请解释脑卒中的 TOAST 分型。",
        "什么是过敏性紫癜？",
        "请解释特发性肺纤维化。",
        "什么是抗磷脂综合征？",
        "请解释库欣综合征的诊断。",
        "什么是深静脉血栓的 Virchow 三联？",
        "请解释系统性红斑狼疮的诊断标准。",
        "什么是克罗恩病与溃疡性结肠炎的鉴别？",
        "请解释肾盂肾炎与膀胱炎的区别。",
    ],
    "art": [
        "什么是文艺复兴的人文主义？",
        "请解释巴洛克与洛可可的区别。",
        "印象派与后印象派的差异。",
        "请解释立体主义的多视角。",
        "什么是抽象表现主义？",
        "请解释超现实主义的自动书写。",
        "什么是达达主义的反艺术？",
        "请解释极简主义的美学主张。",
        "什么是观念艺术的核心？",
        "请解释波普艺术与消费文化。",
        "什么是装置艺术？",
        "请解释行为艺术的身体性。",
        "什么是大地艺术？",
        "请解释日本浮世绘对印象派的影响。",
        "什么是中国画的留白？",
        "请解释书法中的气韵。",
        "什么是古典音乐的奏鸣曲式？",
        "请解释十二音技法。",
        "什么是序列音乐？",
        "请解释爵士乐的即兴结构。",
        "什么是电影的长镜头理论？",
        "请解释蒙太奇的爱森斯坦理论。",
        "什么是意大利新现实主义？",
        "请解释新浪潮的作者论。",
    ],
    "history": [
        "什么是三十年战争？",
        "请解释凡尔赛体系的形成。",
        "什么是明治维新的改革内容？",
        "请解释俄国 1905 革命。",
        "什么是魏玛共和国的失败。",
        "请解释冷战铁幕的来源。",
        "什么是古巴导弹危机的过程？",
        "请解释马歇尔计划的动机。",
        "什么是越南战争的影响。",
        "请解释 1968 年全球抗议浪潮。",
        "什么是石油危机对世界经济的影响？",
        "请解释中国抗日战争的中流砥柱。",
        "什么是万隆会议的意义？",
        "请解释非洲独立浪潮。",
        "什么是拉丁美洲的依附理论？",
        "请解释苏联解体的内部原因。",
        "什么是东欧剧变的过程？",
        "请解释两德统一的历史背景。",
        "什么是马岛战争的影响？",
        "请解释海湾战争的国际法争议。",
        "什么是 9/11 事件后反恐格局？",
        "请解释 2008 金融危机的根源。",
        "什么是阿拉伯之春的连锁反应？",
        "请解释乌克兰危机的历史脉络。",
    ],
    "engineering": [
        "什么是有限元分析？",
        "请解释热力学第二定律。",
        "什么是 PID 控制器？",
        "请解释卡尔曼滤波的核心。",
        "什么是快速傅里叶变换？",
        "请解释香农采样定理。",
        "什么是 PID 与 LQR 的区别？",
        "请解释自动控制中的根轨迹。",
        "什么是材料力学中的胡克定律？",
        "请解释流体力学中的伯努利方程。",
        "什么是热传导的傅里叶定律？",
        "请解释质量传递的菲克定律。",
        "什么是相图中的共晶点？",
        "请解释金属的位错理论。",
        "什么是聚合物的玻璃化转变？",
        "请解释半导体 PN 结的工作原理。",
        "什么是 MOSFET 的阈值电压？",
        "请解释 CMOS 电路的功耗来源。",
        "什么是通信系统的误码率？",
        "请解释 OFDM 的多载波原理。",
        "什么是 MIMO 信道容量？",
        "请解释 CDMA 的扩频机制。",
        "什么是 TCP 的拥塞控制？",
        "请解释 RSA 加密算法的数学基础。",
    ],
}


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}", flush=True)
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}", flush=True)


def main():
    t0 = time.time()
    today = time.strftime("%Y%m%d")
    n_decisions = N_MICRO // DECISION_EVERY
    print("=" * 64, flush=True)
    print(f"自举门槛 B1 探索自主性：{N_MICRO} 次 micro-sleep + {n_decisions} 次自主选主题决策",
          flush=True)
    print(f"  6 个主题池 × 24 条 = 144 条候选 prompt（哲学/法律/医学/艺术/历史/工程）", flush=True)
    print("=" * 64, flush=True)

    print("\n[1/5] 装配 9 成员 production cortex（冻结，不写 checkpoint）...", flush=True)
    cortex, _tok, _mods = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name=COLLAB_NAME,
        extra_neurons_dir=EXTRA_NEURONS_DIR,
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS,
    )
    target_ids = [nid for nid in cortex.neurons
                  if nid.startswith("zh_") and "dialogue" in nid]
    print(f"  装配 {len(cortex.neurons)} 神经元，judge 目标 = {target_ids}", flush=True)

    tmp_data = os.path.join("data", "_tmp_b1")
    os.makedirs(tmp_data, exist_ok=True)
    cfg = SleepConfig(
        training_enabled=False,
        judge_driven_replay=True,
        lora_decay_per_sleep=DECAY,
    )
    sleep_engine = SleepEngine(config=cfg, data_dir=tmp_data)
    sc = SleepConsolidator(replay_buffer_size=400)
    sleep_engine.set_brain_interfaces(cortex=cortex, sleep_consolidator=sc)

    a1_groups = {
        "dialogue": DIALOGUE_PROMPTS,
        "knowledge": KNOWLEDGE_PROMPTS,
        "unfamiliar": UNFAMILIAR_PROMPTS,
    }
    print(f"\n[2/5] 注入 A1 真实版 24 条 + 6 个主题池 144 条记忆（初始 168 条）...", flush=True)
    for i, text in enumerate(DIALOGUE_PROMPTS + KNOWLEDGE_PROMPTS + UNFAMILIAR_PROMPTS):
        vec = field_state_of(cortex, text)
        sleep_engine.record_field_memory(vec, f"a1_{i}", text=text)
        sc.record_high_resonance_state(
            field_state=vec, resonance_score=0.9, step=0,
            active_nids=target_ids, threshold=0.5, text=text)
    for tname, prompts in TOPIC_POOLS.items():
        for i, text in enumerate(prompts):
            vec = field_state_of(cortex, text)
            sleep_engine.record_field_memory(vec, f"{tname}_{i}", text=text)
            sc.record_high_resonance_state(
                field_state=vec, resonance_score=0.85, step=0,
                active_nids=target_ids, threshold=0.5, text=text)
    r_init = SleepReport(timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), duration_seconds=0)
    sleep_engine._sleep_phase_field_consolidation(r_init)
    print(f"  注入 168 条 + 场固化 {r_init.field_memories_consolidated} 条", flush=True)

    print(f"\n[3/5] 跑 {N_MICRO} 次 micro-sleep（每 {DECISION_EVERY} 步一次自主选主题）...",
          flush=True)
    device = next(cortex._shared_embedding.parameters()).device
    decision_log = []
    selected_counts = {tname: 0 for tname in TOPIC_POOLS}
    n_crashes = 0
    n_exploration_random = 0
    checkpoint_curve = []

    for step in range(1, N_MICRO + 1):
        t_step = time.time()

        if step % DECISION_EVERY == 1:
            decision_idx = (step - 1) // DECISION_EVERY
            nll_per_pool = {}
            for tname, prompts in TOPIC_POOLS.items():
                sample_prompts = prompts[:2]
                nlls = []
                for text in sample_prompts:
                    jnll = sleep_engine._sample_judge_nll(
                        text, target_ids, device, cortex._shared_embedding)
                    if jnll is not None and jnll < 1e6:
                        nlls.append(jnll)
                nll_per_pool[tname] = float(np.mean(nlls)) if nlls else 0.0

            sorted_pools = sorted(nll_per_pool.items(), key=lambda x: -x[1])
            chosen_topic = sorted_pools[0][0]
            second_topic = sorted_pools[1][0] if len(sorted_pools) > 1 else None
            selected_counts[chosen_topic] += 1
            decision_log.append({
                "decision_idx": decision_idx,
                "step": step,
                "nll_per_pool": nll_per_pool,
                "chosen_topic": chosen_topic,
                "second_topic": second_topic,
                "top2_nll_gap": (sorted_pools[0][1] - sorted_pools[1][1]
                                 if len(sorted_pools) > 1 else 0.0),
            })

            for j, text in enumerate(TOPIC_POOLS[chosen_topic]):
                vec = field_state_of(cortex, text)
                sleep_engine.record_field_memory(
                    vec, f"b1_step{step}_{chosen_topic}_{j}", text=text)
                sc.record_high_resonance_state(
                    field_state=vec, resonance_score=0.9, step=step,
                    active_nids=target_ids, threshold=0.5, text=text)

            print(f"  decision {decision_idx:2d}  step {step:4d}  "
                  f"chosen={chosen_topic:12s}  nll={nll_per_pool[chosen_topic]:.4f}  "
                  f"2nd={second_topic}({nll_per_pool.get(second_topic, 0):.4f})",
                  flush=True)

        report = SleepReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            duration_seconds=0,
        )
        try:
            sleep_engine._sleep_phase_field_consolidation(report)
            if step % 100 == 0:
                sleep_engine._sleep_phase_synaptic_consolidation(report)
                sleep_engine._sleep_phase_forward_replay(report)
        except Exception as e:
            n_crashes += 1
            print(f"  [WARN] micro-sleep {step} 异常: {type(e).__name__}: {e}",
                  flush=True)
            if n_crashes > 5:
                check(f"micro-sleep {step} 不崩溃", False, f"crashes={n_crashes}")
                break
            continue
        dt_step = time.time() - t_step

        if step % DECISION_EVERY == 0 or step == 1:
            checkpoint_curve.append({
                "step": step,
                "elapsed_total_s": round(time.time() - t0, 1),
                "dt_step_s": round(dt_step, 1),
            })

    print(f"\n[4/5] 决策统计：{len(decision_log)} 次决策", flush=True)
    for tname in TOPIC_POOLS:
        cnt = selected_counts[tname]
        print(f"  {tname:12s}: 选中 {cnt:2d} 次 ({100*cnt/max(1,len(decision_log)):.1f}%)",
              flush=True)

    n_decisions_done = len(decision_log)
    top_topic = max(selected_counts.items(), key=lambda x: x[1])
    top_concentration = top_topic[1] / max(1, n_decisions_done)
    nontrivial_concentration = sum(1 for c in selected_counts.values() if c >= 2)
    nontrivial_ratio = nontrivial_concentration / len(TOPIC_POOLS)

    print(f"\n  top topic = {top_topic[0]} ({top_concentration*100:.1f}%)", flush=True)
    print(f"  ≥ 2 次选中的主题数 = {nontrivial_concentration} / {len(TOPIC_POOLS)} "
          f"({nontrivial_ratio*100:.1f}%)", flush=True)

    print(f"\n[5/5] 后测量 A1 真实版 3 组...", flush=True)
    post = measure_group_stds(sleep_engine, cortex, target_ids, a1_groups)
    for g in ("dialogue", "knowledge", "unfamiliar"):
        d = post[g]
        print(f"  post {g}: std={d['std']}  mean={d['mean']}", flush=True)

    print("\n" + "=" * 64, flush=True)
    print("B1 4 维判据：", flush=True)
    print("=" * 64, flush=True)

    b1a = top_concentration >= 0.30
    check(f"B1.a top 主题占比 ≥ 30% (它向某个方向集中)",
          b1a,
          f"{top_topic[0]}={top_concentration*100:.1f}%")

    b1b = top_concentration >= 0.15
    check(f"B1.b top 主题占比 ≥ 15% (向某个方向显著集中)",
          b1b,
          f"{top_topic[0]}={top_concentration*100:.1f}%")

    b1c = n_crashes == 0
    check(f"B1.c 0 崩溃 / 0 NaN",
          b1c,
          f"crashes={n_crashes}/{N_MICRO}")

    elapsed_min = (time.time() - t0) / 60
    b1d = elapsed_min <= 60
    check(f"B1.d 1000 步 ≤ 60 min",
          b1d,
          f"elapsed={elapsed_min:.1f} min")

    b1_pass = (failed == 0)
    if b1_pass:
        verdict = (f"B1 PASS：{n_decisions_done} 次决策 top 主题 "
                   f"{top_concentration*100:.1f}% ≥ 30%，"
                   f"它向 {top_topic[0]} 方向集中（≥ 15%），0 崩溃，"
                   f"耗时 {elapsed_min:.1f} min ≤ 60 min")
        next_step = ("B1 通过。下一步：B2 —— play 引擎常态运行下，"
                     "它能不能在不被喂新经验时仍维持 100 步无遗忘（autonomous 续航）。")
    else:
        if not b1a:
            verdict = (f"B1 半 PASS：top {top_concentration*100:.1f}% < 30% — "
                       f"6 个主题 NLL 太接近，决策信号弱")
            next_step = ("B1 失败：top 主题占比 < 30%。需重设主题池，"
                         "让 NLL 差异更大（哲学 NLL 17/法律 14/医学 16 等差异）"
                         "或加入【近期未选惩罚】让方向更集中。")
        else:
            verdict = f"B1 失败（{passed} PASS / {failed} FAIL）"
            next_step = "B1 失败：需重审其他判据"
    print(f"\n判定: {verdict}", flush=True)
    print(f"下一步: {next_step}", flush=True)

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", f"play_engine_b1_explore_{today}.json")
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": f"B1 探索自主性：{N_MICRO} 次 micro-sleep + {n_decisions} 次自主选主题",
        "cortex": {
            "n_neurons": len(cortex.neurons),
            "judge_target_ids": target_ids,
            "collab_name": COLLAB_NAME,
            "lora_decay_per_sleep": DECAY,
        },
        "config": {
            "n_micro": N_MICRO,
            "decision_every": DECISION_EVERY,
            "decay": DECAY,
            "n_decisions": n_decisions_done,
            "n_topic_pools": len(TOPIC_POOLS),
            "n_prompts_per_pool": 24,
        },
        "selected_counts": selected_counts,
        "top_topic": top_topic[0],
        "top_concentration": top_concentration,
        "nontrivial_concentration_count": nontrivial_concentration,
        "decision_log": decision_log,
        "checkpoint_curve": checkpoint_curve,
        "crash_count": n_crashes,
        "post_groups": {g: {k: v for k, v in post[g].items() if k != "nlls"} for g in a1_groups},
        "passed": passed,
        "failed": failed,
        "verdict": verdict,
        "next_step": next_step,
        "elapsed_seconds": time.time() - t0,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out_path}", flush=True)
    print(f"总耗时: {time.time() - t0:.1f}s", flush=True)
    sys.exit(0 if b1_pass else 1)


if __name__ == "__main__":
    main()
