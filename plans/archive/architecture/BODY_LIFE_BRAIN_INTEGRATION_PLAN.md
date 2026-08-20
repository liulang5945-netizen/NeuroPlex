# 躯体与生命系统适配大脑架构计划 (Body-Life-Brain Integration Plan)

> **🔧 子系统设计参考**
> 本文档是 body/life/brain 三系统整合的设计参考。
> 大部分内容已实现（life_scheduler 调质映射、feed_engine 域分类、sleep_engine P7 训练、metabolism 神经调质），未实现部分（explore/play 按域反馈）为后续工作。
> 当前进度见 [`plans/active/BIO_INSPIRED_ARCHITECTURE_PLAN.md`](../../active/BIO_INSPIRED_ARCHITECTURE_PLAN.md)。

> 核心思想：**让躯体和生命系统更适配这个大脑**（神经元共振架构）。
> 当前问题：躯体/生命系统是为一代号单体 ModelSelf 设计的，与 Cortex + ResonanceEnsemble 脱节。

---

## 一、当前断点全景

### 1.1 身体系统（body/）

| 模块 | 当前行为 | 与大脑的断点 |
|------|---------|------------|
| BodyCore | 容器，管理 model/tokenizer/device | 不感知 Cortex，仍引用旧 ModelSelf |
| senses.py | 输入→推理引擎 | 输入未按域路由到对应神经元 |
| limbs.py | 工具执行 | 执行结果不反馈到学习 |
| metabolism.py | 硬件检测 | 不参与神经调质（dopamine/serotonin） |

### 1.2 生命系统（life/）

| 模块 | 当前行为 | 与大脑的断点 |
|------|---------|------------|
| life_scheduler | 5 维需求驱动行为 | ✅ 已实现：4/5 需求映射调质（stress→DA↓, curiosity→DA↑, boredom→5HT↓, fatigue→NE↓）；hunger→neurogenesis 已实现（Cortex.add_neuron 运行时闭环） |
| feed_engine | 4 源进食→samples | ✅ 已修复：samples 按域分类，含 "text" 字段 |
| sleep_engine Phase 2 | 训练 Cortex neurons | ✅ 已修复：P7 经验驱动学习（shared_embedding + lm_head 协同）|
| sleep_engine Phase 3.5 | 知识蒸馏 | 依赖 `taiji/agent_ext/` 不存在 |
| evolution_engine | 4 阶段成长 | 不触发 neurogenesis |
| explore_engine | 联网学习 | 结果不按域分类 |
| play_engine | 自由探索 | 不强化 CoactivationTracker |
| sleep_engine 调质 | ✅ 已实现：双信号驱动 NeuromodulatorState | 自主调控学习率，跨会话持久化 |
| metabolism → 调质 | ✅ 已实现：CPU 负载→NE，内存→DA 覆盖，资源健康→5HT 覆盖 | 三调质全接线完成 |

### 1.3 两套睡眠机制

| 机制 | 层级 | 状态 |
|------|------|------|
| life/sleep_engine.py | 生命系统级 | 6 阶段完整，但训练目标过时 |
| resonance/neuro_modulation.py | 神经元级 | 重放+修剪，但独立运行 |

**问题**：两套机制应协同，目前完全独立。

---

## 二、适配设计原则

### 2.1 大脑是唯一认知主体

- **Cortex + ResonanceEnsemble 是唯一的推理和学习主体**
- 躯体是大脑的"感官和四肢"，生命系统是大脑的"本能和节律"
- 所有学习最终都落到神经元参数更新（per-neuron lm_head / side_channels / neurogenesis / STDP）

### 2.2 知识按域路由

- 感知输入按 domain（zh/en/code/math/general）路由到对应神经元
- feed_engine 摄入的样本按域分类
- explore_engine 探索结果按域存储
- sleep_engine 训练时按域调用对应神经元进行独立 lm_head 训练

### 2.3 生命需求映射到神经元行为

| 生命需求 | 神经元级行为 |
|---------|------------|
| hunger（饥饿） | 触发 neurogenesis（创建新神经元填补知识盲区） |
| fatigue（疲劳） | 触发 SleepConsolidator（修剪弱连接） |
| boredom（无聊） | 触发 PlayEngine 自由共振（强化 CoactivationTracker） |
| stress（压力） | 提高不应期长度（强制休息） |
| curiosity（好奇） | 触发 ExploreEngine + 神经调质 dopamine↑（学习率↑） |

### 2.4 两套睡眠协同

- **life/sleep_engine 是外层调度**：决定何时睡、睡多久、睡哪个阶段
- **resonance/SleepConsolidator 是内层执行**：在 sleep_engine 的 Phase 2/3 中被调用
- sleep_engine Phase 2 训练目标改为：Cortex 神经元独立 lm_head + STDP + 可选 neurogenesis
- sleep_engine Phase 3 调用 SleepConsolidator 做突触修剪和 fingerprint 更新

---

## 三、具体适配方案

### 3.1 BodyCore 适配

**目标**：BodyCore 感知 Cortex，senses 按域路由。

**改动**：
```python
# body/core.py
class BodyCore:
    def __init__(self, ...):
        self.cortex: Optional[Cortex] = None  # 新增：引用大脑

    def set_cortex(self, cortex: Cortex) -> None:
        """注入大脑（取代旧的 set_model_self）。"""
        self.cortex = cortex
        # 感知系统路由到 Cortex
        if self.senses:
            self.senses.set_cortex(cortex)

    def is_healthy(self) -> bool:
        """健康检查：大脑必须加载。"""
        return self.cortex is not None and self.cortex.is_loaded
```

### 3.2 Senses 按域路由

**目标**：输入根据内容自动路由到对应域神经元。

**改动**：
```python
# body/senses.py
class InputSensor:
    def __init__(self, ...):
        self.cortex: Optional[Cortex] = None
        self.domain_detector: Optional[DomainDetector] = None

    def set_cortex(self, cortex: Cortex) -> None:
        self.cortex = cortex

    def _detect_domain(self, input_text: str) -> str:
        """检测输入所属域（zh/en/code/math/general）。"""
        # 简单规则：代码块→code，数学公式→math，英文为主→en，中文为主→zh，其他→general
        if self.domain_detector:
            return self.domain_detector.detect(input_text)
        # fallback 启发式
        if "```" in input_text or "def " in input_text:
            return "code"
        if any(c in input_text for c in ["∑", "∫", "\\frac", "=="]):
            return "math"
        # ...
        return "general"

    def process_input(self, input_text: str) -> dict:
        """处理输入，自动路由到对应域。"""
        domain = self._detect_domain(input_text)
        return {
            "domain": domain,
            "input": input_text,
            "cortex_response": self.cortex.think(input_text, domain_hint=domain),
        }
```

### 3.3 Limbs 执行结果反馈

**目标**：工具执行结果作为 feed_engine 的输入源。

**改动**：
```python
# body/limbs.py
class LimbSystem:
    def __init__(self, ...):
        self.feed_engine: Optional[FeedEngine] = None  # 新增引用

    def set_feed_engine(self, feed_engine) -> None:
        self.feed_engine = feed_engine

    def run_python(self, code: str) -> dict:
        result = self._execute_python(code)
        # 执行结果反馈到 feed_engine（作为"实践知识"）
        if self.feed_engine and result["success"]:
            self.feed_engine.feed_from_practice(
                code=code,
                output=result["output"],
                success=result["success"],
            )
        return result
```

### 3.4 Metabolism 驱动神经调质

**目标**：硬件状态影响神经调质（dopamine/serotonin/norepinephrine）。

**改动**：
```python
# body/metabolism.py
class MetabolismSystem:
    def __init__(self, ...):
        self.neuromodulator: Optional[NeuromodulatorState] = None

    def set_neuromodulator(self, nm: NeuromodulatorState) -> None:
        self.neuromodulator = nm

    def update_neuromodulator(self) -> None:
        """根据硬件状态更新神经调质。"""
        if not self.neuromodulator:
            return
        hw = self.analyze_hardware()
        # 低 RAM → stress ↑ → serotonin ↓（不满足）
        if hw.ram_available_gb < 4:
            self.neuromodulator.set_targets(serotonin=0.3)
        # GPU 可用 → norepinephrine ↑（警觉）
        if hw.has_gpu:
            self.neuromodulator.set_targets(norepinephrine=0.7)
        # CPU 模式 → norepinephrine ↓（放松）
        else:
            self.neuromodulator.set_targets(norepinephrine=0.4)
```

### 3.5 LifeScheduler 需求映射到神经元行为

**目标**：5 维需求驱动具体的神经元级操作。

**改动**：
```python
# life/life_scheduler.py
class LifeScheduler:
    def __init__(self, ...):
        self.cortex: Optional[Cortex] = None
        self.lifecycle: Optional[LifecycleManager] = None
        self.sleep_consolidator: Optional[SleepConsolidator] = None
        self.coactivation: Optional[CoactivationTracker] = None

    def set_brain_interfaces(
        self,
        cortex, lifecycle, sleep_consolidator, coactivation,
        neuromodulator,
    ) -> None:
        """注入大脑相关组件。"""
        self.cortex = cortex
        self.lifecycle = lifecycle
        self.sleep_consolidator = sleep_consolidator
        self.coactivation = coactivation
        self.neuromodulator = neuromodulator

    def _on_hunger_high(self, value: float) -> None:
        """饥饿→触发 neurogenesis。"""
        if value > 70 and self.lifecycle:
            # 检测知识盲区，触发新生
            # 由 feed_engine 提供 domain 错误率
            pass

    def _on_fatigue_high(self, value: float) -> None:
        """疲劳→触发 SleepConsolidator 修剪。"""
        if value > 80 and self.sleep_consolidator:
            # 强制睡眠巩固
            self.sleep_consolidator.consolidate(
                neurons=self.cortex.neurons,
                coactivation_tracker=self.coactivation,
            )

    def _on_boredom_high(self, value: float) -> None:
        """无聊→触发自由共振（强化 CoactivationTracker）。"""
        if value > 60 and self.cortex:
            # 让所有神经元自由共振一段时间
            self._trigger_free_resonance()

    def _on_stress_high(self, value: float) -> None:
        """压力→提高不应期（强制休息）。"""
        if value > 70:
            # 临时增加所有神经元的 refractory_cooldown
            for neuron in self.cortex.neurons.values():
                neuron.config.refractory_cooldown = 4  # 临时翻倍

    def _on_curiosity_high(self, value: float) -> None:
        """好奇→dopamine↑（学习率↑）。"""
        if value > 85 and self.neuromodulator:
            self.neuromodulator.set_targets(dopamine=0.8)
```

### 3.6 FeedEngine 按域分类样本

**目标**：摄入的样本按 domain 分类，供对应神经元训练。

**改动**：
```python
# life/feed_engine.py
class FeedEngine:
    def feed(self, source: str, content: str, domain: str = None) -> dict:
        """进食，自动检测或指定 domain。"""
        if domain is None:
            domain = self._detect_domain(content)
        sample = self._convert_to_sample(content, source, domain)
        # 按域存储
        self._domain_samples.setdefault(domain, []).append(sample)
        return {"domain": domain, "sample": sample}

    def get_pending_samples_by_domain(self) -> dict:
        """获取按域分类的待消化样本。"""
        return dict(self._domain_samples)

    def feed_from_practice(self, code: str, output: str, success: bool) -> None:
        """从实践（工具执行）中学习。"""
        domain = "code"
        content = f"代码:\n{code}\n\n输出:\n{output}\n\n成功: {success}"
        self.feed(source="practice", content=content, domain=domain)
```

### 3.7 SleepEngine 适配 Cortex（核心改动）

**目标**：Phase 2 训练目标从 ModelSelf 改为 Cortex 神经元。

**改动**：
```python
# life/sleep_engine.py
class SleepEngine:
    def __init__(self, ...):
        self.cortex: Optional[Cortex] = None  # 取代 model_self
        self.lifecycle: Optional[LifecycleManager] = None
        self.sleep_consolidator: Optional[SleepConsolidator] = None
        self.stdp_tracker: Optional[STDPTracker] = None
        self.coactivation: Optional[CoactivationTracker] = None

    def set_brain_interfaces(
        self,
        cortex, lifecycle, sleep_consolidator, stdp_tracker, coactivation,
    ) -> None:
        """注入大脑相关组件。"""
        self.cortex = cortex
        self.lifecycle = lifecycle
        self.sleep_consolidator = sleep_consolidator
        self.stdp_tracker = stdp_tracker
        self.coactivation = coactivation

    def _sleep_phase_model_training(self) -> dict:
        """Phase 2: 深睡训练（适配 Cortex）。"""
        if not self.cortex:
            return {"status": "skipped", "reason": "no cortex"}

        # 获取按域分类的待训练样本
        domain_samples = self.feed_engine.get_pending_samples_by_domain()

        results = {}
        for domain, samples in domain_samples.items():
            if domain not in self.cortex.neurons:
                continue
            neuron = self.cortex.neurons[domain]

            # 训练该神经元的独立 lm_head（P7: 域专用 vocab）
            result = self._train_neuron_lm_head(neuron, samples, domain)
            results[domain] = result

            # 记录 PPL，驱动凋亡判断
            if self.lifecycle:
                ppl = result.get("final_ppl", 999)
                self.lifecycle.apoptosis.record_ppl(domain, ppl)

        # 应用 STDP 更新（用睡眠中的重放数据）
        if self.stdp_tracker:
            self.stdp_tracker.apply_all_updates(self.cortex.neurons)

        # 检查是否需要 neurogenesis
        if self.lifecycle:
            self._check_neurogenesis_needs()

        return {"trained_domains": results}

    def _train_neuron_lm_head(self, neuron, samples, domain: str) -> dict:
        """P7: 训练单个神经元的独立 lm_head（域专用 vocab）。

        旧方案 (_train_neuron_delta): W_base 共享冻结 + 低秩残差 U_i@V_i 训练
        P7 方案: 每神经元独立完整 lm_head + 域 tokenizer（vocab=10k-20k）
        """
        # 域专用 vocab == neuron.lm_head 输出维度
        # 训练独立 lm_head 权重（不再有冻结的 W_base）
        params = [neuron.lm_head.weight]

        # 可选：也训练 embedding adapter
        if hasattr(neuron, 'embed_adapter'):
            params.append(neuron.embed_adapter.weight)

        optimizer = torch.optim.AdamW(params, lr=5e-5)

        total_loss = 0
        for sample in samples:
            with torch.enable_grad():
                result = neuron.forward(
                    sample["embeddings"],
                    field_state=None,
                    round_num=1,
                    return_logits=True,
                )
                logits = result["logits"]
                # domain_token targets (not general 256K)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    sample["targets"].view(-1),
                    ignore_index=-100,
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

        avg_loss = total_loss / max(len(samples), 1)
        return {"avg_loss": avg_loss, "final_ppl": math.exp(min(avg_loss, 20))}

    def _check_neurogenesis_needs(self) -> None:
        """检查是否需要创建新神经元。"""
        if not self.lifecycle or not self.cortex:
            return
        # 从 feed_engine 获取各 domain 错误率
        domain_errors = self.feed_engine.get_domain_error_rates()
        for domain, error_rate in domain_errors.items():
            if self.lifecycle.neurogenesis.record_domain_error(domain, error_rate):
                # 触发新生：从 teacher 蒸馏新神经元
                self._create_new_neuron(domain)

    def _create_new_neuron(self, domain: str) -> None:
        """创建新神经元（neurogenesis）。"""
        # 1. 从 teacher 蒸馏
        # 2. 注册到 lifecycle.maturity（幼稚态）
        # 3. 加入 cortex.neurons
        # 4. 注册到 column_registry（功能柱）
        pass  # 具体实现依赖 distill 流程

    def _sleep_phase_knowledge_integration(self) -> dict:
        """Phase 3: 知识整合（调用 SleepConsolidator）。"""
        if self.sleep_consolidator and self.cortex:
            return self.sleep_consolidator.consolidate(
                neurons=self.cortex.neurons,
                coactivation_tracker=self.coactivation,
                current_step=self._sleep_count,
            )
        return {"status": "skipped"}
```

### 3.8 EvolutionEngine 触发 Neurogenesis

**目标**：进化阶段转换时触发 neurogenesis。

**改动**：
```python
# life/evolution_engine.py
class EvolutionEngine:
    def __init__(self, ...):
        self.lifecycle: Optional[LifecycleManager] = None
        self.cortex: Optional[Cortex] = None

    def _on_stage_transition(self, old_stage: str, new_stage: str) -> None:
        """阶段转换时触发 neurogenesis。"""
        if not self.lifecycle or not self.cortex:
            return
        # infant→child: 创建初始神经元集合
        if old_stage == "infant" and new_stage == "child":
            for domain in ["zh", "en", "code", "math", "general"]:
                if domain not in self.cortex.neurons:
                    # 触发新生
                    pass
        # child→adolescent: 增加专业神经元
        elif old_stage == "child" and new_stage == "adolescent":
            # 根据使用频率增加专门神经元
            pass
```

### 3.9 PlayEngine 自由共振

**目标**：玩耍时让神经元自由共振，强化 CoactivationTracker。

**改动**：
```python
# life/play_engine.py
class PlayEngine:
    def __init__(self, ...):
        self.cortex: Optional[Cortex] = None
        self.coactivation: Optional[CoactivationTracker] = None

    def play_resonance(self, duration_rounds: int = 10) -> dict:
        """自由共振玩耍：让神经元自由组合共振。"""
        if not self.cortex:
            return {"status": "skipped"}

        # 生成随机/创造性输入
        creative_inputs = self._generate_creative_inputs()

        for i, inp in enumerate(creative_inputs[:duration_rounds]):
            # 跑一次共振
            result = self.cortex.ensemble.forward(inp, return_logits=False)
            active_ids = list(result.get("final_scores", {}).keys())

            # 记录到 CoactivationTracker
            if self.coactivation:
                # 转换为 int id
                active_int_ids = [hash(nid) % 100000 for nid in active_ids]
                self.coactivation.update(active_int_ids)

        return {"rounds_played": min(len(creative_inputs), duration_rounds)}
```

---

## 四、集成后的完整闭环

```
┌─────────────────────────────────────────────────────────────┐
│                     用户/环境                                │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Body: Senses（按域路由）                                    │
│  ├── 检测 domain (zh/en/code/math/general)                  │
│  └── 路由到 Cortex 对应神经元                                │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Brain: Cortex + ResonanceEnsemble                          │
│  ├── 多轮共振（含不应期、兴奋/抑制）                         │
│  ├── STDP 实时更新 side_channels                            │
│  └── 神经调质调节学习率/场写入强度                           │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Body: Limbs（执行）                                        │
│  ├── 工具调用                                               │
│  └── 执行结果 → feed_engine（实践知识）                      │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Life: LifeScheduler（需求驱动）                            │
│  ├── hunger↑ → neurogenesis 触发                            │
│  ├── fatigue↑ → SleepConsolidator 修剪                      │
│  ├── boredom↑ → PlayEngine 自由共振                         │
│  ├── stress↑ → refractory 加长                              │
│  └── curiosity↑ → dopamine↑ → 学习率↑                       │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Life: FeedEngine（知识摄入）                                │
│  ├── 4 源进食（对话/知识库/数据/文件）                       │
│  ├── 按域分类存储                                           │
│  └── 产出 domain_samples                                    │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Life: SleepEngine（睡眠整合）                               │
│  Phase 1: 浅睡 - 记忆巩固（ContextManager + WorkingMemory）  │
│  Phase 2: 深睡 - 训练神经元独立 lm_head + STDP               │
│  Phase 3: 知识整合 - SleepConsolidator 修剪 + fingerprint    │
│  Phase 3.5: 知识蒸馏（按域）                                 │
│  Phase 4: 评估 - apoptosis 检查                             │
│  Phase 5: 梦境 - RecursiveImprover 弱项→训练数据             │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Brain: 神经元参数已更新                                     │
│  ├── 独立 lm_head 训练（P7: 域专用 vocab）                   │
│  ├── side_channels STDP 强化                                │
│  ├── 弱连接修剪                                             │
│  ├── fingerprint 更新                                       │
│  ├── neurogenesis 新神经元（幼稚态）                         │
│  └── apoptosis 清理弱神经元                                  │
└────────────────────────┬────────────────────────────────────┘
                         ↓
                    回到 Cortex 推理（闭环）
```

---

## 五、实施优先级

### P0（必须，否则闭环不成立）
1. **SleepEngine Phase 2 适配 Cortex**：训练目标从 ModelSelf 改为神经元独立 lm_head
2. **FeedEngine 按域分类**：samples 必须按 domain 存储
3. **BodyCore.set_cortex**：身体感知大脑存在

### P1（重要，显著提升闭环质量）
4. **LifeScheduler 需求→神经元行为映射**：5 维需求驱动 neurogenesis/sleep/play
5. **SleepEngine Phase 3 调用 SleepConsolidator**：两套睡眠协同
6. **Senses 按域路由**：输入自动检测 domain

### P2（增强，完善闭环细节）
7. **Limbs 执行结果反馈 feed_engine**：实践知识闭环
8. **Metabolism 驱动神经调质**：硬件状态影响学习
9. **EvolutionEngine 触发 neurogenesis**：进化阶段转换
10. **PlayEngine 自由共振**：玩耍强化 CoactivationTracker

### P3（可选，锦上添花）
11. **DomainDetector 独立模块**：更准确的域检测
12. **ExploreEngine 按域存储**：探索结果分类

---

## 六、CPU 模式兼容性

### 6.1 CPU 模式下的调整

| 机制 | CPU 行为 |
|------|---------|
| CUDA stream 并行 | 自动退化为串行（已实现） |
| lm_head 独立模式 | 正常工作，每 neuron 仅 5-10M 参数量 |
| 睡眠训练 | 正常工作，但慢 |
| neurogenesis | 正常工作（蒸馏也支持 CPU） |
| 神经调质 | 纯标量运算，无影响 |

### 6.2 CPU 模式规模建议

| 神经元数 | CPU 可行性 |
|---------|-----------|
| 5（当前） | ✓ 流畅 |
| 10-20 | ✓ 可用，每 token 几秒 |
| 50+ | ✗ 不现实，考虑减少 max_rounds |

### 6.3 CPU 优化建议

- 推理时 `torch.no_grad()` + `torch.compile`（如果 PyTorch 版本支持）
- 睡眠训练用小 batch_size（1-4）
- 考虑 int8 量化（lm_head.weight，每 neuron 独立量化）
- max_rounds 从 3 降到 2

---

## 七、与一代架构的兼容性

### 7.1 旧 ModelSelf 引用清理

需要搜索所有 `model_self` / `ModelSelf` 引用，替换为 `cortex`：

| 文件 | 改动 |
|------|------|
| life/sleep_engine.py | `self.model_self` → `self.cortex` |
| life/life_scheduler.py | 引用 model_self 的地方改 cortex |
| api/routes_*.py | 推理路由从 model_self 改 cortex |
| body/core.py | set_model_self → set_cortex |

### 7.2 旧依赖清理

sleep_engine 引用的不存在模块需要处理：
- `taiji/agent_ext/data_collector` → 用 feed_engine 替代
- `taiji/agent_ext/knowledge_learner` → 用 sleep_engine Phase 2 训练替代
- `taiji/infra/self_evaluator` → 用 QualityFilter 替代
- `taiji/infra/user_profile` → 暂时移除或简单实现
- `taiji/infra/auto_upgrade` → 用 lifecycle.apoptosis 替代

---

## 八、决策结果（用户确认）

| 决策点 | 决策方案 | 实施细节 |
|--------|---------|---------|
| 1. DomainDetector | **A+C 混合**：规则粗筛 + general 细调 | 规则启发式作为 fallback，general 神经元输出路由分数（待训练） |
| 2. neurogenesis teacher | **C 分场景**：子域用 neuron，新域用 1.5B | 已有域的子专家用现有 expert 做 teacher；全新域仍用外部 1.5B |
| 3. RecursiveImprover | **B 转型为策略层** | 保留弱项分析+训练数据生成；废弃 `design_next_generation`；执行交给 neurogenesis |
| 4. PlannerSystem 反馈 | **B+C 组合** | 失败->neurogenesis 信号 + dopamine↓；成功->feed_engine 实践样本 + dopamine↑ |
| 5. ModelSelf | **C 完全移除，单神经元模式作 fallback** | Cortex N=1 时跳过共振直接 forward；N=0 抛错 |
