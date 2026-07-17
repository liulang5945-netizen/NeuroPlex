# Phase 4 Integration Plan — ResonanceEnsemble 上层适配

## 状态：代码就位，待集成

### 已创建

| 文件 | 用途 |
|------|------|
| [`taiji/brain/cortex.py`](taiji/brain/cortex.py) | 共振场意识中心，封装 ResonanceEnsemble + generate() |

### 集成要点

#### 1. API 层 (`api/chat_strategies.py`)

当前调用链：
```
chat_strategies.py → taiji.generate_stream() → 1.5B ModelSelf
```

新调用链：
```
chat_strategies.py → cortex.generate() → ResonanceEnsemble → 5 neurons
```

需要改动：
- 在 `app_state` 中初始化 `Cortex` 实例
- 在 `_taiji_inference()` 中将 `taiji.generate_stream()` 替换为 `cortex.generate()`
- 保留 feature flag，支持切换：`USE_RESONANCE = os.environ.get("TAIJI_USE_RESONANCE", "0") == "1"`

#### 2. Agent 层 (`taiji/agent/`)

- `planner.py`: `self.model.forward()` → `cortex.think()`
- `reflector.py`: 同上
- Agent 的知识检索结果注入 shared embedding 上下文

#### 3. Life 层 (`taiji/life/`)

- `life_scheduler.py`: 训练目标从单体模型 → 神经元培育/淘汰
- `sleep_engine.py`: 睡眠 → 神经元内部参数整合 + 抱合生长
- `evolution_engine.py`: 进化 → 神经元培育 + 淘汰 + 重组
- `feed_engine.py`: 数据投喂 → 触发神经元蒸馏训练

#### 4. 启动时加载 (`api/app.py`)

```python
# 在 lifespan 中加载 Cortex
from taiji.brain.cortex import Cortex
app_state.cortex = Cortex(neurons_dir="data/neurons")
```

### 执行顺序

1. `app.py`: 启动时加载 Cortex + neurons
2. `chat_strategies.py`: 推理时调用 cortex.generate()
3. `agent/planner.py`: Agent 规划用 cortex.think()
4. `life/`: 按需逐步适配（非阻塞）
