# Seed / Taiji（taiji-neuron）项目全面分析报告

> 分析日期：2026-08-22 ｜ 分析方式：源码实读 + 结构勘察
> 项目定位：**Seed** 是模型与项目名，**Taiji** 是其原生预测计算基底（TPF），目标是整体替代 Transformer 底层。当前状态为"可执行研究原型"，同时附带一套已产品化的 API 服务 + Web 前端 + 桌面客户端（基于冻结的 Legacy NeuroPlex 基线运行）。

---

## 一、项目架构

### 1.1 总体分层

项目本质上是**"一套仓库、两条轨道、三层结构"**：

```text
┌────────────────────────────────────────────────────────────┐
│  桌面端 desktop/（PyQt6 + QtWebEngine 壳，NSIS 分发）        │
│  前端  frontend/（Vue 3 SPA，Vite 构建）                    │
├────────────────────────────────────────────────────────────┤
│  API 层 api/（FastAPI，21 个 router、约 150 个端点）         │
│    REST + SSE + WebSocket                                  │
├───────────────────────────────┬────────────────────────────┤
│  轨道 A：Legacy NeuroPlex     │  轨道 B：Native Taiji（活跃）│
│  neuroplex/（冻结基线）        │  taiji/（原生基底）          │
│  九成员 Transformer 种群       │  seed/（公共模型门面）       │
│  对话/agent/RAG/生命引擎       │  纯局部学习，无反向传播      │
└───────────────────────────────┴────────────────────────────┘
```

- **轨道 B（活跃研究方向）**：`taiji/` + `seed/` + `tests/taiji_native/` + `scripts/training/verify_taiji_*`。README/CONTRIBUTING 明确规定：`taiji/` 禁止 import `seed`、`neuroplex`、`transformers`、tokenizer、注意力实现；`seed/` 只能依赖公开 Taiji API。CI 只跑这条轨道的验证。
- **轨道 A（冻结的产品基线）**：`neuroplex/` 九成员 Transformer 种群。API 服务的全部业务逻辑（chat/agent/RAG/生命引擎）实际都委托给它——**"Seed/Taiji"目前只是产品名与路由前缀，产品后端尚未迁移到原生基底**。

### 1.2 目录结构与职责

| 目录 | 职责 | 规模 |
|---|---|---|
| `taiji/` | 原生预测基底：config/sparse/state/organs/memory/fabric/model/environment | 9 文件，约 3100 行核心代码 |
| `seed/` | 公共模型门面，纯委托 Taiji | 2 文件，约 200 行 |
| `neuroplex/` | 冻结基线：resonance（共振场）、brain、agent、agent_ext、life（八大生命引擎）、body、domains（五域）、services、tools、multimodal、safety、core | 113 个 py 文件 |
| `api/` | FastAPI 服务层：21 个 routes_*、middleware、training | 38 个 py 文件 |
| `frontend/` | Vue 3 SPA：6 视图 + 15 组件 + 10 composables + 3 Pinia store | src 下约 35 个源文件（node_modules 未入库价值） |
| `desktop/` | PyQt6 桌面壳 + PyInstaller/NSIS 打包 | 6 文件 |
| `scripts/` | training（~150）/ data_prep（~50）/ archive（~130 历史实验）/ maintenance / utils | 314 个 py |
| `tests/` | taiji_native（10，新架构契约）/ resonance（5）/ seed（2）/ 根级契约测试（10）；**tests/taiji 为空** | 27 个 py |
| `data/` `logs/` `reports/` | 训练数据与 ckpt、训练历史 JSON、验证实验报告（JSON） | 840+ 文件 |
| `plans/active/` | 9 份架构方向文档（TAIJI_SUBSTRATE_ARCHITECTURE 等） | 24 个 md |
| `_libs/` | 内置 sentencepiece 0.2.2 发行版（含二进制，免 pip） | 22 文件 |

### 1.3 关键依赖规则（CONTRIBUTING 强制）

- `seed → taiji`（单向）；`api → neuroplex`（现状）；`taiji` 对外零依赖（仅 PyTorch 张量引擎）
- 正常学习路径禁止 `backward()` 与全局优化器
- 每个持久状态必须定义 update/decay/reset/checkpoint/lesion 行为
- 能力声明必须有确定性基准 + 因果消融（lesion）对照

---

## 二、技术栈

### 2.1 语言与运行时

- **Python ≥ 3.10**（核心运行时）；Node ≥ 20.19（前端构建）

### 2.2 后端

| 类别 | 技术 |
|---|---|
| 张量引擎 | PyTorch ≥ 2.0（Taiji 仅当张量库用，不用 autograd） |
| Web 框架 | FastAPI + uvicorn + starlette；SSE 流式 + WebSocket |
| 鉴权 | 自研 JWT（HMAC-SHA256）+ PBKDF2 密码（实现在 neuroplex/core/security.py） |
| RAG | 自研混合检索：sentence-transformers（MiniLM 多语）稠密向量 + 自实现 BM25（0.4 权重），可选 CrossEncoder 重排，npy/pkl 持久化，无外部向量库 |
| Legacy 依赖组 | transformers、sentencepiece、peft、accelerate、langchain 全家桶、jieba、tensorboard 等（`[legacy]` extra） |
| 可选组 | gpu（bitsandbytes/scipy）、voice（edge-tts/pyttsx3）、desktop（PyQt6）、build（pyinstaller） |

### 2.3 前端

- **Vue 3.5**（Composition API）+ vue-router 4（hash 模式）+ **Pinia 3**
- **Naive UI 2.40**（unplugin 按需自动导入）、Monaco Editor、xterm.js、marked + highlight.js + DOMPurify
- **Vite 8** 构建；自定义插件剥离 `crossorigin` 以兼容 QWebEngineView

### 2.4 桌面与分发

- PyQt6 + QtWebEngine 壳（加载 `http://127.0.0.1:8000` 后端静态服务，非本地文件）
- PyInstaller（onedir + windowed）→ NSIS MUI2 中文安装器（SeedSetup.exe，v1.6.0）

### 2.5 工具链

- pytest（`-p no:cacheprovider`，slow marker）、black、ruff
- **GitHub Actions CI**：Python 3.10/3.12 矩阵，CPU torch，跑 `verify_taiji_native_v5` 与 `verify_taiji_n7_context` 端到端验证
- 无 Docker、无 .env、无 compose——本地/单机部署形态

---

## 三、核心功能

### 3.1 Taiji 原生基底（研究核心）

替代 Transformer 各职责的完整映射（README）：

| Transformer 职责 | Taiji Native v6 实现 |
|---|---|
| tokenizer + embedding | 256 原始字节感受器 + 边界感受器（257 one-hot） |
| 位置编码 | 因果 tick 与持久状态 |
| 自注意力 | 稀疏互逆预测 + 循环转移 |
| FFN/残差 | 膜积分、侧抑制、自适应阈值、trace |
| KV cache / 检索 | 有界动态状态 + 分布式联想 engram（无 K/V 槽位） |
| 全局反向传播 | 既有边上的局部预测/状态/动作/记忆增量 |
| LM head | 全状态稀疏感受器银行 + 单动作种群 |
| 自回归解码 | 动作字节经同一感受器回流 |

核心能力模块：

1. **TaijiFabric**（预测循环 tick）：逐区域执行 decoder 预测→误差→互逆回投→循环预测→延迟自上而下→膜积分→侧抑制竞争→自适应阈值→trace→局部学习（restructure 先于权重）。
2. **EpisodicField**（分布式情景记忆）：固定稀疏编码器把皮层状态/动作/结果/奖励/正弦时钟/episode 哈希/来源投影到重叠场单元；cue→event 循环补全 + 8 组 readout；新颖度与奖励门控写入；**零事件槽位分配**。已验证一次性跨 episode 回忆 87.5%（vs 25% 消融对照）。
3. **主动环境交互**（act/settle_action 事务）：待定动作与资格原子化 checkpoint，结果结算后写入一个分布式 engram；N11 验证在线 100% 成功率 vs 50% 随机。
4. **内生回放**（M6）：value 轴 + 时钟 + 噪声种子生成回放，trace 作疲劳偏移防单 engram 垄断，W 不被自生成模式修改（防假记忆）。
5. **checkpoint**：`taiji-native-v6` 格式、STATE_VERSION=5，含拓扑、参数、全部快状态、事务状态与 RNG；支持精确续跑。

基准成绩（committed verification）：字节循环准确率 0→94.12%、惊喜下降 97.98%、128 步自由生成全对、N7 上下文 8/8、稀疏迁移前向误差 ≤ 2.98e-8。

### 3.2 Legacy NeuroPlex（产品后端实际承载体）

- **九成员 Transformer 种群**：同 zh 域不同数据增广/规格（compact 51M / standard 134M / expert + hub 神经元），共享 3072 维 embedding 场做共振融合，leader 选举 + judge 判定头仲裁质量。
- **生物隐喻运行时**：共振场（ResonanceField）、theta-gamma 振荡耦合、STDP、神经调质（DA/5-HT/NE/ACh 驱动 lr/不应期/场写入强度）、族群蒸馏、睡眠巩固（SleepEngine/SleepConsolidator）。
- **八大生命引擎**（life/）：feed / sleep / play / explore / science / evolution / integrate / recursive_improver——自主生命周期闭环。
- **Agent 体系**：ReAct 引擎、planner/reflector、语义/工作记忆、MCP 管理器、自我修改、沙箱执行。
- **五域 tokenizer**：zh 50k / en 16k / code 12k / math 10k / general 16k（SentencePiece）。

### 3.3 API 服务（约 150 端点，21 个 router）

- **chat**：`POST /api/chat/stream`（SSE，ReAct 结构化输出 life/thought/action/observation/final，引擎不可用回退 Cortex.generate）、会话历史 CRUD、文件上传解析（txt/pdf/docx，20MB）
- **agent**：工具注册/执行、react 与 react/stream、多角色协作、MCP 市场管理、agent 记忆
- **workspace**：文件树读写、任意 Python 运行、项目创建/删除、代码分析、依赖安装（admin）、插件上传
- **RAG**：上传/检索/配置/预览，混合检索
- **模型管理**：模型市场下载/切换/GGUF 导出、热重载
- **生命引擎**：/api/life 与 /api/taiji/* 30+ 端点（feed/sleep/play/evolve/self_mod 在线学习闭环）
- **多模态**：TTS、语音列表、文生图、图像描述
- **系统**：硬件信息、重启、更新检查/热补丁上传（upload_update/upload_patch/upload_ui/reload_modules）、WebSocket 终端（/ws/terminal，cmd.exe/bash）
- **训练**：/api/train/* 数据集管理与训练控制

### 3.4 前端页面与桌面端

- 6 个路由页：聊天（`/`）、知识库（/kb）、训练（/train）、Agent 配置（/agent）、工作区（/workspace）、生命状态（/life）、设置（/settings），全部懒加载
- 桌面端：双子进程架构（uvicorn:8000 + websocket 服务:8765），QThread 每 10s 巡检自动重启崩溃子进程，系统托盘，loading.html 轮询 /api/health 后跳转

---

## 四、数据流与交互

### 4.1 聊天主数据流（产品侧）

```text
前端 ChatView (authFetch, JWT Bearer)
  → POST /api/chat/stream
    → 生命调度 handle_user_directive
    → _apply_rag（rag_kb 混合检索注入上下文）
    → chat_strategies._stream_unified
        · 注入日期时间 system_prompt
        · context_manager.build_context（记忆 + 历史）
        · ReActEngine.run_stream → 结构化 SSE 事件
        · 失败回退 Cortex.generate（asyncio.to_thread）
    → 写 evolution / recursive_improver / DataCollector(jsonl)
    → [DONE]
```

### 4.2 Taiji 内部数据流（研究侧，一个因果 tick）

```text
observe(symbol)
  ├─ 结算上一个动作事务 → EpisodicField 绑定 cue/action/reward/outcome
  ├─ ByteSensor.encode（257 one-hot）
  ├─ TaijiFabric.step（预测→误差→回投→循环→top-down→膜/抑制/阈值/trace→局部学习）
  ├─ 拼接全部区域 fast activity + slow trace → 皮层状态
  ├─ EpisodicField.recall（循环补全 + 共振门控读出 + 延迟一 tick 反馈注入）
  ├─ SparseReceptorBank（每个皮层坐标恰一条固定带极性边 → K 共享通道）
  └─ ByteMotor.probabilities → 原子安装下一 TaijiState
```

主动环：`act(affordances) → environment.step → settle_action(reward) → observe(outcome, learn_motor=False)`。

### 4.3 前后端交互方式

- **REST + SSE** 为主（chat、react/stream 流式）；**WS** 两条通道：`/ws/terminal`（8765?）与实时推送（8765 独立端口，自动重连）
- API base 解析：开发走 Vite 代理（/api → 8000）；生产非 8000 端口时强制拼 `http://host:8000`；`?taiji_client=desktop` 标识桌面端
- 401 广播 `taiji-auth-expired` 事件；5xx 指数退避重试

---

## 五、配置与部署

### 5.1 配置体系

| 来源 | 内容 |
|---|---|
| `pyproject.toml` | 依赖分组：核心（仅 torch）/ dev / legacy / gpu / voice / desktop / build；入口脚本 `seed-legacy-desktop` |
| 环境变量 | `TAIJI_ALLOWED_ORIGINS`（CORS）、`TAIJI_API_KEYS`、`TAIJI_EMBEDDING_MODEL`、`TAIJI_PIP_INDEX` 等（config.py apply_env_overrides） |
| settings_service | 持久化运行时设置（terminal_enabled、workspace_path 等） |
| 硬编码 | 端口 8000/8765、限流阈值、CORS 默认值、max_steps 散落各处 |

### 5.2 部署流程

- **开发**：`pip install -e ".[dev]"` → `python -m api.main`（uvicorn 127.0.0.1:8000）+ `cd frontend && npm run dev`（Vite 5173 代理）
- **桌面分发**：`desktop/build.py` 三步——前端 `npm run build` → PyInstaller onedir/windowed（数据文件含 frontend/dist、taiji_data/final、torch hidden imports）→ 复制 knowledge_store 等运行时目录 → NSIS 打包 SeedSetup.exe
- **CI**：GitHub Actions（3.10/3.12 矩阵），CPU torch，跑两个核心 verify 脚本
- 打包版（run_app.py）额外支持：30 包依赖自检自动 pip 安装、`HotUpdateImporter` 从 update_code/ 热补丁

---

## 六、关键代码逻辑与设计模式

### 6.1 SparseSynapses——统一突触算子（taiji/sparse.py，412 行）

压缩固定 fan-in 拓扑：`pre_index[post, F] int32 + edge_weight[post, F] float32`，post 索引隐含于行号。四个原语：
- `forward(pre)`：按行 gather 加权求和（_postsynaptic 证据）
- `backproject(error)`：`scatter_add_` 互逆自下而上回投
- `local_update(error, trace)`：仅存边上 error × eligibility（error/√L0 归一），学习率与衰减由 learn_scale 同步门控（防弱门控 replay 变净遗忘）
- `structural_update`：沉默伙伴 weakest-first 退休（保护新零权边）+ donor 按活动 argsort + 能量捕获率终止 + error 阈值门控

无 mask、无稠密外积、权重 `requires_grad=False`。load 时验证形状/越界/行内重复/self-contact（scatter_add 对重复索引会双计）——**整个系统（fabric/motor/memory）共用这一个算子类，是典型的策略/原语复用模式**。

### 6.2 TaijiFabric.step()——规范 tick 顺序（fabric.py，407 行）

逐区域：decoder 预测 → 误差 → 循环预测 → 互逆回投 → 延迟 top-down → 增益加权 drive（含 episodic feedback 的 activity/trace 两段切片）→ 有界泄漏膜积分 → `ReLU(u−θ)` → 学得 lateral bank 逐单元竞争 → `tanh(ReLU(u−θ−i))` → homeostasis 阈值积分（replay 时冻结）→ trace EMA。学习顺序：**restructure 先于权重**（结构换边后立即获首次写入）；lateral 反 Hebb 更新（a_i·a_j−mean²，clamp≥0 保抑制极性）。README 明言：改操作顺序即架构变更，需新 state version。

### 6.3 EpisodicField 门控记忆（memory.py，814 行，系统最大模块）

- **写**：novelty=‖event−W·cue‖/‖event‖ 与 |r| 联合门控学习率；W 先 cue→event 全率、再 event→event 半率自联想；action readout 乘 reward——"记录动作、按效价调证词"的三因子学习
- **读**：J 次迭代循环补全；confidence = familiarity × resonance，所有读出与 cortical feedback 均乘 confidence；feedback 延迟一 tick 进 fabric 避免代数环
- **回放**：value 轴 + 时钟 + 噪声作种子；trace 作零均值疲劳偏移阈值（尖频适应，防单 engram 垄断）；W 不被自生成模式修改

### 6.4 事务状态机（state.py）

`PendingAction` / `PendingExperience` 保证动作信用不错配：待定动作与资格原子化进 checkpoint，直到结果结算。这是主动环境下因果正确性的关键设计。

### 6.5 产品侧关键逻辑

- **ReAct 引擎 + SSE**：结构化事件流（life/thought/action/observation/final），失败自动降级到 Cortex.generate 的线程池包装
- **自研 JWT 中间件**：纯 ASGI 实现（JWTAuthMiddleware），secret 首次生成后存 `.jwt_secret` 文件
- **run_app.py 热更新**：`HotUpdateImporter` 插入 sys.meta_path，从 update_code/ 加载补丁——支持线上修复（同时也是最大攻击面）

### 6.6 设计模式总结

门面（Taiji/Seed 双层门面）、原语复用（SparseSynapses 四算子统一全系统）、payload 序列化（to_payload/load_payload 精确持久化）、事务状态机、冻结 dataclass 配置契约（TaijiConfig 70+ 参数全校验）、不可变结果对象（MemoryRecall/TaijiStep/TaijiDecision）。文档质量突出：docstring 内嵌实测数据（能量捕获率、7:1 棘轮分析），属"实验驱动设计"的范本。

---

## 七、潜在问题与优化建议

### 7.1 P0 —— 安全（产品侧）

1. **鉴权链路实际损坏**：`routes_auth.py` 委托的 `neuroplex/services/auth_service.py` 是 9 行空 stub（无 login 等函数）→ 登录端点运行时必然 500，除非热更新补丁替换。真实逻辑在 `neuroplex/core/security.py`，但从未接线。
2. **默认全裸奔**：`auth.enabled=False` 时 JWTAuthMiddleware 放行全部请求；`/assets`、`/workspace_data`、`/ws/` 前缀**永远公开**；`security.py` 的 create_auth_middleware 是从未启用的死代码。
3. **高危执行面**（仅靠 127.0.0.1 绑定缓解，一旦改绑定即远程 RCE）：`/api/workspace/run`（任意 Python）、`/api/agent/tools/execute`、`/ws/terminal`（完整 shell）、`/api/system/restart`、`upload_update/upload_patch/reload_modules`（热更新=任意代码执行）。terminal 声明 300s 空闲超时但**未实现**。
4. **其他**：JWT 走 URL query（易入日志）；run_app 启动期自动 pip 安装（供应链风险）；workspace 插件上传无类型白名单。

**建议**：修复 auth_service 接线并默认启用鉴权；执行类端点统一收敛到沙箱执行器（sandbox_executor 已存在）；热更新端点加签名校验；terminal 实现超时与审计日志。

### 7.2 P0 —— 功能正确性

- 多处 `return HTTPException(...)` 而非 `raise`（routes_rag 等）→ 错误以 200 状态码返回对象，前端错误处理被绕过。全局 grep 修复。
- 大量 `except: pass` 吞异常，故障排查困难。

### 7.3 P1 —— Taiji 性能天花板

1. `SparseSynapses.__init__` 逐行 Python 循环 + 每行画 `randn(in_features)`：初始化时间/内存峰值 O(out×in)，与"不保留 dense"的设计意图矛盾，规模上不可扩展。可改为对每行仅采样 F 个索引（torch.randint + gather）或 top-k 初始化。
2. 单样本 1-D 张量逐 tick 循环，无批处理；大量 `.item()` 触发 GPU 同步。研究验证可接受，但规模化前需批化 tick 或至少消除逐元素同步。
3. 文档自认小规模下 edge 字节达 dense 的 111%（默认配置投影 98.59%）——按边语义当前并非加速，需诚实标注。
4. `parameter_count(active_only)` 两分支返回相同值（死代码/未实现）。

### 7.4 P1 —— 工程冗余与一致性

1. **双入口**：`desktop/main.py` 与 `api/run_app.py` 功能重叠（自述"计划合并未完成"）；`desktop/build.py` 与 `seed.spec` 打包配置重复且 hidden imports 不一致（build.py 版本过时）。
2. **硬编码端口** 8000/8765 散落 vite.config、apiClient、useWebSocket、desktop/main.py，无统一配置，端口冲突无退避。
3. **双层限流重复**（security.py 100/min + 内置分类限流），defaultdict 滑窗潜在内存增长。
4. 巨型文件：ensemble.py ~3800 行、TrainingView.vue 1249 行、ChatView.vue 1014 行、useTraining.js 909 行、memory.py 814 行——需拆分。
5. INTERFACE_REFERENCE.md 记录的 legacy 接口陷阱（side_channels 别名、6 元组解包、同公式双语义等）本身即是技术债清单，冻结策略正确但不要再投入。

### 7.5 P2 —— 仓库卫生

- `frontend/node_modules`、`dist` 在工作区（确认 .gitignore 覆盖）；`shoot-fe.cjs` 截图脚本硬编码 `E:/taiji/.taiji_test_tmp` 项目外路径，属临时脚本混入仓库。
- `desktop/settings.json` 把用户窗口状态写入代码目录（打包后写入安装目录，存在权限隐患）。
- `scripts/archive/` 130 个历史脚本、logs/reports/data 840+ 产物文件混入仓库，建议用 artifact 存储或 LFS 分离。
- `tests/taiji/` 为空目录（仅 pycache）——删除或并入 taiji_native。
- 双份依赖清单（requirements.txt / requirements-legacy.txt）与 pyproject extra 并存，建议收敛到 pyproject 单一来源。

### 7.6 总体评价

这是一个**罕见的、方法论严谨的研究项目**：每个能力声明配套确定性基准 + 因果消融、种子面板而非单种子决策、冻结基线供同预算对照、AST 级边界契约测试防架构污染。核心技术风险在性能可扩展性（初始化 O(out×in)、无批处理）；产品侧最大风险在安全面（鉴权损坏 + 任意执行端点）。两条轨道的"研究严谨"与"产品裸奔"形成鲜明反差——建议在原生基底尚未接管产品后端之前，优先收紧 API 安全边界。

---
*报告基于源码实读与结构勘察生成；行号与细节以当前工作区快照为准。*
