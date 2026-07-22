<template>
  <section class="dedicated-view training-view">
    <!-- 顶栏标题 -->
    <div class="view-header">
      <div class="header-left">
        <h2>微调训练</h2>
        <span class="header-sub">
          <span v-if="taijiModelInfo.size">{{ taijiModelInfo.size }} · {{ taijiModelInfo.config?.num_hidden_layers }}层 · {{ taijiModelInfo.config?.hidden_size }}维</span>
          <span v-else>态极微调 · LoRA 低秩适配</span>
        </span>
      </div>
      <div class="header-actions">
        <n-button v-if="trainState === 'running' || trainState === 'paused'" type="error" round @click="stopTraining(toast)">
          <template #icon><StopCircle :size="14" /></template>停止训练
        </n-button>
        <n-button v-if="trainState === 'running'" type="primary" round @click="saveCheckpoint">
          <template #icon><PackageIcon :size="14" /></template>保存检查点
        </n-button>
        <n-button v-if="trainState === 'idle' || trainState === 'completed'" type="primary" round @click="startTaijiTraining(toast)">
          <template #icon><Zap :size="14" /></template>开始态极微调
        </n-button>
      </div>
    </div>

    <div class="view-body">
      <!-- 标签页 -->
      <div class="tabs" role="tablist">
        <button class="tab" :class="{ active: activeTab === 'overview' }" role="tab" @click="activeTab = 'overview'">训练概览</button>
        <button class="tab" :class="{ active: activeTab === 'hyperparams' }" role="tab" @click="activeTab = 'hyperparams'">超参数</button>
        <button class="tab" :class="{ active: activeTab === 'dataset' }" role="tab" @click="activeTab = 'dataset'">数据集</button>
        <button class="tab" :class="{ active: activeTab === 'logs' }" role="tab" @click="activeTab = 'logs'">日志</button>
      </div>

      <!-- ═══ Tab 1 · 训练概览 ═══ -->
      <section class="tab-panel" :class="{ active: activeTab === 'overview' }">
        <!-- 进度英雄卡 -->
        <div v-if="trainState === 'running' || trainState === 'paused'" class="tk-card">
          <div class="progress-hero">
            <div>
              <div class="card-sub" style="margin-bottom:6px">{{ taijiModelInfo.size || '态极模型' }} · LoRA 微调</div>
              <div class="pct">{{ trainProgress }}<span class="unit">%</span></div>
            </div>
            <div class="right-meta">
              <span class="status-chip-run" v-if="trainState === 'running'">训练中</span>
              <span class="status-chip-paused" v-else>已暂停</span>
            </div>
          </div>
          <div class="progress-bar">
            <div class="fill" :class="{ paused: trainState === 'paused' }" :style="{ width: trainProgress + '%' }"></div>
          </div>
          <div class="progress-meta">
            <span class="pm-item"><svg class="pm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>当前 epoch <b>{{ trainMetrics.epoch }} / {{ trainMetrics.total_epochs }}</b></span>
            <span class="pm-item"><svg class="pm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>step <b>{{ trainMetrics.total_steps > 0 ? Math.round(trainMetrics.epoch * trainMetrics.total_steps / trainMetrics.total_epochs) : '--' }} / {{ trainMetrics.total_steps || '--' }}</b></span>
            <span class="pm-item"><svg class="pm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l1.5-5 3 10 1.5-5h8"/></svg>剩余约 <b>{{ fmtTime(trainMetrics.eta) }}</b></span>
            <span class="pm-item"><svg class="pm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6h16v12H4z"/><path d="M4 10h16M8 14h4"/></svg>吞吐 <b>{{ trainMetrics.samples_per_sec >= 0.005 ? (trainMetrics.samples_per_sec < 0.1 ? trainMetrics.samples_per_sec.toFixed(2) : trainMetrics.samples_per_sec.toFixed(1)) + '/s' : '--' }}</b></span>
          </div>
        </div>

        <!-- 空状态（未开始训练） -->
        <div v-else class="tk-card">
          <div class="progress-hero">
            <div>
              <div class="card-sub" style="margin-bottom:6px">{{ taijiModelInfo.size || '态极模型' }} · LoRA 微调</div>
              <div class="pct">0<span class="unit">%</span></div>
            </div>
            <div class="right-meta">
              <span class="status-chip-idle">待开始</span>
            </div>
          </div>
          <div class="progress-bar"><div class="fill" style="width:0%"></div></div>
          <div class="progress-meta">
            <span class="pm-item"><svg class="pm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>当前 epoch <b>0 / {{ taijiTrainParams.num_epochs }}</b></span>
            <span class="pm-item"><svg class="pm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>step <b>0 / --</b></span>
            <span class="pm-item"><svg class="pm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l1.5-5 3 10 1.5-5h8"/></svg>剩余约 <b>--</b></span>
            <span class="pm-item"><svg class="pm-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6h16v12H4z"/><path d="M4 10h16M8 14h4"/></svg>吞吐 <b>--</b></span>
          </div>
        </div>

        <!-- 双列图表 -->
        <div class="charts-grid">
          <!-- Loss 曲线 -->
          <div class="tk-card chart-card">
            <div class="chart-head">
              <h4>Loss 曲线</h4>
              <span class="legend"><i style="background:var(--chart-2)"></i>训练损失</span>
            </div>
            <canvas v-if="trainLoss.length >= 2" ref="lossCanvasRef" class="loss-canvas" width="600" height="170"></canvas>
            <div v-else class="chart-empty">暂无数据，训练开始后自动绘制</div>
          </div>

          <!-- 学习率曲线（静态示意 + 当前值） -->
          <div class="tk-card chart-card">
            <div class="chart-head">
              <h4>学习率曲线</h4>
              <span class="legend"><i style="background:var(--chart-3)"></i>warmup + cosine decay</span>
            </div>
            <svg class="chart-svg" viewBox="0 0 500 280" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="学习率曲线">
              <defs>
                <linearGradient id="lrFill-vue" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="var(--chart-3)" stop-opacity="0.26"/>
                  <stop offset="100%" stop-color="var(--chart-3)" stop-opacity="0"/>
                </linearGradient>
              </defs>
              <!-- 网格 -->
              <line class="grid-line" x1="50" y1="20" x2="480" y2="20"/>
              <line class="grid-line" x1="50" y1="95" x2="480" y2="95"/>
              <line class="grid-line" x1="50" y1="169" x2="480" y2="169"/>
              <line class="axis-line" x1="50" y1="244" x2="480" y2="244"/>
              <!-- Y 轴标签 -->
              <text class="axis-text" x="44" y="23" text-anchor="end">3e-5</text>
              <text class="axis-text" x="44" y="98" text-anchor="end">2e-5</text>
              <text class="axis-text" x="44" y="172" text-anchor="end">1e-5</text>
              <text class="axis-text" x="44" y="247" text-anchor="end">0</text>
              <!-- X 轴标签 -->
              <text class="axis-text" x="50" y="260" text-anchor="middle">0</text>
              <text class="axis-text" x="136" y="260" text-anchor="middle">200</text>
              <text class="axis-text" x="222" y="260" text-anchor="middle">400</text>
              <text class="axis-text" x="308" y="260" text-anchor="middle">600</text>
              <text class="axis-text" x="394" y="260" text-anchor="middle">800</text>
              <text class="axis-text" x="480" y="260" text-anchor="middle">1000</text>
              <text class="axis-text" x="265" y="275" text-anchor="middle">step</text>
              <!-- warmup 标记区 -->
              <rect x="50" y="20" width="43" height="224" fill="var(--chart-1)" opacity="0.05"/>
              <text class="axis-text" x="71" y="14" text-anchor="middle" fill="var(--chart-1)">warmup</text>
              <!-- 面积填充 -->
              <path d="M50 244 L72 158 L93 72 L136 77 L179 92 L222 115 L265 143 L308 173 L351 201 L394 224 L437 239 L480 244 L50 244 Z" fill="url(#lrFill-vue)"/>
              <!-- 学习率线 -->
              <path d="M50 244 L72 158 L93 72 L136 77 L179 92 L222 115 L265 143 L308 173 L351 201 L394 224 L437 239 L480 244" fill="none" stroke="var(--chart-3)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
              <!-- 峰值标注 -->
              <circle cx="93" cy="72" r="4" fill="var(--chart-3)" stroke="var(--card)" stroke-width="2"/>
              <g transform="translate(103,58)">
                <rect x="0" y="0" width="92" height="20" rx="5" fill="var(--chart-3)"/>
                <text x="46" y="14" text-anchor="middle" font-family="var(--font-mono)" font-size="11" font-weight="600" fill="var(--primary-foreground)">峰值 2.3e-5</text>
              </g>
              <line x1="97" y1="68" x2="103" y2="64" stroke="var(--chart-3)" stroke-width="1.2"/>
            </svg>
          </div>
        </div>

        <!-- 指标卡行 -->
        <div class="metrics-row">
          <div class="metric-card">
            <div class="m-label"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 17l5-6 4 4 5-8 4 5"/></svg>Train Loss</div>
            <div class="m-value">{{ trainMetrics.current_loss != null ? trainMetrics.current_loss.toFixed(2) : '--' }}</div>
            <span class="m-trend down"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width:12px;height:12px"><path d="M7 10l5 5 5-5"/></svg>持续下降</span>
          </div>
          <div class="metric-card">
            <div class="m-label"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 17l5-6 4 4 5-8 4 5"/><path d="M4 21h16"/></svg>Eval Loss</div>
            <div class="m-value">--</div>
            <span class="m-trend flat">等待评估</span>
          </div>
          <div class="metric-card">
            <div class="m-label"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.5" fill="currentColor"/></svg>Accuracy</div>
            <div class="m-value">--<span style="font-size:1.1rem;color:var(--muted-foreground)">%</span></div>
            <span class="m-trend flat">等待评估</span>
          </div>
          <div class="metric-card">
            <div class="m-label"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M13 3 5 14h6l-1 7 8-11h-6l1-7Z"/></svg>学习率</div>
            <div class="m-value">{{ trainMetrics.lr != null ? trainMetrics.lr.toExponential(1).replace('e', 'e') : '--' }}<span style="font-size:1.1rem;color:var(--muted-foreground)">{{ trainMetrics.lr != null ? '' : '' }}</span></div>
            <span class="m-trend flat">cosine decay 中</span>
          </div>
        </div>

        <!-- 检查点列表（概览页展示） -->
        <div v-if="pendingCheckpoints.length > 0" class="tk-card" style="margin-top:18px">
          <div class="card-head">
            <h3><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" style="width:17px;height:17px;color:var(--primary)"><path d="M5 4h11l3 3v13H5z"/><path d="M8 4v5h7M8 14h8"/></svg>检查点列表</h3>
            <span class="card-sub">{{ pendingCheckpoints.length }} 个可用</span>
          </div>
          <div class="ckpt-list">
            <div v-for="ckpt in pendingCheckpoints" :key="ckpt.filename" class="ckpt-item">
              <span class="ckpt-ic"><PackageIcon :size="16" /></span>
              <div class="ckpt-body">
                <div class="ckpt-name">{{ ckpt.filename }}</div>
                <div class="ckpt-meta">
                  <span class="ckpt-tag">Epoch <b>{{ ckpt.epoch }}</b></span>
                  <span class="ckpt-tag">Step <b>{{ ckpt.step }}</b></span>
                  <span class="ckpt-tag">Loss <b>{{ ckpt.loss?.toFixed(4) || '--' }}</b></span>
                </div>
              </div>
              <n-button size="small" type="info" quaternary round @click="resumeFromCheckpoint(toast, $confirm)">
                <template #icon><RefreshCw :size="14" /></template>恢复
              </n-button>
            </div>
          </div>
        </div>

        <!-- 发布与导出（概览页展示） -->
        <div v-if="trainState === 'completed' || publishingState !== 'idle'" class="tk-card" style="margin-top:18px">
          <div class="card-head">
            <h3><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" style="width:17px;height:17px;color:var(--primary)"><rect x="4" y="4" width="7" height="7" rx="1.5"/><rect x="13" y="4" width="7" height="7" rx="1.5"/><rect x="4" y="13" width="7" height="7" rx="1.5"/><rect x="13" y="13" width="7" height="7" rx="1.5"/></svg>发布与导出</h3>
          </div>
          <n-text depth="3" class="publish-desc">{{ t('publish_desc') }}</n-text>
          <div class="ctrl-row" style="margin-top:12px">
            <n-button type="primary" round :disabled="publishingState !== 'idle'" @click="publishModel(toast)">
              {{ publishingState === 'publishing' ? '发布中...' : t('publish_model_btn') }}
            </n-button>
            <n-button type="info" round :disabled="publishingState !== 'idle'" @click="exportModelToGGUF(toast, $confirm)">
              {{ publishingState === 'publishing' ? '导出中...' : t('export_gguf_btn') }}
            </n-button>
          </div>
        </div>
      </section>

      <!-- ═══ Tab 2 · 超参数 ═══ -->
      <section class="tab-panel" :class="{ active: activeTab === 'hyperparams' }">
        <div class="tk-card">
          <div class="card-head">
            <h3><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" style="width:17px;height:17px;color:var(--primary)"><circle cx="6" cy="7" r="2.2"/><circle cx="6" cy="17" r="2.2"/><circle cx="18" cy="12" r="2.2"/><path d="M8 7.5 12 5M8 16.5 12 19M8 16l7-4M8 8l7 3"/></svg>训练超参数</h3>
            <span class="card-sub">LoRA 微调 · {{ taijiModelInfo.size || '态极模型' }}</span>
          </div>
          <div class="hp-grid">
            <div class="hp-field">
              <label class="hp-label"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M13 3 5 14h6l-1 7 8-11h-6l1-7Z"/></svg>学习率 (Learning Rate)</label>
              <n-input-number v-model:value="taijiTrainParams.learning_rate" :min="0.00001" :step="0.00001" />
              <span class="hp-desc">控制每步参数更新幅度，过大会发散，过小收敛慢</span>
            </div>
            <div class="hp-field">
              <label class="hp-label"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="15" rx="2"/><path d="M3 9h18M8 4v5"/></svg>批大小 (Batch Size)</label>
              <n-input-number v-model:value="taijiTrainParams.batch_size" :min="1" :max="32" />
              <span class="hp-desc">每次迭代处理的样本数，受 GPU 显存约束</span>
            </div>
            <div class="hp-field">
              <label class="hp-label"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>Epoch 数</label>
              <n-input-number v-model:value="taijiTrainParams.num_epochs" :min="1" :max="100" />
              <span class="hp-desc">完整遍历训练数据的次数，过多易过拟合</span>
            </div>
            <div class="hp-field">
              <label class="hp-label"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l1.5-5 3 10 1.5-5h8"/></svg>预热步数 (Warmup Steps)</label>
              <n-input-number v-model:value="taijiTrainParams.warmup_steps" :min="0" :max="1000" />
              <span class="hp-desc">学习率从零线性升至峰值的步数</span>
            </div>
            <div class="hp-field">
              <label class="hp-label"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12c2-4 6-4 8 0s6 4 8 0"/></svg>权重衰减 (Weight Decay)</label>
              <n-input-number v-model:value="taijiTrainParams.weight_decay" :min="0" :step="0.001" />
              <span class="hp-desc">L2 正则化系数，抑制过拟合</span>
            </div>
            <div class="hp-field">
              <label class="hp-label"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10v4M11 10v4M15 10v4"/></svg>梯度累积 (Gradient Accumulation)</label>
              <n-input-number v-model:value="taijiTrainParams.gradient_accumulation_steps" :min="1" :max="32" />
              <span class="hp-desc">累积多步再更新，等效增大批量以节省显存</span>
            </div>
            <div class="hp-field">
              <label class="hp-label"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5h16v14H4z"/><path d="M4 9h16M9 9v10"/></svg>最大序列长度 (Max Seq Len)</label>
              <n-input-number v-model:value="taijiTrainParams.max_length" :min="64" :max="4096" />
              <span class="hp-desc">单样本最大 token 数，影响显存与上下文</span>
            </div>
            <div class="hp-field">
              <label class="hp-label"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="7" height="7" rx="1.5"/><rect x="13" y="4" width="7" height="7" rx="1.5"/><rect x="4" y="13" width="7" height="7" rx="1.5"/><rect x="13" y="13" width="7" height="7" rx="1.5"/></svg>LoRA rank</label>
              <n-input-number v-model:value="taijiTrainParams.lora_rank" :min="1" :max="64" />
              <span class="hp-desc">低秩适配矩阵的秩，影响可训练参数量</span>
            </div>
          </div>
          <div class="hp-actions">
            <n-button type="primary" round @click="applyAndRestart">应用并重启训练</n-button>
            <n-button round @click="resetDefaults">恢复默认值</n-button>
            <span style="flex:1"></span>
            <span class="card-sub" style="align-self:center">保存步频 {{ taijiTrainParams.save_steps }} · 保留 {{ taijiTrainParams.keep_checkpoints }} 个检查点</span>
          </div>
        </div>
      </section>

      <!-- ═══ Tab 3 · 数据集 ═══ -->
      <section class="tab-panel" :class="{ active: activeTab === 'dataset' }">
        <!-- 数据上传 -->
        <div class="tk-card" style="margin-bottom:18px">
          <div class="card-head">
            <h3><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" style="width:17px;height:17px;color:var(--primary)"><path d="M12 5v14M5 12h14"/></svg>数据上传</h3>
          </div>
          <FileUploadQueue
            ref="trainUploadRef"
            upload-endpoint="/api/train/upload_dataset"
            accept=".jsonl,.json,.txt,.csv,.md,.pdf,.docx,.doc,.xlsx,.xls,.pptx,.html,.htm,.epub,.rtf,.xml,.log,.py,.js,.ts,.css,.java,.c,.cpp,.sh,.sql,.png,.jpg,.jpeg,.bmp,.gif,.webp,.tiff,.tif"
            icon="BarChart2" title="训练数据上传" upload-icon="Download"
            :drop-text="t('train_upload')" :accept-hint="t('train_support')"
            success-text="✅ 数据集上传成功"
            @all-uploaded="loadTrainDatasets"
          />
        </div>

        <!-- 数据集表格 -->
        <div class="ds-wrap">
          <div style="display:flex;align-items:center;gap:10px;padding:14px 18px;border-bottom:1px solid var(--border)">
            <h3 style="margin:0;font-size:1rem;font-weight:600">训练数据集</h3>
            <span class="card-sub">· {{ trainFiles.length }} 个文件</span>
            <span style="flex:1"></span>
            <n-checkbox v-if="trainFiles.length" :checked="isAllSelected()" @update:checked="toggleSelectAll" size="small">全选</n-checkbox>
            <n-button v-if="selectedDatasets.length > 0" size="small" type="error" round @click="deleteSelectedDatasets(toast)">
              <template #icon><Trash2 :size="14" /></template>删除选中 ({{ selectedDatasets.length }})
            </n-button>
            <n-button size="small" round @click="loadTrainDatasets">
              <template #icon><RefreshCw :size="14" /></template>刷新
            </n-button>
          </div>
          <table v-if="trainFiles.length" class="ds-table">
            <thead>
              <tr>
                <th style="width:40px"></th>
                <th>数据集名称</th>
                <th style="width:120px">大小</th>
                <th style="width:120px">状态</th>
                <th style="width:160px">操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="f in trainFiles" :key="f" :class="{ selected: selectedDatasets.includes(f) }">
                <td>
                  <n-checkbox :checked="selectedDatasets.includes(f)" @update:checked="toggleDataset(f)" />
                </td>
                <td>
                  <span class="ds-name">
                    <span class="ds-ic" style="background:linear-gradient(135deg,var(--chart-1),var(--chart-2))">
                      <PackageIcon :size="14" />
                    </span>
                    {{ f }}
                  </span>
                </td>
                <td><span class="ds-num">--</span></td>
                <td><span class="sc sc-ok">已就绪</span></td>
                <td>
                  <div class="ds-act">
                    <button @click="previewDataset(f)"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>查看</button>
                    <button class="danger" @click="deleteTrainFile(f)"><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 7h14M9 7V5h6v2M7 7l1 12h8l1-12"/></svg>移除</button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
          <n-empty v-else :description="t('train_no_data')" style="padding:40px 0" />

          <!-- 预览 -->
          <div v-if="trainPreview" class="ds-preview">
            <div class="ds-preview-head">{{ t('dataset_preview') }} ({{ trainPreview.count || 0 }} {{ t('samples') }})</div>
            <div v-for="(s, i) in (trainPreview.samples || [])" :key="i" class="preview-sample">
              <div class="preview-label">{{ t('instruction') }}</div>
              <div class="preview-text">{{ s.instruction }}</div>
              <div class="preview-label">{{ t('output') }}</div>
              <div class="preview-text">{{ s.output }}</div>
            </div>
          </div>
        </div>
      </section>

      <!-- ═══ Tab 4 · 日志 ═══ -->
      <section class="tab-panel" :class="{ active: activeTab === 'logs' }">
        <div v-if="trainLog" class="log-panel">
          <div class="log-head">
            <div class="log-dots">
              <i style="background:var(--destructive)"></i>
              <i style="background:var(--warning, #eab308)"></i>
              <i style="background:var(--chart-2)"></i>
            </div>
            <span class="log-title">training.log — {{ taijiModelInfo.size || '态极模型' }}</span>
            <span class="log-spacer"></span>
            <span class="log-tag">实时 · tail -f</span>
            <n-button size="tiny" quaternary round class="log-clear-btn" @click="clearTrainLog">
              <template #icon><Trash2 :size="14" /></template>清空
            </n-button>
          </div>
          <pre class="log-body" ref="trainLogRef">{{ trainLog }}</pre>
        </div>
        <div v-else class="tk-card">
          <div class="log-empty">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="width:48px;height:48px;color:var(--muted-foreground);opacity:0.5"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M8 10h8M8 14h5"/></svg>
            <p>暂无训练日志</p>
            <span class="card-sub">训练开始后将实时输出日志</span>
          </div>
        </div>
      </section>

      <!-- 训练控制（悬浮底栏样式，便于随时操作） -->
      <div v-if="trainState === 'running' || trainState === 'paused'" class="train-ctrl-bar">
        <div class="ctrl-row">
          <n-button v-if="trainState === 'running'" type="warning" round @click="pauseTraining(toast)">
            <template #icon><Pause :size="14" /></template>{{ t('pause_training') }}
          </n-button>
          <n-button v-if="trainState === 'paused'" type="primary" round @click="resumeTraining(toast)">
            <template #icon><Play :size="14" /></template>{{ t('resume_training') }}
          </n-button>
          <n-button type="error" round @click="stopTraining(toast)">
            <template #icon><StopCircle :size="14" /></template>{{ t('stop_training') }}
          </n-button>
          <n-button v-if="trainState === 'idle' && pendingCheckpoints.length > 0" type="info" round @click="forcePublish(toast, $confirm)">
            <template #icon><PackageIcon :size="14" /></template>强制发布
          </n-button>
        </div>
      </div>

      <!-- 发布进度 -->
      <div v-if="publishingState === 'publishing'" class="tk-card" style="margin-top:18px">
        <div class="card-head">
          <h3><svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" style="width:17px;height:17px;color:var(--primary)"><path d="M5 4h11l3 3v13H5z"/><path d="M8 4v5h7M8 14h8"/></svg>发布进度</h3>
          <n-tag round>{{ trainProgress }}%</n-tag>
        </div>
        <n-progress type="line" :percentage="trainProgress" :processing="true" />
        <n-text depth="3" class="progress-desc">{{ trainProgressDesc }}</n-text>
        <n-button type="error" round style="margin-top:12px" @click="cancelPublish()">
          <template #icon><Square :size="14" /></template>取消发布
        </n-button>
      </div>

      <!-- 硬件诊断 -->
      <n-alert v-if="trainDevice.message" type="info" class="hw-alert" style="margin-top:18px">
        <template #icon><Monitor :size="16" /></template>
        {{ trainDevice.message }}
      </n-alert>
    </div>
  </section>
</template>

<script setup>
import { Monitor, Clock, Hourglass, Activity, Zap, BookOpen, GraduationCap, Trash2, Upload, Package as PackageIcon, RefreshCw, Gamepad2, Play, Pause, Square, StopCircle, TrendingUp, FileText as FileTextIcon, Info } from 'lucide-vue-next';

import { inject, watch, nextTick, ref } from 'vue';
import FileUploadQueue from '../components/FileUploadQueue.vue';
import { useApi } from '../composables/useApi.js';
import {
  trainState, trainLog, trainLoss, trainFiles,
  selectedDatasets, trainPreview,
  publishingState, trainProgress, trainProgressDesc,
  pendingCheckpoints, trainMetrics, trainDevice,
  lossCanvasRef, trainLogRef, fmtTime,
  clearTrainLog,
  loadTrainDatasets, previewDataset, deleteTrainFile, deleteSelectedDatasets,
  toggleSelectAll, toggleDataset, isAllSelected,
  pauseTraining, resumeTraining, stopTraining,
  loadCheckpoints, resumeFromCheckpoint,
  publishModel, forcePublish, exportModelToGGUF,
  cancelPublish, drawLossChart,
  taijiModelInfo, taijiTrainParams,
  startTaijiTraining,
} from '../composables/useTraining.js';

const toast = inject('toast');
const $confirm = inject('$confirm');
const { t } = useApi();

// 标签页状态
const activeTab = ref('overview');

// 保存检查点（占位，实际由后端处理）
const saveCheckpoint = () => {
  toast?.success?.('检查点保存请求已发送');
};

// 应用并重启训练
const applyAndRestart = () => {
  if (trainState.value === 'running' || trainState.value === 'paused') {
    stopTraining(toast);
    setTimeout(() => startTaijiTraining(toast), 800);
  } else {
    startTaijiTraining(toast);
  }
};

// 恢复默认值
const resetDefaults = () => {
  Object.assign(taijiTrainParams, {
    num_epochs: 5, batch_size: 4, learning_rate: 1e-4,
    max_length: 512, save_steps: 50, log_steps: 5,
    keep_checkpoints: 3,
  });
  toast?.success?.('已恢复默认超参数');
};

watch(() => trainLoss.value.length, () => {
  nextTick(() => drawLossChart());
});

loadTrainDatasets();
loadCheckpoints();
</script>

<style scoped>
/* ═══ 训练视图 · 画布设计语言对齐 ═══ */
.training-view {
  --chart-1: var(--chart-1, #557fff);
  --chart-2: var(--chart-2, #0065fd);
  --chart-3: var(--chart-3, #0057da);
  --chart-4: var(--chart-4, #0043ad);
  --chart-5: var(--chart-5, #002e7d);
  --font-mono: "JetBrains Mono", "Fira Code", "Consolas", "Courier New", monospace;
  --destructive: var(--danger, #ef4444);
  --warning: var(--warning, #eab308);
}

/* ===== 顶栏标题 ===== */
.view-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 18px;
}
.header-left { display: flex; flex-direction: column; gap: 3px; }
.header-left h2 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 700;
  color: var(--foreground, var(--text));
  letter-spacing: -0.01em;
}
.header-sub {
  font-size: 0.78rem;
  color: var(--muted-foreground, var(--text-muted));
}
.header-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: none;
}

/* ===== 标签页（匹配画布） ===== */
.tabs {
  display: flex;
  gap: 2px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 22px;
}
.tab {
  appearance: none;
  border: 0;
  background: transparent;
  color: var(--muted-foreground, var(--text-muted));
  padding: 11px 18px;
  font: 600 0.88rem/1 var(--font-sans, inherit);
  cursor: pointer;
  position: relative;
  transition: color 0.15s;
}
.tab:hover { color: var(--foreground, var(--text)); }
.tab.active { color: var(--primary); }
.tab.active::after {
  content: "";
  position: absolute;
  left: 14px;
  right: 14px;
  bottom: -1px;
  height: 2px;
  background: var(--primary);
  border-radius: 2px;
}

/* ===== Tab Panel ===== */
.tab-panel { display: none; }
.tab-panel.active {
  display: block;
  animation: tkFade 0.22s ease;
}
@keyframes tkFade {
  from { opacity: 0; transform: translateY(5px); }
  to { opacity: 1; transform: none; }
}

/* ===== 卡片基础（匹配画布 tk-card） ===== */
.tk-card {
  background: var(--card, var(--bg-card));
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) * 0.7);
  padding: 22px;
  transition: border-color 0.16s ease;
}
.tk-card:hover { border-color: color-mix(in srgb, var(--primary) 20%, var(--border)); }
.tk-card .card-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}
.tk-card h3 {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--foreground, var(--text));
  display: flex;
  align-items: center;
  gap: 8px;
}
.tk-card .card-sub {
  color: var(--muted-foreground, var(--text-muted));
  font-size: 0.78rem;
}

/* ===== 进度英雄卡（匹配画布） ===== */
.progress-hero {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 24px;
  flex-wrap: wrap;
}
.progress-hero .pct {
  font-size: 2.7rem;
  font-weight: 700;
  line-height: 1;
  color: var(--foreground, var(--text));
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em;
}
.progress-hero .pct .unit {
  font-size: 1.1rem;
  color: var(--muted-foreground, var(--text-muted));
  font-weight: 500;
  margin-left: 2px;
}
.right-meta { display: flex; align-items: center; gap: 18px; }

/* 状态胶囊 */
.status-chip-run {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 5px 12px;
  border-radius: 999px;
  font: 600 0.76rem var(--font-sans, inherit);
  background: color-mix(in srgb, var(--chart-1) 16%, transparent);
  color: var(--chart-1);
}
.status-chip-run::before {
  content: "";
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
  animation: tkPulse 1.4s ease-in-out infinite;
}
.status-chip-paused {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 5px 12px;
  border-radius: 999px;
  font: 600 0.76rem var(--font-sans, inherit);
  background: color-mix(in srgb, var(--warning) 16%, transparent);
  color: var(--warning);
}
.status-chip-paused::before {
  content: "";
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
}
.status-chip-idle {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 5px 12px;
  border-radius: 999px;
  font: 600 0.76rem var(--font-sans, inherit);
  background: color-mix(in srgb, var(--muted-foreground, var(--text-muted)) 16%, transparent);
  color: var(--muted-foreground, var(--text-muted));
}
.status-chip-idle::before {
  content: "";
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
}
@keyframes tkPulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.35; }
}

/* 进度条 */
.progress-bar {
  height: 12px;
  border-radius: 999px;
  background: var(--muted, var(--bg-muted));
  overflow: hidden;
  margin-top: 16px;
  position: relative;
}
.progress-bar .fill {
  height: 100%;
  background: linear-gradient(90deg, var(--chart-1), var(--chart-2));
  border-radius: 999px;
  transition: width 0.4s ease;
}
.progress-bar .fill.paused {
  background: linear-gradient(90deg, var(--warning), #f97316);
}

/* 进度元信息 */
.progress-meta {
  display: flex;
  gap: 30px;
  color: var(--muted-foreground, var(--text-muted));
  font-size: 0.82rem;
  margin-top: 14px;
  flex-wrap: wrap;
}
.progress-meta .pm-item {
  display: flex;
  align-items: center;
  gap: 6px;
}
.progress-meta b {
  color: var(--foreground, var(--text));
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.progress-meta .pm-icon {
  width: 15px;
  height: 15px;
  opacity: 0.7;
}

/* ===== 双列图表 ===== */
.charts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 18px;
  margin: 18px 0;
}
.chart-card .chart-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}
.chart-card h4 {
  margin: 0;
  font-size: 0.92rem;
  font-weight: 600;
  color: var(--foreground, var(--text));
}
.chart-card .legend {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.74rem;
  color: var(--muted-foreground, var(--text-muted));
}
.chart-card .legend i {
  width: 11px;
  height: 11px;
  border-radius: 3px;
  display: inline-block;
}
.chart-svg {
  width: 100%;
  height: auto;
  display: block;
}
.axis-text {
  font: 600 9px var(--font-mono);
  fill: var(--muted-foreground, var(--text-muted));
}
.grid-line {
  stroke: var(--border);
  stroke-width: 1;
}
.axis-line {
  stroke: var(--border);
  stroke-width: 1.2;
}
.chart-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 170px;
  color: var(--muted-foreground, var(--text-muted));
  font-size: 0.82rem;
  background: color-mix(in srgb, var(--muted) 40%, transparent);
  border-radius: calc(var(--radius) * 0.5);
}

/* Loss canvas 适配 */
.loss-canvas {
  width: 100%;
  height: auto;
  max-height: 200px;
  display: block;
}

/* ===== 指标卡（匹配画布 4 列） ===== */
.metrics-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-top: 18px;
}
.metric-card {
  background: var(--card, var(--bg-card));
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) * 0.6);
  padding: 16px 18px;
  transition: border-color 0.16s ease, transform 0.16s ease;
}
.metric-card:hover {
  border-color: color-mix(in srgb, var(--primary) 35%, var(--border));
  transform: translateY(-2px);
}
.metric-card .m-label {
  font-size: 0.76rem;
  color: var(--muted-foreground, var(--text-muted));
  display: flex;
  align-items: center;
  gap: 7px;
}
.metric-card .m-label .ic {
  width: 15px;
  height: 15px;
  color: var(--chart-2);
}
.metric-card .m-value {
  font-size: 1.7rem;
  font-weight: 700;
  margin-top: 8px;
  color: var(--foreground, var(--text));
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.01em;
}
.metric-card .m-trend {
  font-size: 0.74rem;
  margin-top: 5px;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.m-trend.up { color: var(--chart-2); }
.m-trend.down { color: var(--chart-1); }
.m-trend.flat { color: var(--muted-foreground, var(--text-muted)); }

/* ===== 超参数网格（匹配画布 8 字段） ===== */
.hp-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px 32px;
}
.hp-field {
  display: flex;
  flex-direction: column;
  gap: 7px;
}
.hp-field .hp-label {
  font-size: 0.84rem;
  font-weight: 600;
  color: var(--foreground, var(--text));
  display: flex;
  align-items: center;
  gap: 7px;
}
.hp-field .hp-label .ic {
  width: 15px;
  height: 15px;
  color: var(--muted-foreground, var(--text-muted));
}
.hp-field .hp-desc {
  font-size: 0.74rem;
  color: var(--muted-foreground, var(--text-muted));
  line-height: 1.4;
}

/* Naive UI 输入框适配 */
.hp-field :deep(.n-input-number) {
  width: 100%;
}
.hp-field :deep(.n-input) {
  border-radius: calc(var(--radius) * 0.42);
}
.hp-field :deep(.n-input .n-input__border),
.hp-field :deep(.n-input .n-input__state-border) {
  border-radius: calc(var(--radius) * 0.42);
}
.hp-field :deep(.n-input .n-input__input),
.hp-field :deep(.n-input .n-input__textarea-el) {
  font-family: var(--font-mono);
  font-size: 0.88rem;
}

/* 超参数操作栏 */
.hp-actions {
  display: flex;
  gap: 10px;
  margin-top: 26px;
  padding-top: 20px;
  border-top: 1px solid var(--border);
  align-items: center;
}

/* ===== 数据集表格（匹配画布） ===== */
.ds-wrap {
  background: var(--card, var(--bg-card));
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) * 0.7);
  overflow: hidden;
}
.ds-table {
  width: 100%;
  border-collapse: collapse;
}
.ds-table th {
  text-align: left;
  font: 600 0.72rem/1 var(--font-mono);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted-foreground, var(--text-muted));
  padding: 12px 18px;
  border-bottom: 1px solid var(--border);
  background: color-mix(in srgb, var(--muted) 35%, transparent);
}
.ds-table td {
  padding: 14px 18px;
  border-bottom: 1px solid color-mix(in srgb, var(--border) 55%, transparent);
  font-size: 0.86rem;
  color: var(--foreground, var(--text));
}
.ds-table tr:last-child td { border-bottom: 0; }
.ds-table tbody tr { transition: background 0.12s; }
.ds-table tbody tr:hover {
  background: color-mix(in srgb, var(--accent, var(--primary-light)) 14%, transparent);
}
.ds-table tbody tr.selected {
  background: color-mix(in srgb, var(--primary) 8%, transparent);
}
.ds-name {
  display: flex;
  align-items: center;
  gap: 11px;
  font-weight: 500;
}
.ds-ic {
  width: 30px;
  height: 30px;
  border-radius: 8px;
  display: grid;
  place-items: center;
  color: var(--primary-foreground, #fff);
  flex: none;
  font-size: 0.9rem;
}
.ds-ic :deep(svg) {
  width: 15px;
  height: 15px;
}
.ds-num {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  color: var(--foreground, var(--text));
}
.sc {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px;
  border-radius: 999px;
  font: 500 0.74rem var(--font-sans, inherit);
}
.sc::before {
  content: "";
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
}
.sc-run {
  background: color-mix(in srgb, var(--chart-1) 16%, transparent);
  color: var(--chart-1);
}
.sc-ok {
  background: color-mix(in srgb, var(--chart-2) 16%, transparent);
  color: var(--chart-2);
}
.sc-wait {
  background: color-mix(in srgb, var(--muted-foreground, var(--text-muted)) 16%, transparent);
  color: var(--muted-foreground, var(--text-muted));
}
.ds-act {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.ds-act button {
  border: 0;
  background: transparent;
  color: var(--muted-foreground, var(--text-muted));
  cursor: pointer;
  padding: 5px 9px;
  border-radius: 7px;
  font-size: 0.78rem;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  transition: background 0.14s ease, color 0.14s ease;
}
.ds-act button .ic {
  width: 14px;
  height: 14px;
}
.ds-act button:hover {
  background: var(--muted, var(--bg-muted));
  color: var(--foreground, var(--text));
}
.ds-act button.danger:hover {
  color: var(--destructive, var(--danger));
}

/* 数据集预览 */
.ds-preview {
  padding: 18px;
  border-top: 1px solid var(--border);
  background: color-mix(in srgb, var(--muted) 25%, transparent);
}
.ds-preview-head {
  font-size: 0.84rem;
  font-weight: 600;
  color: var(--foreground, var(--text));
  margin-bottom: 10px;
}
.preview-sample {
  padding: 12px 14px;
  background: var(--card, var(--bg-card));
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) * 0.5);
  margin-bottom: 8px;
}
.preview-label {
  font-size: 0.74rem;
  color: var(--primary);
  font-weight: 600;
  margin-bottom: 4px;
}
.preview-text {
  font-size: 0.84rem;
  color: var(--muted-foreground, var(--text-secondary));
  word-break: break-all;
  line-height: 1.6;
  margin-bottom: 8px;
}
.preview-text:last-child { margin-bottom: 0; }

/* ===== 日志面板（匹配画布浅色终端） ===== */
.log-panel {
  background: var(--card, var(--bg-card));
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) * 0.7);
  overflow: hidden;
}
.log-head {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  background: color-mix(in srgb, var(--muted) 45%, var(--card, var(--bg-card)));
}
.log-dots { display: flex; gap: 6px; }
.log-dots i {
  width: 11px;
  height: 11px;
  border-radius: 50%;
  display: block;
}
.log-head .log-title {
  font-size: 0.82rem;
  font-weight: 600;
  font-family: var(--font-mono);
  color: var(--foreground, var(--text));
}
.log-head .log-spacer { flex: 1; }
.log-head .log-tag {
  font: 600 0.7rem var(--font-mono);
  color: var(--muted-foreground, var(--text-muted));
  padding: 3px 9px;
  border: 1px solid var(--border);
  border-radius: 999px;
}
.log-clear-btn {
  margin-left: 8px;
}
.log-body {
  font-family: var(--font-mono);
  font-size: 0.8rem;
  line-height: 1.75;
  padding: 16px 20px;
  overflow-x: auto;
  background: color-mix(in srgb, var(--background, var(--bg)) 86%, var(--card, var(--bg-card)));
  color: var(--foreground, var(--text));
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 400px;
  overflow-y: auto;
  margin: 0;
}

.log-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 60px 20px;
  text-align: center;
}
.log-empty p {
  margin: 0;
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--foreground, var(--text));
}

/* ===== 检查点列表 ===== */
.ckpt-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.ckpt-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-left: 3px solid var(--primary);
  border-radius: calc(var(--radius) * 0.5);
  background: var(--card, var(--bg-card));
  transition: border-color 0.16s ease, background 0.16s ease, transform 0.16s ease;
}
.ckpt-item:hover {
  border-color: color-mix(in srgb, var(--primary) 30%, var(--border));
  border-left-color: var(--primary);
  background: color-mix(in srgb, var(--primary) 5%, transparent);
  transform: translateX(2px);
}
.ckpt-ic {
  width: 36px;
  height: 36px;
  flex-shrink: 0;
  display: grid;
  place-items: center;
  border-radius: 10px;
  background: var(--primary-light, color-mix(in srgb, var(--primary) 12%, transparent));
  color: var(--primary);
}
.ckpt-body { flex: 1; min-width: 0; }
.ckpt-name {
  font-size: 0.86rem;
  font-weight: 600;
  color: var(--foreground, var(--text));
  font-family: var(--font-mono);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ckpt-meta {
  font-size: 0.76rem;
  color: var(--muted-foreground, var(--text-muted));
  margin-top: 5px;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.ckpt-tag {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border-radius: 6px;
  background: var(--muted, var(--bg-muted));
}
.ckpt-meta b {
  color: var(--foreground, var(--text));
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}

/* ===== 发布描述 ===== */
.publish-desc {
  display: block;
  font-size: 0.84rem;
  color: var(--muted-foreground, var(--text-muted));
  line-height: 1.5;
}

/* ===== 控制按钮行 ===== */
.ctrl-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}

/* ===== 训练控制底栏 ===== */
.train-ctrl-bar {
  position: sticky;
  bottom: 0;
  margin-top: 24px;
  padding: 14px 18px;
  background: var(--card, var(--bg-card));
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) * 0.7);
  box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.06);
  z-index: 10;
}

/* ===== 硬件诊断 ===== */
.hw-alert {
  border-radius: calc(var(--radius) * 0.6);
}

/* ===== 进度描述 ===== */
.progress-desc {
  display: block;
  font-size: 0.82rem;
  color: var(--muted-foreground, var(--text-muted));
  margin-top: 8px;
  word-break: break-all;
  line-height: 1.5;
}

/* ===== 响应式（匹配画布断点） ===== */
@media (max-width: 1180px) {
  .charts-grid, .hp-grid {
    grid-template-columns: 1fr;
  }
  .metrics-row {
    grid-template-columns: repeat(2, 1fr);
  }
}
@media (max-width: 720px) {
  .metrics-row {
    grid-template-columns: 1fr;
  }
  .progress-hero {
    flex-direction: column;
    align-items: flex-start;
  }
  .view-header {
    flex-direction: column;
    align-items: flex-start;
  }
  .header-actions {
    width: 100%;
    justify-content: flex-end;
  }
}

@media (prefers-reduced-motion: reduce) {
  .tab-panel.active { animation: none; }
  .status-chip-run::before { animation: none; }
  .metric-card:hover { transform: none; }
  .ckpt-item:hover { transform: none; }
}
</style>
