<template>
  <section class="dedicated-view">
    <div class="view-header">
      <h2>模型训练</h2>
      <n-tag v-if="taijiModelInfo.size" type="info" size="small">
        {{ taijiModelInfo.size }} · {{ taijiModelInfo.config?.num_hidden_layers }}层 · {{ taijiModelInfo.config?.hidden_size }}维
      </n-tag>
    </div>
    <div class="view-body">
      <!-- 数据上传 -->
      <n-card size="small" :bordered="false" style="margin-bottom:16px;">
        <template #header><div style="display:flex;align-items:center;gap:8px;"><Upload :size="18" /> {{ t('train_upload') }}</div></template>
        <FileUploadQueue
          ref="trainUploadRef"
          upload-endpoint="/api/train/upload_dataset"
          accept=".jsonl,.json,.txt,.csv,.md,.pdf,.docx,.doc,.xlsx,.xls,.pptx,.html,.htm,.epub,.rtf,.xml,.log,.py,.js,.ts,.css,.java,.c,.cpp,.sh,.sql,.png,.jpg,.jpeg,.bmp,.gif,.webp,.tiff,.tif"
          icon="BarChart2" title="训练数据上传" upload-icon="Download"
          :drop-text="t('train_upload')" :accept-hint="t('train_support')"
          success-text="✅ 数据集上传成功"
          @all-uploaded="loadTrainDatasets"
        />
      </n-card>

      <!-- 数据集列表 -->
      <n-card size="small" :bordered="false" style="margin-bottom:16px;">
        <template #header>
          <div style="display:flex;align-items:center;justify-content:space-between;">
            <div style="display:flex;align-items:center;gap:8px;"><PackageIcon :size="18" /> {{ t('train_datasets') }}</div>
            <n-space>
              <n-button size="small" @click="loadTrainDatasets"><template #icon><RefreshCw :size="14" /></template>{{ t('reload') }}</n-button>
              <n-button v-if="selectedDatasets.length > 0" size="small" type="error" @click="deleteSelectedDatasets(toast)">
                <template #icon><Trash2 :size="14" /></template>删除选中 ({{ selectedDatasets.length }})
              </n-button>
            </n-space>
          </div>
        </template>
        <div v-if="trainFiles.length">
          <n-space style="margin-bottom:8px;">
            <n-checkbox :checked="isAllSelected()" @update:checked="toggleSelectAll">{{ t('select_all') }}</n-checkbox>
            <n-text depth="3" style="font-size:12px;">已选 {{ selectedDatasets.length }}/{{ trainFiles.length }}</n-text>
          </n-space>
          <n-list bordered hoverable>
            <n-list-item v-for="f in trainFiles" :key="f">
              <n-checkbox :checked="selectedDatasets.includes(f)" @update:checked="toggleDataset(f)" style="margin-right:8px;" />
              <n-ellipsis style="max-width:400px;">{{ f }}</n-ellipsis>
              <template #suffix>
                <n-space size="small">
                  <n-button size="tiny" @click="previewDataset(f)">{{ t('preview') }}</n-button>
                  <n-button size="tiny" type="error" quaternary @click="deleteTrainFile(f)"><template #icon><Trash2 :size="14" /></template></n-button>
                </n-space>
              </template>
            </n-list-item>
          </n-list>
        </div>
        <n-empty v-else :description="t('train_no_data')" />
        <!-- 预览 -->
        <n-card v-if="trainPreview" size="small" :bordered="true" style="margin-top:12px;">
          <template #header>{{ t('dataset_preview') }} ({{ trainPreview.count || 0 }} {{ t('samples') }})</template>
          <div v-for="(s, i) in (trainPreview.samples || [])" :key="i" style="margin-bottom:12px;">
            <n-text strong>{{ t('instruction') }}:</n-text>
            <n-text>{{ s.instruction }}</n-text>
            <br /><n-text strong>{{ t('output') }}:</n-text>
            <n-text>{{ s.output }}</n-text>
          </div>
        </n-card>
      </n-card>

      <!-- 训练参数 -->
      <n-card size="small" :bordered="false" style="margin-bottom:16px;">
        <template #header><div style="display:flex;align-items:center;gap:8px;">态极训练参数</div></template>
        <n-grid :cols="3" :x-gap="12" :y-gap="12">
          <n-gi><n-form-item label="Epochs"><n-input-number v-model:value="taijiTrainParams.num_epochs" :min="1" :max="100" /></n-form-item></n-gi>
          <n-gi><n-form-item label="Batch Size"><n-input-number v-model:value="taijiTrainParams.batch_size" :min="1" :max="32" /></n-form-item></n-gi>
          <n-gi><n-form-item label="Learning Rate"><n-input-number v-model:value="taijiTrainParams.learning_rate" :min="0.00001" :step="0.00001" /></n-form-item></n-gi>
          <n-gi><n-form-item label="Max Length"><n-input-number v-model:value="taijiTrainParams.max_length" :min="64" :max="2048" /></n-form-item></n-gi>
          <n-gi><n-form-item label="Save Steps"><n-input-number v-model:value="taijiTrainParams.save_steps" :min="10" :max="1000" /></n-form-item></n-gi>
          <n-gi><n-form-item label="Keep Checkpoints"><n-input-number v-model:value="taijiTrainParams.keep_checkpoints" :min="1" :max="10" /></n-form-item></n-gi>
        </n-grid>
      </n-card>

      <!-- 控制 -->
      <n-card size="small" :bordered="false" style="margin-bottom:16px;">
        <template #header><div style="display:flex;align-items:center;gap:8px;"><Gamepad2 :size="18" /> 控制</div></template>
        <n-space>
          <n-button v-if="trainState === 'idle' || trainState === 'completed'" type="primary" @click="startTaijiTraining(toast)"><template #icon><Zap :size="14" /></template>开始态极微调</n-button>
          <n-button v-if="trainState === 'idle' && pendingCheckpoints.length > 0" type="info" @click="resumeFromCheckpoint(toast, $confirm)"><template #icon><RefreshCw :size="14" /></template>恢复训练 ({{ pendingCheckpoints.length }})</n-button>
          <n-button v-if="trainState === 'running'" type="warning" @click="pauseTraining(toast)"><template #icon><Pause :size="14" /></template>{{ t('pause_training') }}</n-button>
          <n-button v-if="trainState === 'paused'" type="primary" @click="resumeTraining(toast)"><template #icon><Play :size="14" /></template>{{ t('resume_training') }}</n-button>
          <n-button v-if="trainState === 'running' || trainState === 'paused'" type="error" @click="stopTraining(toast)"><template #icon><StopCircle :size="14" /></template>{{ t('stop_training') }}</n-button>
          <n-button v-if="trainState === 'idle' && pendingCheckpoints.length > 0" type="info" @click="forcePublish(toast, $confirm)"><template #icon><PackageIcon :size="14" /></template>强制发布</n-button>
        </n-space>
      </n-card>

      <!-- 训练进度 -->
      <div v-if="trainState === 'running' || trainState === 'paused'">
        <n-alert v-if="trainDevice.message" type="info" style="margin-bottom:12px;">
          <template #icon><Monitor :size="16" /></template>
          {{ trainDevice.message }}
        </n-alert>
        <n-card size="small" :bordered="false" style="margin-bottom:16px;">
          <template #header><div style="display:flex;align-items:center;justify-content:space-between;"><div style="display:flex;align-items:center;gap:8px;"><TrendingUp :size="18" /> 训练进度</div><n-tag :type="trainState === 'paused' ? 'warning' : 'info'">{{ trainProgress }}%</n-tag></div></template>
          <n-progress type="line" :percentage="trainProgress" :status="trainState === 'paused' ? 'warning' : 'info'" :processing="trainState === 'running'" />
          <n-text depth="3" style="margin-top:8px;display:block;">{{ trainProgressDesc }}</n-text>
        </n-card>
        <!-- 指标 -->
        <n-grid :cols="3" :x-gap="12" :y-gap="12" style="margin-bottom:16px;">
          <n-gi><n-card size="small"><n-statistic label="已用时间"><template #prefix><Clock :size="16" /></template>{{ fmtTime(trainMetrics.elapsed) }}</n-statistic></n-card></n-gi>
          <n-gi><n-card size="small"><n-statistic label="预计剩余"><template #prefix><Hourglass :size="16" /></template>{{ fmtTime(trainMetrics.eta) }}</n-statistic></n-card></n-gi>
          <n-gi><n-card size="small"><n-statistic label="当前 Loss"><template #prefix><Activity :size="16" /></template>{{ trainMetrics.current_loss != null ? trainMetrics.current_loss.toFixed(4) : '--' }}</n-statistic></n-card></n-gi>
          <n-gi><n-card size="small"><n-statistic label="吞吐量"><template #prefix><Zap :size="16" /></template>{{ trainMetrics.samples_per_sec >= 0.005 ? (trainMetrics.samples_per_sec < 0.1 ? trainMetrics.samples_per_sec.toFixed(2) : trainMetrics.samples_per_sec.toFixed(1)) + '/s' : '--' }}</n-statistic></n-card></n-gi>
          <n-gi><n-card size="small"><n-statistic label="Epoch"><template #prefix><BookOpen :size="16" /></template>{{ trainMetrics.epoch }}/{{ trainMetrics.total_epochs }}</n-statistic></n-card></n-gi>
          <n-gi><n-card size="small"><n-statistic label="学习率"><template #prefix><GraduationCap :size="16" /></template>{{ trainMetrics.lr != null ? trainMetrics.lr.toExponential(2) : '--' }}</n-statistic></n-card></n-gi>
        </n-grid>
        <!-- Loss 曲线 -->
        <n-card v-if="trainLoss.length >= 2" size="small" :bordered="false" style="margin-bottom:16px;">
          <template #header><div style="display:flex;align-items:center;justify-content:space-between;"><div style="display:flex;align-items:center;gap:8px;"><TrendingUp :size="18" /> Loss 曲线</div><n-tag size="small">最新: {{ trainLoss[trainLoss.length-1]?.loss?.toFixed(4) || '--' }}</n-tag></div></template>
          <canvas ref="lossCanvasRef" class="loss-canvas" width="600" height="160"></canvas>
        </n-card>
        <!-- 日志 -->
        <n-card v-if="trainLog" size="small" :bordered="false">
          <template #header>
            <div style="display:flex;align-items:center;justify-content:space-between;">
              <div style="display:flex;align-items:center;gap:8px;"><FileTextIcon :size="18" /> 训练日志</div>
              <n-button size="tiny" quaternary @click="clearTrainLog"><template #icon><Trash2 :size="14" /></template>清空</n-button>
            </div>
          </template>
          <pre class="train-log-content" ref="trainLogRef">{{ trainLog }}</pre>
        </n-card>
      </div>

      <!-- 发布进度 -->
      <n-card v-if="publishingState === 'publishing'" size="small" :bordered="false" style="margin-top:12px;">
        <template #header><div style="display:flex;align-items:center;justify-content:space-between;"><div style="display:flex;align-items:center;gap:8px;"><PackageIcon :size="18" /> 发布进度</div><n-tag>{{ trainProgress }}%</n-tag></div></template>
        <n-progress type="line" :percentage="trainProgress" :processing="true" />
        <n-text depth="3" style="margin-top:8px;display:block;">{{ trainProgressDesc }}</n-text>
        <n-button type="error" style="margin-top:8px;" @click="cancelPublish()"><template #icon><Square :size="14" /></template>取消发布</n-button>
      </n-card>

      <!-- 发布与导出 -->
      <n-card v-if="trainState === 'completed' || publishingState !== 'idle'" size="small" :bordered="false" style="margin-top:16px;">
        <template #header><div style="display:flex;align-items:center;gap:8px;"><PackageIcon :size="18" /> 发布与导出</div></template>
        <n-text depth="3" style="display:block;margin-bottom:12px;">{{ t('publish_desc') }}</n-text>
        <n-space>
          <n-button type="primary" :disabled="publishingState !== 'idle'" @click="publishModel(toast)">
            {{ publishingState === 'publishing' ? '发布中...' : t('publish_model_btn') }}
          </n-button>
          <n-button type="info" :disabled="publishingState !== 'idle'" @click="exportModelToGGUF(toast, $confirm)">
            {{ publishingState === 'publishing' ? '导出中...' : t('export_gguf_btn') }}
          </n-button>
        </n-space>
      </n-card>
    </div>
  </section>
</template>

<script setup>
import { Monitor, Clock, Hourglass, Activity, Zap, BookOpen, GraduationCap, Trash2, Upload, Package as PackageIcon, RefreshCw, Gamepad2, Play, Pause, Square, StopCircle, TrendingUp, FileText as FileTextIcon } from 'lucide-vue-next';

import { inject, watch, nextTick } from 'vue';
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

watch(() => trainLoss.value.length, () => {
  nextTick(() => drawLossChart());
});

loadTrainDatasets();
loadCheckpoints();
</script>

<style scoped>
/* ===== 训练面板 ===== */
.train-dashboard {
  margin-bottom: 14px;
  padding: 18px 20px;
  background: linear-gradient(135deg, rgba(15,23,42,0.8) 0%, var(--bg-elevated) 100%);
  border-radius: var(--radius-md);
  border: 1px solid var(--primary-light);
  box-shadow: var(--shadow-md);
  -webkit-backdrop-filter: blur(12px);
  backdrop-filter: blur(12px);
}

.train-log-panel {
  margin-bottom: 14px;
  padding: 16px 18px;
  background: rgba(10,14,26,0.8);
  border-radius: var(--radius-md);
  border: 1px solid rgba(48,54,61,0.5);
  box-shadow: var(--shadow-md);
}

.train-log-content {
  font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', 'Courier New', monospace;
  font-size: 0.8rem;
  color: #c9d1d9;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 300px;
  overflow-y: auto;
  margin: 0;
}

/* ===== 训练面板辅助 ===== */
.train-dataset-actions { display: flex; gap: 8px; }
.train-select-row { display: flex; gap: 10px; margin-bottom: 10px; align-items: center; }
.train-controls { display: flex; gap: 10px; flex-wrap: wrap; }
.train-log-header { background: transparent; justify-content: space-between; }
.train-publish-actions { display: flex; gap: 10px; margin-top: 10px; }
.train-preview-header { background: transparent; padding: 10px 0; }

/* ===== 训练预设按钮 ===== */
.preset-buttons { display: flex; flex-wrap: wrap; gap: 10px; }

.preset-btn {
  padding: 10px 18px;
  border: 1px solid var(--primary-light);
  border-radius: var(--radius-sm);
  background: var(--bg-card);
  cursor: pointer;
  font-size: 0.86rem;
  color: var(--text-secondary);
  transition: var(--transition);
  font-family: inherit;
}

.preset-btn:hover { border-color: var(--primary); color: var(--text); background: var(--primary-subtle); }

.preset-btn.active {
  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
  color: white;
  border-color: transparent;
  box-shadow: 0 3px 12px var(--border-strong);
}

/* ===== 训练数据集 ===== */
.train-files-list { display: flex; flex-direction: column; gap: 5px; margin-bottom: 10px; }

.train-file-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background: var(--bg-card);
  border: 1px solid var(--primary-light);
  border-radius: var(--radius-sm);
  font-size: 0.86rem;
  transition: var(--transition-fast);
}

.train-file-item:hover { border-color: var(--border-strong); }
.train-file-item.selected { border-color: var(--primary); background: var(--primary-subtle); }

/* 数据集预览 */
.train-preview-panel { margin-top: 14px; }
.preview-sample {
  padding: 10px 12px;
  background: var(--bg-card);
  border: 1px solid var(--primary-light);
  border-radius: var(--radius-sm);
  margin-bottom: 8px;
  transition: var(--transition-fast);
}

.preview-sample:hover { border-color: var(--border-strong); }
.preview-label { font-size: 0.74rem; color: var(--primary); font-weight: 600; margin-bottom: 4px; }
.preview-text { font-size: 0.84rem; color: var(--text-secondary); margin-bottom: 8px; word-break: break-all; line-height: 1.6; }

/* ===== 当前模型显示 ===== */
.current-model-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }

.model-type-badge {
  font-size: 0.76rem;
  padding: 4px 12px;
  border-radius: 14px;
  font-weight: 600;
  white-space: nowrap;
  letter-spacing: 0.02em;
}

.model-type-badge.gguf { background: rgba(37,99,235,0.1); color: #60a5fa; border: 1px solid rgba(37,99,235,0.2); }
.model-type-badge.huggingface { background: rgba(219,39,119,0.1); color: #f472b6; border: 1px solid rgba(219,39,119,0.2); }
.model-type-badge.self { background: var(--primary-light); color: #a5b4fc; border: 1px solid var(--border-strong); }

.model-path-text { font-size: 0.86rem; color: var(--text-secondary); word-break: break-all; }

.pending-model-row {
  margin-top: 10px;
  padding: 8px 12px;
  background: rgba(245,158,11,0.06);
  border: 1px solid rgba(245,158,11,0.2);
  border-radius: var(--radius-sm);
  font-size: 0.82rem;
  color: var(--warning);
}

/* ===== Danger按钮 ===== */
.danger-btn {
  background: linear-gradient(135deg, rgba(239,68,68,0.9) 0%, rgba(220,38,38,0.9) 100%);
  color: white;
  border: none;
  padding: 10px 18px;
  border-radius: var(--radius-sm);
  font-size: 0.86rem;
  cursor: pointer;
  font-weight: 500;
  font-family: inherit;
  transition: var(--transition);
  box-shadow: 0 3px 12px rgba(239,68,68,0.25);
  position: relative;
  overflow: hidden;
}

.danger-btn:hover { opacity: 0.9; transform: translateY(-1px); box-shadow: 0 5px 20px rgba(239,68,68,0.35); }
.danger-btn:active { transform: translateY(0) scale(0.97); }

/* ===== 训练进度条 ===== */
.progress-bar-track {
  width: 100%;
  height: 20px;
  background: var(--bg-card);
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid var(--primary-light);
}

.progress-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--primary) 0%, rgba(245,158,11,0.8) 100%);
  background-size: 200% 100%;
  animation: progressShine 2.5s linear infinite;
  border-radius: 10px;
  transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  min-width: 2px;
  position: relative;
  box-shadow: 0 0 12px var(--border-strong);
}

@keyframes progressShine {
  0% { background-position: 100% 0; }
  100% { background-position: -100% 0; }
}

.progress-bar-fill.paused {
  background: linear-gradient(135deg, rgba(245,158,11,0.8) 0%, rgba(230,126,34,0.8) 100%);
  animation: progressPulse 2s ease-in-out infinite;
}

@keyframes progressPulse {
  0%, 100% { opacity: 1; box-shadow: 0 0 8px rgba(245,158,11,0.3); }
  50% { opacity: 0.7; box-shadow: 0 0 16px rgba(245,158,11,0.5); }
}

.progress-bar-text {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  font-size: 0.8rem;
  font-weight: 700;
  color: white;
  text-shadow: 0 1px 3px rgba(0,0,0,0.4);
  white-space: nowrap;
}

/* 训练进度百分比 */
.train-progress-pct {
  color: var(--primary);
  font-weight: 700;
  font-size: 1.4rem;
  font-variant-numeric: tabular-nums;
  transition: color 0.3s;
  text-shadow: 0 0 8px rgba(245,158,11,0.2);
}

.train-progress-pct.paused { color: var(--warning); }

/* 训练状态描述 */
.train-status-desc {
  font-size: 0.84rem;
  color: var(--text-secondary);
  margin: 10px 0 0;
  word-break: break-all;
  line-height: 1.5;
}

/* ===== 训练指标卡片 ===== */
.train-metrics-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-bottom: 16px;
}

.train-metric-card {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 14px 16px;
  background: var(--bg-card);
  border: 1px solid var(--primary-light);
  border-left: 3px solid var(--primary);
  border-radius: var(--radius-sm);
  transition: var(--transition-fast);
  -webkit-backdrop-filter: blur(8px);
  backdrop-filter: blur(8px);
}

.train-metric-card:hover {
  border-color: var(--border-strong);
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
}

.train-metric-card:nth-child(2) { border-left-color: var(--success); }
.train-metric-card:nth-child(3) { border-left-color: var(--warning); }

.metric-icon { font-size: 1.3rem; flex-shrink: 0; opacity: 0.8; filter: drop-shadow(0 0 4px var(--border-strong)); }
.metric-info { display: flex; flex-direction: column; min-width: 0; }

.metric-value {
  font-size: 1rem;
  font-weight: 700;
  color: var(--text);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}

.metric-highlight { color: var(--primary); }
.metric-label { font-size: 0.74rem; color: var(--text-muted); margin-top: 2px; }

/* ===== Loss迷你图 ===== */
.loss-canvas {
  width: 100%;
  height: 170px;
  border-radius: var(--radius-sm);
  background: var(--bg-card);
  border: 1px solid var(--primary-light);
  display: block;
}

.loss-latest { color: var(--warning); font-size: 0.84rem; font-weight: 600; font-variant-numeric: tabular-nums; }

/* ===== 训练进度整体 ===== */
.train-progress-area { animation: fadeSlideIn 0.4s ease; }

@keyframes fadeSlideIn {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}

/* 硬件诊断横幅 */
.hardware-diag-banner {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 12px 16px;
  margin-bottom: 12px;
  background: linear-gradient(135deg, var(--primary-subtle), var(--primary-subtle));
  border: 1px solid var(--primary-light);
  border-radius: var(--radius-sm);
  font-size: 0.88rem;
  color: var(--text-secondary);
  animation: fadeSlideIn 0.3s ease;
  -webkit-backdrop-filter: blur(8px);
  backdrop-filter: blur(8px);
}

.hw-diag-icon { font-size: 1.2rem; flex-shrink: 0; padding-top: 2px; filter: drop-shadow(0 0 4px var(--border-strong)); }
.hw-diag-text { line-height: 1.7; white-space: pre-line; }

/* 数据集checkbox */
.train-file-item input[type="checkbox"] {
  accent-color: var(--primary);
  width: 16px;
  height: 16px;
}

/* ===== 训练按钮 ===== */
.train-btn {
  border: none;
  padding: 12px 28px;
  border-radius: var(--radius-sm);
  font-weight: 600;
  cursor: pointer;
  transition: var(--transition);
  font-size: 0.9rem;
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.train-btn.primary { background: linear-gradient(135deg, rgba(16,185,129,0.9) 0%, rgba(5,150,105,0.9) 100%); color: white; box-shadow: 0 4px 14px rgba(16,185,129,0.25); border: 1px solid rgba(16,185,129,0.3); }
.train-btn.danger { background: linear-gradient(135deg, rgba(239,68,68,0.9) 0%, rgba(220,38,38,0.9) 100%); color: white; box-shadow: 0 4px 14px rgba(239,68,68,0.25); border: 1px solid rgba(239,68,68,0.3); }

.train-btn.primary:hover { opacity: 0.9; transform: translateY(-1px); box-shadow: 0 6px 20px rgba(16,185,129,0.35); }
.train-btn.danger:hover { opacity: 0.9; transform: translateY(-1px); box-shadow: 0 6px 20px rgba(239,68,68,0.35); }
.train-btn:active { transform: translateY(0) scale(0.97); }
</style>
