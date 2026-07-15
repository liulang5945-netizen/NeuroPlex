
/**
 * 微调训练相关状态和逻辑
 * 从 App.vue 拆出，减轻主文件臃肿
 *
 * 支持两种训练模式：
 * 1. 普通模式（HuggingFace/GGUF）— LoRA 微调，走 /api/train/stream
 * 2. 态极模式（ModelSelf）— 原生微调，走 /api/taiji/train
 *
 * 当加载态极模型时，自动进入态极模式，显示生命活动面板。
 * 普通模型下不触发任何态极相关内容。
 */
import { ref, reactive, nextTick } from 'vue';
import { API_BASE, authFetch } from './apiClient.js';

// ===== 导出状态 =====

export const trainState = ref('idle');            // idle | running | paused | completed
export const trainLog = ref('');
export const trainLoss = ref([]);
export const trainPreset = ref('standard');
export const trainFiles = ref([]);
export const selectedDatasets = ref([]);
export const trainPreview = ref(null);
export const trainParams = reactive({ lora_r: 16, lora_alpha: 32, epochs: 3, learning_rate: 0.0002 });
export let trainAbortController = null;
export let trainReader = null;
export const publishingState = ref('idle');
export const trainProgress = ref(0);
export const trainProgressDesc = ref('');
export const pendingCheckpoints = ref([]);

export const trainMetrics = reactive({
  elapsed: 0, eta: null, lr: null, epoch: 0, total_epochs: 0,
  grad_norm: null, samples_per_sec: 0, total_steps: 0, current_loss: null,
});

export const trainDevice = reactive({
  device_type: '', device_name: '', gpu_name: null,
  gpu_memory_gb: null, ram_gb: null, message: '',
});

export const lossCanvasRef = ref(null);
export const trainLogRef = ref(null);

// ===== 态极模式状态 =====

export const isTaijiModel = ref(false);           // 当前是否为态极模型
export const taijiModelInfo = reactive({           // 态极模型信息
  size: '', parameters: {}, config: {},
  available_sizes: [], checkpoints: {},
});
export const taijiLifeStatus = reactive({          // 生命状态
  is_sleeping: false, last_sleep: null,
  total_sleeps: 0, auto_sleep_enabled: true,
});
export const taijiTimeline = ref([]);              // 生命活动时间线
export const taijiLifeLoading = ref(false);        // 生命活动操作中
export const taijiTrainParams = reactive({         // 态极专属训练参数
  num_epochs: 5, batch_size: 4, learning_rate: 1e-4,
  max_length: 512, save_steps: 50, log_steps: 5,
  keep_checkpoints: 3,
});

// ===== 辅助函数 =====

export const fmtTime = (s) => {
  if (s == null || !isFinite(s)) return '--';
  if (s < 0) s = 0;
  if (s < 60) return `${Math.round(s)}秒`;
  if (s < 3600) return `${Math.floor(s / 60)}分${Math.round(s % 60)}秒`;
  return `${Math.floor(s / 3600)}时${Math.floor((s % 3600) / 60)}分`;
};

export const autoScrollTrainLog = () => {
  nextTick(() => {
    if (trainLogRef.value) {
      trainLogRef.value.scrollTop = trainLogRef.value.scrollHeight;
    }
  });
};

export const clearTrainLog = () => { trainLog.value = ''; };

// ===== Preset =====

export function applyPreset(preset) {
  trainPreset.value = preset;
  const presets = {
    fast: { lora_r: 8, lora_alpha: 16, epochs: 1, learning_rate: 0.0005 },
    standard: { lora_r: 16, lora_alpha: 32, epochs: 3, learning_rate: 0.0002 },
    quality: { lora_r: 32, lora_alpha: 64, epochs: 5, learning_rate: 0.0001 },
  };
  if (presets[preset]) Object.assign(trainParams, presets[preset]);
}

// ===== 数据集管理 =====

export const isAllSelected = () => {
  return trainFiles.value.length > 0 && selectedDatasets.value.length === trainFiles.value.length;
};

export function toggleSelectAll() {
  if (isAllSelected()) {
    selectedDatasets.value = [];
  } else {
    selectedDatasets.value = [...trainFiles.value];
  }
}

export function toggleDataset(filename) {
  const idx = selectedDatasets.value.indexOf(filename);
  if (idx >= 0) selectedDatasets.value.splice(idx, 1);
  else selectedDatasets.value.push(filename);
}

export async function loadTrainDatasets() {
  try {
    const res = await authFetch(`${API_BASE}/api/train/files`);
    if (res.ok) {
      const data = await res.json();
      trainFiles.value = data.files || [];
    }
  } catch (e) { /* silent */ }
}

export async function previewDataset(filename) {
  try {
    const res = await authFetch(`${API_BASE}/api/train/preview/${encodeURIComponent(filename)}`);
    if (res.ok) trainPreview.value = await res.json();
  } catch (e) { /* console.warn(e) */ }
}

export async function deleteTrainFile(filename) {
  try {
    await authFetch(`${API_BASE}/api/train/file/${encodeURIComponent(filename)}`, { method: 'DELETE' });
    const idx = selectedDatasets.value.indexOf(filename);
    if (idx >= 0) selectedDatasets.value.splice(idx, 1);
    loadTrainDatasets();
  } catch (e) { /* silent */ }
}

export async function deleteSelectedDatasets(_toast) {
  if (selectedDatasets.value.length === 0) return;
  if (!_toast) return;
  let successCount = 0;
  for (const filename of [...selectedDatasets.value]) {
    try {
      await authFetch(`${API_BASE}/api/train/file/${encodeURIComponent(filename)}`, { method: 'DELETE' });
      const idx = selectedDatasets.value.indexOf(filename);
      if (idx >= 0) selectedDatasets.value.splice(idx, 1);
      successCount++;
    } catch (e) {
      if (_toast) _toast(`❌ 删除 ${filename} 失败: ${e.message}`, 'error');
    }
  }
  await loadTrainDatasets();
  if (_toast) _toast(`✅ 已删除 ${successCount} 个文件`, 'success');
}

// ===== 训练流程 =====

export async function startTraining(toast) {
  // 态极模式自动路由：调用态极专属训练接口
  if (isTaijiModel.value) {
    return startTaijiTraining(toast);
  }

  if (selectedDatasets.value.length === 0) { toast('⚠ 请先选择数据集', 'warning'); return; }
  trainState.value = 'running';
  trainLog.value = '';
  trainLoss.value = [];
  trainProgress.value = 0;
  trainProgressDesc.value = '⏳ 正在初始化训练环境...';
  trainAbortController = new AbortController();
  Object.assign(trainMetrics, { elapsed: 0, eta: null, lr: null, epoch: 1, total_epochs: trainParams.epochs, grad_norm: null, samples_per_sec: 0, total_steps: 0, current_loss: null });
  Object.assign(trainDevice, { device_type: '', device_name: '', gpu_name: null, gpu_memory_gb: null, ram_gb: null, message: '' });

  const body = {
    dataset: selectedDatasets.value[0],
    datasets: [...selectedDatasets.value],
    lora_r: trainParams.lora_r,
    lora_alpha: trainParams.lora_alpha,
    epochs: trainParams.epochs,
    learning_rate: trainParams.learning_rate,
    batch_size: 4,
  };

  try {
    const res = await authFetch(`${API_BASE}/api/train/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: trainAbortController.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    trainReader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await trainReader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const payload = line.slice(6);
          if (payload === '[DONE]') {
            if (trainState.value === 'running') trainState.value = 'completed';
            break;
          }
          try {
            const evt = JSON.parse(payload);
            if (evt.type === 'progress') {
              trainProgress.value = Math.round((evt.fraction || 0) * 100);
              trainProgressDesc.value = evt.desc || '';
              if (evt.memory_status) {
                trainLog.value += `🧠 ${evt.memory_status}\n`;
              } else {
                trainLog.value += `${evt.desc}\n`;
              }
              if (evt.loss != null) trainLoss.value.push({ step: evt.step || 0, loss: evt.loss });
              if (evt.elapsed != null) trainMetrics.elapsed = evt.elapsed;
              if (evt.eta != null) trainMetrics.eta = evt.eta;
              if (evt.lr != null) trainMetrics.lr = evt.lr;
              if (evt.epoch != null) trainMetrics.epoch = evt.epoch;
              if (evt.total_epochs != null) trainMetrics.total_epochs = evt.total_epochs;
              if (evt.grad_norm != null) trainMetrics.grad_norm = evt.grad_norm;
              if (evt.samples_per_sec != null) trainMetrics.samples_per_sec = evt.samples_per_sec;
              if (evt.total_steps != null) trainMetrics.total_steps = evt.total_steps;
              if (evt.loss != null) trainMetrics.current_loss = evt.loss;
              if (evt.device_type && !trainDevice.device_type) {
                trainDevice.device_type = evt.device_type;
                trainDevice.device_name = evt.device_name || '';
                trainDevice.gpu_name = evt.gpu_name || null;
                trainDevice.gpu_memory_gb = evt.gpu_memory_gb || null;
                trainDevice.ram_gb = evt.ram_gb || null;
                if (evt.device_type === 'cuda') {
                  trainDevice.message = `训练设备: ${evt.device_name || 'CUDA'} (${evt.gpu_memory_gb || '?'}GB 显存) — GPU 训练性能最优 ✓`;
                } else if (evt.device_type === 'cpu') {
                  trainDevice.message = `训练设备: CPU (${evt.ram_gb || '?'}GB 内存) — ⚠ CPU 训练较慢，建议使用 GPU`;
                } else {
                  trainDevice.message = `训练设备: ${evt.device_type || '未知'}`;
                }
              }
              autoScrollTrainLog();
            } else if (evt.type === 'hardware_diag') {
              trainDevice.device_type = evt.device_type || '';
              trainDevice.device_name = evt.device_name || '';
              trainDevice.gpu_name = evt.gpu_name || null;
              trainDevice.gpu_memory_gb = evt.gpu_memory_gb || null;
              trainDevice.ram_gb = evt.ram_gb || null;
              trainDevice.message = evt.message || '';
              trainLog.value += `${evt.message}\n`;
              autoScrollTrainLog();
            } else if (evt.type === 'log') {
              trainLog.value += evt.message + '\n';
              if (evt.loss != null) trainLoss.value.push({ step: evt.step, loss: evt.loss });
              autoScrollTrainLog();
            } else if (evt.type === 'warning') {
              trainLog.value += `⚠️ ${evt.message}\n`;
              autoScrollTrainLog();
            } else if (evt.type === 'error') {
              trainLog.value += `❌ ${evt.message}\n`;
              trainState.value = 'idle';
              autoScrollTrainLog();
              toast(`❌ 训练失败: ${evt.message}`, 'error');
            } else if (evt.type === 'completed') {
              trainLog.value += `✅ ${evt.message}\n`;
              trainState.value = 'completed';
              trainProgress.value = 100;
              autoScrollTrainLog();
            } else if (evt.type === 'stopped') {
              trainLog.value += `⏹ ${evt.message}\n`;
              trainState.value = 'idle';
              autoScrollTrainLog();
            }
          } catch (e) { console.debug('[useTraining] parse error:', e.message) } /* skip */ }
        }
      }
  } catch (err) {
    if (err.name !== 'AbortError') {
      trainLog.value += `❌ ${err.message}\n`;
      trainState.value = 'idle';
      autoScrollTrainLog();
      toast(`❌ ${err.message}`, 'error');
    } else {
      trainState.value = 'idle';
    }
  }
}

export async function pauseTraining(toast) {
  try {
    await authFetch(`${API_BASE}/api/train/pause`, { method: 'POST' });
    trainState.value = 'paused';
    toast('⏸ 训练已暂停', 'info');
  } catch (e) { toast(`❌ 暂停失败: ${e.message}`, 'error'); }
}

export async function resumeTraining(toast) {
  try {
    await authFetch(`${API_BASE}/api/train/resume`, { method: 'POST' });
    trainState.value = 'running';
    toast('▶ 训练已恢复', 'info');
  } catch (e) { toast(`❌ 恢复失败: ${e.message}`, 'error'); }
}

export async function stopTraining(toast) {
  if (trainAbortController) {
    try { trainAbortController.abort(); } catch (e) { }
    trainAbortController = null;
  }
  if (trainReader) {
    try { trainReader.cancel(); } catch (e) { }
    trainReader = null;
  }
  try { await authFetch(`${API_BASE}/api/train/stop`, { method: 'POST' }); } catch (e) { }
  trainState.value = 'idle';
  trainProgress.value = 0;
  toast('⏹ 训练已停止', 'info');
}

// ===== 检查点 =====

export async function loadCheckpoints() {
  try {
    const res = await authFetch(`${API_BASE}/api/train/checkpoints`);
    if (res.ok) {
      const data = await res.json();
      pendingCheckpoints.value = data.checkpoints || [];
    }
  } catch (e) { /* silent */ }
}

export async function resumeFromCheckpoint(toast, $confirm) {
  if (pendingCheckpoints.value.length === 0) { toast('⚠ 没有找到检查点', 'warning'); return; }

  const latestCkpt = pendingCheckpoints.value[0];
  const datasetMsg = selectedDatasets.value.length > 0
    ? `所选数据集: ${selectedDatasets.value.join(', ')}`
    : '将使用检查点中保存的数据集路径（如文件已被删除，请先上传并选择数据集）';
  const ok = await $confirm({
    title: '🔄 恢复训练',
    message: `将从检查点 "${latestCkpt.filename}" (Epoch ${latestCkpt.epoch}, Step ${latestCkpt.step}, Loss=${latestCkpt.loss?.toFixed(4) || '?'}) 继续训练\n\n${datasetMsg}`,
    confirmText: '确认恢复',
  });
  if (!ok) return;

  trainState.value = 'running';
  trainLog.value = '';
  trainLoss.value = [];
  trainProgress.value = 0;
  trainProgressDesc.value = '🔄 正在从检查点恢复训练...';
  trainAbortController = new AbortController();
  Object.assign(trainMetrics, { elapsed: 0, eta: null, lr: null, epoch: latestCkpt.epoch, total_epochs: latestCkpt.num_epochs || trainParams.epochs, grad_norm: null, samples_per_sec: 0, total_steps: 0, current_loss: null });
  Object.assign(trainDevice, { device_type: '', device_name: '', gpu_name: null, gpu_memory_gb: null, ram_gb: null, message: '' });

  const body = { checkpoint: latestCkpt.filename };
  if (selectedDatasets.value.length > 0) {
    body.datasets = [...selectedDatasets.value];
  }

  try {
    const res = await authFetch(`${API_BASE}/api/train/resume_checkpoint`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: trainAbortController.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    trainReader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let resumeCompleted = false;

    while (true) {
      const { done, value } = await trainReader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const payload = line.slice(6);
          if (payload === '[DONE]') {
            if (!resumeCompleted && trainState.value === 'running') trainState.value = 'completed';
            break;
          }
          try {
            const evt = JSON.parse(payload);
            if (evt.type === 'progress') {
              resumeCompleted = true;
              trainProgress.value = Math.round((evt.fraction || 0) * 100);
              trainProgressDesc.value = evt.desc || '';
              if (evt.memory_status) trainLog.value += `🧠 ${evt.memory_status}\n`;
              else trainLog.value += `${evt.desc}\n`;
              if (evt.loss != null) trainLoss.value.push({ step: evt.step || 0, loss: evt.loss });
              if (evt.elapsed != null) trainMetrics.elapsed = evt.elapsed;
              if (evt.eta != null) trainMetrics.eta = evt.eta;
              if (evt.lr != null) trainMetrics.lr = evt.lr;
              if (evt.epoch != null) trainMetrics.epoch = evt.epoch;
              if (evt.total_epochs != null) trainMetrics.total_epochs = evt.total_epochs;
              if (evt.grad_norm != null) trainMetrics.grad_norm = evt.grad_norm;
              if (evt.samples_per_sec != null) trainMetrics.samples_per_sec = evt.samples_per_sec;
              if (evt.total_steps != null) trainMetrics.total_steps = evt.total_steps;
              if (evt.loss != null) trainMetrics.current_loss = evt.loss;
              autoScrollTrainLog();
            } else if (evt.type === 'hardware_diag') {
              trainDevice.device_type = evt.device_type || '';
              trainDevice.device_name = evt.device_name || '';
              trainDevice.gpu_name = evt.gpu_name || null;
              trainDevice.gpu_memory_gb = evt.gpu_memory_gb || null;
              trainDevice.ram_gb = evt.ram_gb || null;
              trainDevice.message = evt.message || '';
              trainLog.value += `${evt.message}\n`;
              autoScrollTrainLog();
            } else if (evt.type === 'warning') {
              trainLog.value += `⚠️ ${evt.message}\n`;
              autoScrollTrainLog();
            } else if (evt.type === 'error') {
              trainLog.value += `❌ ${evt.message}\n`;
              trainState.value = 'idle';
              autoScrollTrainLog();
              toast(`❌ 恢复训练失败: ${evt.message}`, 'error');
            } else if (evt.type === 'completed') {
              resumeCompleted = true;
              trainLog.value += `✅ ${evt.message}\n`;
              trainState.value = 'completed';
              trainProgress.value = 100;
              autoScrollTrainLog();
            } else if (evt.type === 'stopped') {
              trainLog.value += `⏹ ${evt.message}\n`;
              trainState.value = 'idle';
              autoScrollTrainLog();
            }
          } catch (e) { console.debug('[useTraining] parse error:', e.message) } /* skip */ }
        }
      }
  } catch (err) {
    if (err.name !== 'AbortError') {
      trainLog.value += `❌ ${err.message}\n`;
      trainState.value = 'idle';
      autoScrollTrainLog();
      toast(`❌ ${err.message}`, 'error');
    } else {
      trainState.value = 'idle';
    }
  }
}

// ===== 发布（SSE 流式 + 进度条）=====

let publishAbortController = null;

export async function cancelPublish() {
  if (publishAbortController) {
    try { publishAbortController.abort(); } catch (e) { }
    publishAbortController = null;
  }
  // 通知后端重置发布状态
  try { await authFetch(`${API_BASE}/api/system/pub_reset`, { method: 'POST' }); } catch (e) { }
  publishingState.value = 'idle';
  trainLog.value += '⏹ 发布已取消\n';
  autoScrollTrainLog();
}

async function publishStreamCommon(toast, onStartMsg) {
  /* SSE 流式发布逻辑（支持结构化进度事件 + AbortController 取消） */
  publishingState.value = 'publishing';
  trainProgress.value = 0;
  trainProgressDesc.value = onStartMsg;
  trainLog.value += `${onStartMsg}\n`;

  // 创建 AbortController 允许取消发布
  publishAbortController = new AbortController();

  try {
    const res = await authFetch(`${API_BASE}/api/model/publish`, {
      method: 'POST',
      signal: publishAbortController.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let completed = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const payload = line.slice(6);
          if (payload === '[DONE]') { completed = true; break; }

          // 尝试解析结构化 JSON 事件
          try {
            const evt = JSON.parse(payload);
            if (evt.type === 'progress') {
              trainProgress.value = Math.round((evt.fraction || 0) * 100);
              trainProgressDesc.value = evt.desc || '';
              trainLog.value += `${evt.desc}\n`;
              autoScrollTrainLog();
            } else if (evt.type === 'log') {
              trainLog.value += `${evt.message}\n`;
              autoScrollTrainLog();
            } else if (evt.type === 'completed') {
              completed = true;
              trainLog.value += `✅ ${evt.message}\n`;
              trainProgress.value = 100;
              autoScrollTrainLog();
              toast(`✅ ${evt.message}`, 'success');
            } else if (evt.type === 'error') {
              trainLog.value += `❌ ${evt.message}\n`;
              trainProgress.value = 0;
              autoScrollTrainLog();
              toast(`❌ ${evt.message}`, 'error');
            } else if (evt.type === 'warning') {
              trainLog.value += `⚠️ ${evt.message}\n`;
              autoScrollTrainLog();
            }
          } catch (e) { console.debug('[useTraining] parse error:', e.message) }
            // 非 JSON 纯文本，直接输出
            if (payload) trainLog.value += `${payload}\n`;
            autoScrollTrainLog();
          }
        }
      if (completed) break;
    }

    if (!completed && !publishAbortController.signal.aborted) {
      toast('❌ 发布失败：SSE 流提前结束', 'error');
      trainLog.value += '❌ 发布失败：SSE 流提前结束\n';
      autoScrollTrainLog();
    }
  } catch (e) {
    if (e.name !== 'AbortError') {
      toast(`❌ ${e.message}`, 'error');
      trainLog.value += `❌ 发布出错: ${e.message}\n`;
      autoScrollTrainLog();
    }
  } finally {
    publishingState.value = 'idle';
    publishAbortController = null;
  }
}

export async function publishModel(toast) {
  await publishStreamCommon(toast, '📦 开始发布模型...');
}

export async function forcePublish(toast, $confirm) {
  if (pendingCheckpoints.value.length === 0) { toast('⚠ 没有找到检查点', 'warning'); return; }
  const ok = await $confirm({
    title: '📦 强制发布',
    message: `将从最新的检查点合并 LoRA 权重并发布为完整模型。\n\n即使训练未完成，也能得到一个可用的微调模型。\n\n点击确认开始发布。`,
    confirmText: '确认发布',
  });
  if (!ok) return;
  await publishStreamCommon(toast, '📦 强制发布模型...');
}

// ===== GGUF 导出 =====

export async function exportModelToGGUF(toast, $confirm) {
  if (publishingState.value !== 'idle') return;

  const ok = await $confirm({
    title: '📦 导出 GGUF 模型',
    message: `将从最新发布的模型导出为 GGUF 量化格式。\n\n推荐: Q4_K_M（质量/体积平衡）\n导出需要一定时间，请耐心等待。`,
    confirmText: '导出 (Q4_K_M)',
    cancelText: '取消',
  });
  if (!ok) return;

  publishingState.value = 'publishing';
  trainLog.value = '';

  try {
    const res = await authFetch(`${API_BASE}/api/model/export_gguf`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ quant_type: 'Q4_K_M' }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const payload = line.slice(6);
          if (payload === '[DONE]') break;
          try {
            const evt = JSON.parse(payload);
            if (evt.type === 'progress') {
              trainLog.value += `${evt.desc}\n`;
              trainProgress.value = Math.round((evt.fraction || 0) * 100);
              trainProgressDesc.value = evt.desc || '';
            } else if (evt.type === 'completed') {
              trainLog.value += `✅ ${evt.message}\n`;
              toast(`✅ GGUF 导出完成！\n路径: ${evt.file_path}\n大小: ${evt.file_size_gb} GB`, 'success');
            } else if (evt.type === 'error') {
              trainLog.value += `❌ ${evt.message}\n`;
              toast(`❌ 导出失败: ${evt.message}`, 'error');
            }
          } catch (e) { console.debug('[useTraining] parse error:', e.message) } /* skip */ }
        }
      }
  } catch (e) { toast(`❌ ${e.message}`, 'error'); }
  finally { publishingState.value = 'idle'; }
}

// ===== 态极模式：检测 + 生命活动 + 态极微调 =====

/**
 * 检测当前加载的模型是否为态极 ModelSelf。
 * 如果是，自动设置 isTaijiModel=true 并加载态极信息。
 * 应在训练页面 onMounted 时调用。
 */
export async function detectTaijiModel() {
  try {
    const res = await authFetch(`${API_BASE}/api/taiji/model/info`);
    if (res.ok) {
      const data = await res.json();
      if (data.status === 'active') {
        isTaijiModel.value = true;
        Object.assign(taijiModelInfo, {
          size: data.size || '',
          parameters: data.parameters || {},
          config: data.config || {},
          available_sizes: data.available_sizes || [],
          checkpoints: data.checkpoints || {},
        });
        // 自动加载生命状态
        await loadTaijiLifeStatus();
        return true;
      }
    }
  } catch (e) { /* not a taiji model */ }
  isTaijiModel.value = false;
  return false;
}

/**
 * 加载态极生命状态（心跳、空闲时间等）
 */
export async function loadTaijiLifeStatus() {
  try {
    const res = await authFetch(`${API_BASE}/api/taiji/life/status`);
    if (res.ok) {
      const data = await res.json();
      if (data.status === 'ok' && data.data) {
        Object.assign(taijiLifeStatus, data.data);
      }
    }
  } catch (e) { /* silent */ }
}

/**
 * 加载态极生命时间线
 */
export async function loadTaijiTimeline(hours = 24) {
  try {
    const res = await authFetch(`${API_BASE}/api/taiji/life/timeline?hours=${hours}`);
    if (res.ok) {
      const data = await res.json();
      if (data.status === 'ok') {
        taijiTimeline.value = data.timeline || [];
      }
    }
  } catch (e) { /* silent */ }
}

/**
 * 喂养态极
 */
export async function feedTaiji(toast) {
  taijiLifeLoading.value = true;
  try {
    const res = await authFetch(`${API_BASE}/api/taiji/feed`, { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      toast(`🍚 喂养完成！喂食 ${data.items_fed} 条，生成 ${data.samples_generated} 条训练样本`, 'success');
      await loadTaijiLifeStatus();
    } else {
      toast('❌ 喂养失败', 'error');
    }
  } catch (e) { toast(`❌ 喂养失败: ${e.message}`, 'error'); }
  finally { taijiLifeLoading.value = false; }
}

/**
 * 让态极睡觉
 */
export async function sleepTaiji(toast) {
  taijiLifeLoading.value = true;
  try {
    const res = await authFetch(`${API_BASE}/api/taiji/sleep`, { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      const phases = data.phases_completed?.length || 0;
      const loss = data.training_loss ? `，训练 Loss: ${data.training_loss.toFixed(4)}` : '';
      toast(`💤 睡眠完成！完成 ${phases} 个阶段${loss}`, 'success');
      await loadTaijiLifeStatus();
    } else {
      toast('❌ 睡眠失败', 'error');
    }
  } catch (e) { toast(`❌ 睡眠失败: ${e.message}`, 'error'); }
  finally { taijiLifeLoading.value = false; }
}

/**
 * 让态极玩耍
 */
export async function playTaiji(toast) {
  taijiLifeLoading.value = true;
  try {
    const res = await authFetch(`${API_BASE}/api/taiji/play`, { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      const count = data.activities?.length || 0;
      const mood = data.mood || '';
      toast(`🎮 玩耍完成！${count} 个活动，心情: ${mood}`, 'success');
      await loadTaijiLifeStatus();
    } else {
      toast('❌ 玩耍失败', 'error');
    }
  } catch (e) { toast(`❌ 玩耍失败: ${e.message}`, 'error'); }
  finally { taijiLifeLoading.value = false; }
}

/**
 * 启动态极原生微调（SSE 流式）。
 * 态极模式下替代 startTraining()。
 */
export async function startTaijiTraining(toast) {
  trainState.value = 'running';
  trainLog.value = '';
  trainLoss.value = [];
  trainProgress.value = 0;
  trainProgressDesc.value = '⏳ 正在初始化态极微调环境...';
  trainAbortController = new AbortController();
  Object.assign(trainMetrics, {
    elapsed: 0, eta: null, lr: null, epoch: 1,
    total_epochs: taijiTrainParams.num_epochs,
    grad_norm: null, samples_per_sec: 0, total_steps: 0, current_loss: null,
  });
  Object.assign(trainDevice, {
    device_type: '', device_name: '', gpu_name: null,
    gpu_memory_gb: null, ram_gb: null, message: '',
  });

  const body = { ...taijiTrainParams, dataset_files: [...selectedDatasets.value] };

  try {
    const res = await authFetch(`${API_BASE}/api/taiji/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: trainAbortController.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    trainReader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await trainReader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const payload = line.slice(6);
          if (payload === '[DONE]') {
            if (trainState.value === 'running') trainState.value = 'completed';
            break;
          }
          try {
            const evt = JSON.parse(payload);
            if (evt.type === 'progress') {
              trainProgress.value = Math.round((evt.fraction || 0) * 100);
              trainProgressDesc.value = evt.desc || '';
              trainLog.value += `${evt.desc}\n`;
              if (evt.loss != null) trainLoss.value.push({ step: evt.step || 0, loss: evt.loss });
              if (evt.lr != null) trainMetrics.lr = evt.lr;
              if (evt.step != null) trainMetrics.total_steps = evt.step;
              if (evt.loss != null) trainMetrics.current_loss = evt.loss;

              // 瓶颈检测事件
              if (evt.bottleneck) {
                toast(`⚠️ 检测到训练瓶颈！${evt.desc || '建议升级模型以获得更好的效果'}`, 'warning');
              }
              autoScrollTrainLog();
            } else if (evt.type === 'error') {
              trainLog.value += `❌ ${evt.message}\n`;
              trainState.value = 'idle';
              autoScrollTrainLog();
              toast(`❌ 态极微调失败: ${evt.message}`, 'error');
            } else if (evt.type === 'completed') {
              trainLog.value += `✅ ${evt.message}\n`;
              trainState.value = 'completed';
              trainProgress.value = 100;
              autoScrollTrainLog();
              // 刷新态极信息（checkpoint 可能更新了）
              await detectTaijiModel();
            } else if (evt.type === 'stopped') {
              trainLog.value += `⏹ ${evt.message}\n`;
              trainState.value = 'idle';
              autoScrollTrainLog();
            }
          } catch (e) { console.debug('[useTraining] parse error:', e.message) } /* skip */ }
        }
      }
  } catch (err) {
    if (err.name !== 'AbortError') {
      trainLog.value += `❌ ${err.message}\n`;
      trainState.value = 'idle';
      autoScrollTrainLog();
      toast(`❌ ${err.message}`, 'error');
    } else {
      trainState.value = 'idle';
    }
  }
}

// ===== Loss 曲线绘图 =====

export function drawLossChart() {
  const canvas = lossCanvasRef.value;
  if (!canvas || trainLoss.value.length < 2) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const w = rect.width, h = rect.height;
  const pad = { top: 12, right: 16, bottom: 28, left: 48 };
  const pw = w - pad.left - pad.right;
  const ph = h - pad.top - pad.bottom;
  const data = trainLoss.value;
  const losses = data.map(d => d.loss);
  const minL = Math.min(...losses), maxL = Math.max(...losses);
  const range = maxL - minL || 1;

  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(148,163,184,0.12)';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (ph / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
  }

  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px ' + getComputedStyle(document.documentElement).fontFamily;
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = maxL - (range / 4) * i;
    const y = pad.top + (ph / 4) * i + 3;
    ctx.fillText(val.toFixed(3), pad.left - 6, y);
  }

  ctx.textAlign = 'center';
  const steps = data.map(d => d.step);
  ctx.fillText('Step ' + steps[0], pad.left, h - 4);
  ctx.fillText('Step ' + steps[steps.length - 1], w - pad.right, h - 4);

  const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + ph);
  grad.addColorStop(0, 'rgba(99,102,241,0.28)');
  grad.addColorStop(1, 'rgba(99,102,241,0.02)');

  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = pad.left + (i / (data.length - 1)) * pw;
    const y = pad.top + ((maxL - data[i].loss) / range) * ph;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.lineTo(pad.left + pw, pad.top + ph);
  ctx.lineTo(pad.left, pad.top + ph);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = pad.left + (i / (data.length - 1)) * pw;
    const y = pad.top + ((maxL - data[i].loss) / range) * ph;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = '#1a1a1a';
  ctx.lineWidth = 2;
  ctx.lineJoin = 'round';
  ctx.stroke();

  for (let i = 0; i < data.length; i++) {
    const x = pad.left + (i / (data.length - 1)) * pw;
    const y = pad.top + ((maxL - data[i].loss) / range) * ph;
    ctx.beginPath();
    ctx.arc(x, y, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = '#1a1a1a';
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}
