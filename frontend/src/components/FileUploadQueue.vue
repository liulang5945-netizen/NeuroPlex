<template>
  <!-- 隐藏的文件输入框 -->
  <input
    type="file"
    ref="fileInputRef"
    multiple
    :accept="accept"
    @change="onFileSelect"
    style="display: none"
  />

  <!-- 拖拽区域 -->
  <div
    class="upload-dropzone"
    :class="{ 'drag-over': isDragOver }"
    @dragover.prevent="isDragOver = true"
    @dragleave.prevent="isDragOver = false"
    @drop.prevent="onDrop"
    @click="triggerBrowse"
  >
    <component :is="uploadIcon" class="dropzone-icon" :size="32" />
    <p class="dropzone-text">{{ dropText }}</p>
    <p class="dropzone-hint" v-if="acceptHint">{{ acceptHint }}</p>
  </div>

  <!-- 上传队列面板 -->
  <div class="panel-section" v-if="queue.length > 0">
    <div class="panel-header">
      <h3><Clock :size="14" class="queue-title-icon" /> {{ title }}</h3>
      <button class="icon-btn" @click="queue = []" title="清空"><X :size="14" /></button>
    </div>
    <div class="panel-content">
      <div class="file-item" v-for="(file, idx) in queue" :key="file.id">
        <div class="file-info"><component :is="icon" :size="14" class="file-icon" /> {{ file.name }}</div>
        <div class="file-actions">
          <span :class="['status-tag', file.status]">{{ file.statusText }}</span>
          <button class="delete-btn" @click="removeFromQueue(idx)" v-if="file.status !== 'uploading'"><X :size="14" /></button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { UploadCloud, X, FileText, Library, BarChart2, Download, Clock } from 'lucide-vue-next';

import { ref } from 'vue';
import { API_BASE, authFetch } from '../composables/apiClient.js';

const props = defineProps({
  /** 上传接口路径，如 '/api/rag/upload' */
  uploadEndpoint: { type: String, required: true },
  /** 接受的文件扩展名列表 */
  accept: { type: String, default: '' },
  /** 文件图标 */
  icon: { type: String, default: '📄' },
  /** 队列面板标题 */
  title: { type: String, default: '上传队列' },
  /** 上传成功时显示的文字 */
  successText: { type: String, default: '✅ 上传成功' },
  /** 拖拽区图标 */
  uploadIcon: { type: String, default: '📤' },
  /** 拖拽区提示文字 */
  dropText: { type: String, default: '拖拽文件到此处上传，或点击选择文件' },
  /** 支持格式提示 */
  acceptHint: { type: String, default: '' },
});

const emit = defineEmits([
  /** 文件上传成功后触发 { file, response } */
  'upload-success',
  /** 文件上传失败后触发 { file, error } */
  'upload-error',
  /** 文件加入队列时触发 { file } */
  'files-added',
  /** 所有文件上传完成后触发 */
  'all-uploaded',
]);

const queue = ref([]);
const fileInputRef = ref(null);
const isDragOver = ref(false);

/** 并发上传控制：最多同时上传 3 个文件 */
const MAX_CONCURRENT = 3;

/** 触发系统文件选择对话框 */
const triggerBrowse = () => {
  fileInputRef.value?.click();
};

/**
 * 并发上传指定文件列表（至多 MAX_CONCURRENT 个同时上传）
 * @param {FileList|File[]} files
 */
const handleFiles = async (files) => {
  const fileArray = Array.from(files);
  // 为每个文件创建队列项
  const items = fileArray.map((file) => {
    const id = Date.now() + Math.random();
    queue.value.push({ id, name: file.name, status: 'uploading', statusText: '上传中...' });
    emit('files-added', { file });
    return { id, file };
  });

  // 并发上传函数
  const uploadOne = async ({ id, file }) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await authFetch(`${API_BASE}${props.uploadEndpoint}`, { method: 'POST', body: formData });
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || '上传失败');
      }
      const json = await res.json().catch(() => ({}));
      const idx = queue.value.findIndex(f => f.id === id);
      if (idx > -1) queue.value[idx] = { ...queue.value[idx], status: 'success', statusText: props.successText };
      emit('upload-success', { file, response: json });
    } catch (err) {
      const idx = queue.value.findIndex(f => f.id === id);
      if (idx > -1) queue.value[idx] = { ...queue.value[idx], status: 'error', statusText: err.message };
      emit('upload-error', { file, error: err.message });
    }
  };

  // 分块并发：每次最多 MAX_CONCURRENT 个同时上传
  for (let i = 0; i < items.length; i += MAX_CONCURRENT) {
    const batch = items.slice(i, i + MAX_CONCURRENT);
    await Promise.all(batch.map(uploadOne));
  }

  emit('all-uploaded');
};

const removeFromQueue = (idx) => {
  queue.value.splice(idx, 1);
};

const onDrop = (e) => {
  isDragOver.value = false;
  handleFiles(e.dataTransfer.files);
};

const onFileSelect = (e) => {
  handleFiles(e.target.files);
  if (e.target) e.target.value = '';
};

defineExpose({ queue, handleFiles, removeFromQueue, onDrop, triggerBrowse });
</script>

<style scoped>
.upload-dropzone {
  border: 2px dashed var(--border-muted, #cbd5e1);
  border-radius: var(--radius-md, 12px);
  padding: 32px 20px;
  text-align: center;
  cursor: pointer;
  transition: var(--transition, all 0.2s);
  background: var(--bg-muted, #f8fafc);
}
.upload-dropzone:hover,
.upload-dropzone.drag-over {
  border-color: var(--primary);
  background: var(--primary-subtle, var(--primary-subtle));
}
.dropzone-icon {
  font-size: 2.5rem;
  margin-bottom: 8px;
}
.dropzone-text {
  font-size: 0.9rem;
  color: var(--text-secondary, #475569);
  margin: 0 0 4px;
}
.dropzone-hint {
  font-size: 0.75rem;
  color: var(--text-muted, #94a3b8);
  margin: 0;
}

.file-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 10px;
  background: var(--bg, #f1f5f9);
  border-radius: var(--radius-sm, 8px);
  font-size: 0.82rem;
  gap: 8px;
}
.file-info {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.file-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}
.status-tag {
  font-size: 0.75rem;
  padding: 2px 8px;
  border-radius: 10px;
  font-weight: 500;
}
.status-tag.uploading { color: var(--warning); background: var(--warning-light); }
.status-tag.success { color: var(--success); background: var(--success-light); }
.status-tag.error { color: var(--danger); background: var(--danger-light); }
.delete-btn {
  background: none;
  border: none;
  color: var(--text-muted, #999);
  cursor: pointer;
  font-size: 1rem;
  padding: 2px 4px;
  border-radius: 4px;
}
.delete-btn:hover { color: var(--danger, #ef4444); background: var(--danger-light, rgba(239,68,68,0.1)); }
</style>
