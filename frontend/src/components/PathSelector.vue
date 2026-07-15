<template>
  <div class="path-selector" :class="{ 'path-selector-inline': inline }">
    <div class="path-selector-input-group">
      <input
        ref="inputRef"
        :type="type === 'folder' ? 'text' : 'text'"
        :value="modelValue"
        @input="onInput"
        @keydown.enter="onEnter"
        class="path-input form-input"
        :placeholder="placeholder"
        :disabled="disabled"
      />
      <button
        class="path-btn browse-btn"
        :title="type === 'folder' ? '浏览文件夹' : '浏览文件'"
        @click="browse"
        :disabled="disabled"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"/>
        </svg>
        <span v-if="!inline">浏览</span>
      </button>
      <button
        v-if="showOpen && modelValue"
        class="path-btn open-btn"
        title="在资源管理器中打开"
        @click="openInExplorer"
        :disabled="disabled"
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/>
          <polyline points="15 3 21 3 21 9"/>
          <line x1="10" y1="14" x2="21" y2="3"/>
        </svg>
      </button>
      <button
        v-if="modelValue && clearable"
        class="path-btn clear-btn"
        title="清除"
        @click="onClear"
      >
        ✕
      </button>
    </div>
    <div v-if="error" class="path-error">{{ error }}</div>
  </div>
</template>

<script setup>
import { ref } from 'vue';

const props = defineProps({
  modelValue: { type: String, default: '' },
  type: { type: String, default: 'folder' }, // 'file' | 'folder'
  placeholder: { type: String, default: '' },
  disabled: { type: Boolean, default: false },
  inline: { type: Boolean, default: false },
  showOpen: { type: Boolean, default: true },
  clearable: { type: Boolean, default: true },
  fileFilter: { type: String, default: '' },
  dialogTitle: { type: String, default: '' },
});

const emit = defineEmits(['update:modelValue', 'browse', 'open', 'validate']);
import { API_BASE, authFetch } from '../composables/apiClient.js';

const inputRef = ref(null);
const error = ref('');

const onInput = (e) => {
  const val = e.target.value;
  error.value = '';
  emit('update:modelValue', val);
};

const onEnter = () => {
  if (props.modelValue) {
    validateAndEmit();
  }
};

const onClear = () => {
  emit('update:modelValue', '');
  error.value = '';
};

const validateAndEmit = async () => {
  const path = props.modelValue.trim();
  if (!path) return;
  emit('validate', path);
  try {
    const res = await authFetch(`${API_BASE}/api/system/validate_path`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path, type: props.type }),
    });
    const data = await res.json();
    if (data.status === 'ok') {
      error.value = '';
    } else {
      error.value = data.message || '路径不可用';
    }
  } catch (e) { console.warn('[PathSelector] check failed:', e.message) }
    error.value = '路径验证失败';
  }
};

const browse = async () => {
  try {
    const title = encodeURIComponent(props.dialogTitle || (props.type === 'folder' ? '请选择文件夹' : '请选择文件'));
    const endpoint = props.type === 'folder'
      ? `${API_BASE}/api/system/select_folder?title=${title}`
      : `${API_BASE}/api/system/select_file`;
    const res = await authFetch(endpoint);
    const data = await res.json();
    if (data.status === 'ok' && data.path) {
      emit('update:modelValue', data.path);
      error.value = '';
      emit('browse', data.path);
    } else if (data.status === 'error') {
      error.value = data.message || '选择失败';
    }
  } catch (err) {
    error.value = '选择失败: ' + err.message;
  }
};

const openInExplorer = async () => {
  if (!props.modelValue) return;
  emit('open', props.modelValue);
  try {
    await authFetch(`${API_BASE}/api/system/open_folder`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: props.modelValue }),
    });
  } catch (err) {
    console.error('打开文件夹失败:', err);
  }
};

const focus = () => {
  inputRef.value?.focus();
};

defineExpose({ focus });
</script>

<style scoped>
.path-selector { display: flex; flex-direction: column; gap: 4px; }
.path-selector-inline { display: inline-flex; flex-direction: row; align-items: center; gap: 8px; }
.path-selector-input-group {
  display: flex; gap: 4px; align-items: center;
  flex: 1; min-width: 0;
}
.path-input {
  flex: 1; min-width: 0;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 0.82rem;
}
.path-input::placeholder { font-family: inherit; }
.path-btn {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 6px 10px; border-radius: 6px;
  font-size: 0.8rem; cursor: pointer; border: 1px solid var(--border-muted);
  background: var(--bg-card); color: var(--text-secondary);
  transition: var(--transition); flex-shrink: 0;
  white-space: nowrap;
}
.path-btn:hover { background: var(--bg-hover); color: var(--primary); border-color: var(--primary); }
.path-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.browse-btn { color: var(--primary); }
.open-btn { color: var(--success); }
.clear-btn { color: var(--danger); border-color: transparent; padding: 6px 8px; }
.clear-btn:hover { background: var(--danger-light); color: var(--danger); border-color: var(--danger); }
.path-error { font-size: 0.75rem; color: var(--danger); padding: 0 4px; }
</style>
