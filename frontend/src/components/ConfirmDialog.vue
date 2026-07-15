<template>
  <div v-if="visible" class="confirm-overlay" @click.self="handleCancel" @keydown.escape="handleCancel" tabindex="-1" ref="overlayRef">
    <div class="confirm-dialog" role="dialog" aria-modal="true" :aria-labelledby="title ? 'confirm-title' : undefined">
      <div class="confirm-icon"><component :is="type === 'danger' ? AlertTriangle : type === 'warning' ? Bell : MessageSquare" :size="24" /></div>
      <div class="confirm-title" id="confirm-title">{{ title }}</div>
      <div class="confirm-message" v-if="message">{{ message }}</div>
      <div class="confirm-actions">
        <button class="confirm-btn cancel" @click="handleCancel">{{ cancelText }}</button>
        <button :class="['confirm-btn', type]" @click="handleConfirm">{{ confirmText }}</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { AlertTriangle, Bell, MessageSquare } from 'lucide-vue-next';

import { ref, nextTick, watch } from 'vue'

const visible = ref(false)
const overlayRef = ref(null)
const title = ref('')
const message = ref('')
const type = ref('primary')
const confirmText = ref('确定')
const cancelText = ref('取消')

let resolvePromise = null
let previousFocus = null

function show(options = {}) {
  title.value = options.title || '确认操作'
  message.value = options.message || ''
  type.value = options.type || 'primary'
  confirmText.value = options.confirmText || '确定'
  cancelText.value = options.cancelText || '取消'
  // 保存当前焦点，关闭时恢复
  previousFocus = document.activeElement
  visible.value = true
  nextTick(() => {
    overlayRef.value?.focus()
  })
  return new Promise((resolve) => {
    resolvePromise = resolve
  })
}

function handleConfirm() {
  closeAndRestore(true)
}

function handleCancel() {
  closeAndRestore(false)
}

function closeAndRestore(result) {
  visible.value = false
  // 恢复焦点
  if (previousFocus && typeof previousFocus.focus === 'function') {
    try { previousFocus.focus() } catch (_) { /* non-critical */ }
  }
  previousFocus = null
  if (resolvePromise) resolvePromise(result)
}

defineExpose({ show })
</script>

<style scoped>
.confirm-overlay {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(15,23,42,0.55);
  backdrop-filter: blur(6px);
  z-index: 99999;
  display: flex;
  align-items: center;
  justify-content: center;
  animation: fadeIn 0.2s ease;
}

.confirm-dialog {
  background: var(--bg-card);
  border-radius: 16px;
  padding: 28px 32px 24px;
  min-width: 320px;
  max-width: 420px;
  box-shadow: var(--shadow-xl);
  animation: confirmIn 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  text-align: center;
}

.confirm-icon {
  font-size: 2.4rem;
  margin-bottom: 12px;
}

.confirm-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 8px;
  line-height: 1.4;
}

.confirm-message {
  font-size: 0.9rem;
  color: var(--text-secondary);
  margin-bottom: 20px;
  line-height: 1.5;
}

.confirm-actions {
  display: flex;
  gap: 10px;
  justify-content: center;
}

.confirm-btn {
  padding: 8px 24px;
  border-radius: 10px;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  border: none;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  min-width: 80px;
}

.confirm-btn.cancel {
  background: var(--bg-muted);
  color: var(--text-secondary);
}
.confirm-btn.cancel:hover {
  background: var(--bg-hover);
}

.confirm-btn.primary {
  background: var(--primary-gradient);
  color: white;
  box-shadow: 0 2px 8px rgba(91,122,138,0.3);
}
.confirm-btn.primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(91,122,138,0.4);
}

.confirm-btn.danger {
  background: var(--danger);
  color: white;
  box-shadow: 0 2px 8px rgba(199,90,90,0.3);
}
.confirm-btn.danger:hover {
  opacity: 0.9;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(199,90,90,0.4);
}

.confirm-btn.warning {
  background: var(--warning);
  color: white;
  box-shadow: 0 2px 8px rgba(196,138,63,0.3);
}
.confirm-btn.warning:hover {
  opacity: 0.9;
  transform: translateY(-1px);
}

@keyframes fadeIn {
  from { opacity: 0; } to { opacity: 1; }
}

@keyframes confirmIn {
  from { opacity: 0; transform: scale(0.92) translateY(12px); }
  to { opacity: 1; transform: scale(1) translateY(0); }
}
</style>
