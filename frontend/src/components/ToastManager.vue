<template>
  <div class="toast-container" role="alert" aria-live="polite" aria-atomic="false">
    <transition-group name="toast">
      <div
        v-for="toast in toasts"
        :key="toast.id"
        :class="['toast-item', toast.type]"
      >
        <component :is="icons[toast.type]" :size="16" class="toast-icon" />
        <span class="toast-msg">{{ toast.message }}</span>
        <button class="toast-close" @click="removeToast(toast.id)">✕</button>
        <div class="toast-progress" :style="{ animationDuration: toast.duration + 'ms' }"></div>
      </div>
    </transition-group>
  </div>
</template>

<script setup>
import { CheckCircle, XCircle, Info, AlertTriangle } from 'lucide-vue-next';

import { ref } from 'vue'

const toasts = ref([])
const icons = {
  success: 'CheckCircle',
  error: 'XCircle',
  warning: 'AlertTriangle',
  info: 'Info'
}

let counter = 0

function sanitizeMessage(msg) {
  // 截断过长的错误消息，去除堆栈跟踪
  if (typeof msg !== 'string') return String(msg)
  // 截掉 Traceback/stack trace 行
  const lines = msg.split('\n')
  const clean = []
  for (const line of lines) {
    if (line.match(/^\s*(File |Traceback|  File "|    |\w+Error:)/)) break
    clean.push(line)
  }
  const result = clean.join('\n').trim() || msg.slice(0, 200)
  return result.length > 300 ? result.slice(0, 297) + '...' : result
}

function showToast(message, type = 'info', duration = 3000) {
  const id = ++counter
  toasts.value.push({ id, message: sanitizeMessage(message), type, duration })
  setTimeout(() => removeToast(id), duration)
}

function removeToast(id) {
  const idx = toasts.value.findIndex(t => t.id === id)
  if (idx > -1) toasts.value.splice(idx, 1)
}

defineExpose({ showToast })
</script>

<style scoped>
.toast-container {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 99990;
  display: flex;
  flex-direction: column;
  gap: 10px;
  pointer-events: none;
}
.toast-item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 18px;
  padding-right: 40px;
  border-radius: 12px;
  font-size: 0.9rem;
  font-weight: 500;
  box-shadow: 0 8px 32px rgba(0,0,0,0.12);
  min-width: 280px;
  max-width: 420px;
  overflow: hidden;
  pointer-events: auto;
  backdrop-filter: blur(12px);
}
.toast-item.success { background: rgba(16,185,129,0.95); color: white; }
.toast-item.error { background: rgba(239,68,68,0.95); color: white; }
.toast-item.warning { background: rgba(245,158,11,0.95); color: white; }
.toast-item.info { background: var(--primary); color: white; }
.toast-icon { font-size: 1.2rem; }
.toast-msg { flex: 1; line-height: 1.4; }
.toast-close {
  position: absolute;
  top: 8px;
  right: 10px;
  background: none;
  border: none;
  color: rgba(255,255,255,0.7);
  cursor: pointer;
  font-size: 0.8rem;
  padding: 4px;
  border-radius: 4px;
  transition: all 0.2s;
}
.toast-close:hover { color: white; background: rgba(255,255,255,0.15); }
.toast-progress {
  position: absolute;
  bottom: 0;
  left: 0;
  height: 3px;
  background: rgba(255,255,255,0.4);
  animation: toastProgress linear forwards;
}
@keyframes toastProgress { from { width: 100%; } to { width: 0%; } }

.toast-enter-active { animation: toastIn 0.3s ease; }
.toast-leave-active { animation: toastOut 0.25s ease; }
@keyframes toastIn { from { opacity: 0; transform: translateX(40px) scale(0.95); } to { opacity: 1; transform: translateX(0) scale(1); } }
@keyframes toastOut { from { opacity: 1; transform: translateX(0) scale(1); } to { opacity: 0; transform: translateX(40px) scale(0.95); } }
</style>
