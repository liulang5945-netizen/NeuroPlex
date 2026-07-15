<template>
  <div class="log-panel">
    <div class="log-header">
      <div class="log-title">
        <Terminal :size="14" />
        <span>系统日志</span>
      </div>
      <div class="log-controls">
        <select v-model="filter" class="log-filter">
          <option value="all">全部</option>
          <option value="life">生命</option>
          <option value="train">训练</option>
          <option value="agent">Agent</option>
          <option value="model">模型</option>
          <option value="error">错误</option>
        </select>
        <button class="log-btn" @click="clearLogs" title="清空"><Trash2 :size="13" /></button>
        <button class="log-btn" :class="{ on: autoScroll }" @click="autoScroll = !autoScroll" title="自动滚动">
          <ArrowDown :size="13" />
        </button>
      </div>
    </div>
    <div class="log-body" ref="logBody">
      <div v-if="!filteredLogs.length" class="log-empty">暂无日志</div>
      <div v-for="(log, i) in filteredLogs" :key="i" class="log-entry" :class="'log-' + log.level">
        <span class="log-time">{{ formatTime(log.timestamp) }}</span>
        <span class="log-badge" :class="'badge-' + log.level">{{ log.source }}</span>
        <span class="log-msg">{{ log.message }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick, onMounted } from 'vue'
import { Terminal, Trash2, ArrowDown } from 'lucide-vue-next'
import { useRuntimeStore } from '@/stores/runtimeStore.js'

const runtimeStore = useRuntimeStore()
const filter = ref('all')
const autoScroll = ref(true)
const logBody = ref(null)

const logs = computed(() => runtimeStore.logs || [])

const filteredLogs = computed(() => {
  if (filter.value === 'all') return logs.value
  if (filter.value === 'error') return logs.value.filter(l => l.level === 'error' || l.level === 'warn')
  return logs.value.filter(l => l.source === filter.value)
})

function formatTime(ts) {
  if (!ts) return ''
  const d = new Date(ts)
  return d.toLocaleTimeString('zh-CN', { hour12: false })
}

function clearLogs() {
  runtimeStore.clearLogs()
}

watch(filteredLogs, () => {
  if (autoScroll.value) {
    nextTick(() => {
      if (logBody.value) logBody.value.scrollTop = logBody.value.scrollHeight
    })
  }
}, { deep: true })
</script>

<style scoped>
.log-panel {
  display: flex; flex-direction: column;
  height: 100%; background: var(--bg); border: 1px solid var(--border);
  border-radius: var(--radius-md); overflow: hidden;
}
.log-header {
  display: flex; justify-content: space-between; align-items: center;
  padding: 8px 12px; border-bottom: 1px solid var(--border);
  background: var(--bg-surface);
}
.log-title {
  display: flex; align-items: center; gap: 6px;
  font-size: 12px; font-weight: 600; color: var(--text);
}
.log-controls { display: flex; align-items: center; gap: 4px; }
.log-filter {
  height: 24px; padding: 0 6px; font-size: 11px;
  background: var(--bg-elevated); color: var(--text-secondary);
  border: 1px solid var(--border); border-radius: var(--radius-sm); outline: none;
}
.log-btn {
  width: 24px; height: 24px; display: flex; align-items: center; justify-content: center;
  background: transparent; border: 1px solid transparent; border-radius: var(--radius-sm);
  color: var(--text-muted); cursor: pointer; transition: var(--transition-fast);
}
.log-btn:hover { background: var(--bg-hover); color: var(--text); }
.log-btn.on { color: var(--primary); background: var(--primary-subtle); }
.log-body {
  flex: 1; overflow-y: auto; padding: 6px 0; font-family: var(--font-mono); font-size: 12px;
}
.log-empty { text-align: center; padding: 24px; color: var(--text-muted); font-size: 12px; }
.log-entry {
  display: flex; align-items: baseline; gap: 8px; padding: 3px 12px;
  line-height: 1.5; border-bottom: 1px solid var(--border-subtle);
}
.log-entry:hover { background: var(--bg-hover); }
.log-time { color: var(--text-muted); font-size: 11px; flex-shrink: 0; }
.log-badge {
  padding: 0 5px; border-radius: 3px; font-size: 10px; font-weight: 500;
  flex-shrink: 0; line-height: 16px;
}
.badge-life { background: var(--success-light); color: var(--success); }
.badge-train { background: var(--info-light); color: var(--info); }
.badge-agent { background: var(--primary-light); color: var(--primary); }
.badge-model { background: var(--purple); color: white; opacity: 0.8; }
.badge-error, .badge-warn { background: var(--danger-light); color: var(--danger); }
.log-msg { color: var(--text-secondary); word-break: break-all; }
.log-error .log-msg { color: var(--danger); }
.log-warn .log-msg { color: var(--warning); }
</style>
