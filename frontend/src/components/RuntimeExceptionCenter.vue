/**
 * 运行时异常中心组件
 * ===================
 *
 * 显示运行时异常，告诉用户：
 * - 发生了什么
 * - 影响什么
 * - 怎么恢复
 */
<template>
  <div v-if="visibleIssues.length" class="exception-center">
    <transition-group name="exception" tag="div" class="exception-list">
      <div
        v-for="issue in visibleIssues"
        :key="issue.createdAt || issue.title"
        class="exception-card"
        :class="'exception-' + issue.level"
      >
        <div class="exception-icon">
          <AlertTriangle v-if="issue.level === 'danger'" :size="15" />
          <CircleAlert v-else-if="issue.level === 'warning'" :size="15" />
          <Info v-else :size="15" />
        </div>
        <div class="exception-body">
          <div class="exception-title">{{ issue.title }}</div>
          <div class="exception-message">{{ issue.message }}</div>
        </div>
        <button class="exception-dismiss" @click="dismissIssue(issue)">×</button>
      </div>
    </transition-group>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { AlertTriangle, CircleAlert, Info } from 'lucide-vue-next'
import { useRuntimeStore } from '@/stores/runtimeStore.js'

const runtimeStore = useRuntimeStore()

// 过滤掉终端相关的问题，只显示关键异常
const visibleIssues = computed(() =>
  runtimeStore.issues
    .filter(i => i.title !== '终端不可用')
    .slice(0, 2)
)

function dismissIssue(issue) {
  runtimeStore.clearException(issue.title)
  if (issue.title === '运行时断开') {
    runtimeStore.syncHealth('connecting')
  } else if (issue.title === '需要重新登录') {
    runtimeStore.auth.error = ''
  }
}
</script>

<style scoped>
.exception-center {
  position: fixed;
  bottom: 24px;
  right: 24px;
  z-index: 9990;
  max-width: 360px;
}

.exception-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.exception-card {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 12px 14px;
  border-radius: var(--radius-md);
  background: var(--glass-bg);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--glass-border);
  box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.exception-card.exception-danger { border-left: 3px solid var(--danger); }
.exception-card.exception-warning { border-left: 3px solid var(--warning); }
.exception-card.exception-info { border-left: 3px solid var(--info); }

.exception-icon {
  width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;
  border-radius: var(--radius-sm); color: var(--text-muted); background: var(--bg-muted); flex-shrink: 0;
}
.exception-danger .exception-icon { color: var(--danger); background: rgba(239,68,68,0.1); }
.exception-warning .exception-icon { color: var(--warning); background: rgba(245,158,11,0.1); }

.exception-body { flex: 1; min-width: 0; }
.exception-title { font-size: 13px; font-weight: 600; color: var(--text); margin-bottom: 2px; }
.exception-message { font-size: 12px; color: var(--text-secondary); line-height: 1.5; }

.exception-dismiss {
  background: none; border: none; color: var(--text-muted); cursor: pointer;
  font-size: 16px; padding: 0 4px; flex-shrink: 0; border-radius: 6px; transition: all 0.15s;
}
.exception-dismiss:hover { color: var(--text); background: var(--bg-hover); }

@keyframes slideIn {
  from { opacity: 0; transform: translateX(20px); }
  to { opacity: 1; transform: translateX(0); }
}
.exception-enter-active { animation: slideIn 0.3s ease; }
.exception-leave-active { animation: slideIn 0.3s ease reverse; }
</style>
