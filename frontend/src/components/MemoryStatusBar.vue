<template>
  <div :class="['memory-ring', levelClass]" :title="tooltipText">
    <svg viewBox="0 0 28 28">
      <circle class="ring-bg" cx="14" cy="14" r="11" />
      <circle class="ring-fill" cx="14" cy="14" r="11"
        :stroke-dasharray="circumference"
        :stroke-dashoffset="dashOffset" />
    </svg>
    <span class="ring-text">{{ usedPct }}%</span>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRuntimeStore } from '../stores/runtimeStore.js'

const props = defineProps({
  pollInterval: { type: Number, default: 5000 },
})

const emit = defineEmits(['memory-warning'])
const runtimeStore = useRuntimeStore()

const memoryData = ref(null)
const errorCount = ref(0)
const lastWarningLevel = ref(0)
let pollTimer = null

const circumference = 2 * Math.PI * 11 // ≈ 69.12

const levelClass = computed(() => {
  if (!memoryData.value) return 'level-loading'
  return `level-${memoryData.value.level}`
})

const usedPct = computed(() => {
  if (!memoryData.value) return 0
  return Math.round((1 - memoryData.value.available_pct) * 100)
})

const dashOffset = computed(() => {
  return circumference * (1 - usedPct.value / 100)
})

const tooltipText = computed(() => {
  if (!memoryData.value) return '正在检测系统内存...'
  const d = memoryData.value
  return (
    `${d.level_emoji} ${d.level_desc} | ` +
    `可用: ${d.available_gb.toFixed(1)} GB (${(d.available_pct * 100).toFixed(1)}%) | ` +
    `趋势: ${d.trend === 'dropping_fast' ? '⚠️ 快速下降' :
            d.trend === 'dropping' ? '📉 下降中' :
            d.trend === 'recovering' ? '📈 恢复中' : '➡️ 稳定'}`
  )
})

async function fetchMemory() {
  try {
    const data = await runtimeStore.refreshMemory()
    if (!data) throw new Error('memory unavailable')
    if (data.status === 'ok') {
      memoryData.value = data
      errorCount.value = 0
      if (data.level >= 3 && data.level !== lastWarningLevel.value) {
        emit('memory-warning', data)
      }
      lastWarningLevel.value = data.level
    } else {
      errorCount.value++
    }
  } catch (e) {
    errorCount.value++
    if (errorCount.value > 10) {
      memoryData.value = null
    }
  }
}

onMounted(() => {
  fetchMemory()
  pollTimer = setInterval(fetchMemory, props.pollInterval)
})

onUnmounted(() => {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
})
</script>

<style scoped>
.memory-ring {
  position: relative;
  width: 28px;
  height: 28px;
  flex-shrink: 0;
}

.memory-ring svg {
  width: 100%;
  height: 100%;
  transform: rotate(-90deg);
}

.ring-bg {
  fill: none;
  stroke: var(--border);
  stroke-width: 3;
  opacity: 0.3;
}

.ring-fill {
  fill: none;
  stroke-width: 3;
  stroke-linecap: round;
  transition: stroke-dashoffset 1s ease, stroke 0.5s ease;
}

.ring-text {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.55rem;
  font-weight: 600;
  font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
  line-height: 1;
}

/* 级别颜色 */
.level-0 .ring-fill { stroke: #166534; }
.level-0 .ring-text { color: #166534; }
.level-1 .ring-fill { stroke: #854d0e; }
.level-1 .ring-text { color: #854d0e; }
.level-2 .ring-fill { stroke: #9a3412; }
.level-2 .ring-text { color: #9a3412; }
.level-3 .ring-fill { stroke: #991b1b; }
.level-3 .ring-text { color: #991b1b; }
.level-4 .ring-fill { stroke: #374151; }
.level-4 .ring-text { color: #374151; }
.level-loading .ring-fill { stroke: #6b7280; }
.level-loading .ring-text { color: #6b7280; }

/* 深色模式 */
:root.theme-dark .level-0 .ring-fill { stroke: #86efac; }
:root.theme-dark .level-0 .ring-text { color: #86efac; }
:root.theme-dark .level-1 .ring-fill { stroke: #fde047; }
:root.theme-dark .level-1 .ring-text { color: #fde047; }
:root.theme-dark .level-2 .ring-fill { stroke: #fdba74; }
:root.theme-dark .level-2 .ring-text { color: #fdba74; }
:root.theme-dark .level-3 .ring-fill { stroke: #fca5a5; }
:root.theme-dark .level-3 .ring-text { color: #fca5a5; }
:root.theme-dark .level-4 .ring-fill { stroke: #9ca3af; }
:root.theme-dark .level-4 .ring-text { color: #9ca3af; }
:root.theme-dark .level-loading .ring-fill { stroke: #6b7280; }
:root.theme-dark .level-loading .ring-text { color: #6b7280; }

/* 告急闪烁 */
.level-3 .ring-fill,
.level-4 .ring-fill {
  animation: ring-pulse 1.5s ease-in-out infinite;
}

@keyframes ring-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
</style>
