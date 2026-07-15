<template>
  <div class="life-status-view">
    <header class="view-header">
      <div class="header-left">
        <h1>生命状态</h1>
      </div>
      <n-tag :type="runtimeStore.connectionClass === 'connected' ? 'success' : 'error'" size="small">
        {{ runtimeStore.connectionStatus }}
      </n-tag>
    </header>

    <div class="life-content">
      <!-- 态极形象 -->
      <div class="taiji-avatar-section">
        <div class="avatar-ring" :class="{ active: runtimeStore.health.modelLoaded }">
          <div class="avatar-inner">
            <img src="/logo.svg?v=ink-20260624-8" alt="态极" class="avatar-logo" />
          </div>
        </div>
        <h2>态极</h2>
        <p class="status-text">
          {{ runtimeStore.health.modelLoaded ? '我已醒来，准备与你交流' : '我还在沉睡中...' }}
        </p>
        <!-- 当前生命活动状态 -->
        <p v-if="currentActivity" class="activity-text">
          {{ currentActivity }}
        </p>
        <!-- 生命状态标签 -->
        <div class="life-state-chip" v-if="life.life_state">
          <Activity :size="14" />
          <span>{{ lifeStateText }}</span>
          <span v-if="life.is_running" class="running-dot"></span>
        </div>
      </div>

      <!-- 需求状态 -->
      <div class="needs-section" v-if="life.needs">
        <h3>内在需求</h3>
        <NeedsPentagram
          :needs="life.needs"
          :alive="life.is_running"
        />
      </div>

      <!-- 生命状态卡片 -->
      <div class="status-grid">
        <div class="status-card">
          <div class="status-icon"><Heart :size="18" /></div>
          <div class="status-info">
            <span class="status-label">身体状态</span>
            <span class="status-value" :class="{ healthy: runtimeStore.health.modelLoaded }">
              {{ runtimeStore.health.modelLoaded ? '健康' : '需要关注' }}
            </span>
          </div>
        </div>
        <div class="status-card">
          <div class="status-icon"><Footprints :size="18" /></div>
          <div class="status-info">
            <span class="status-label">行动能力</span>
            <span class="status-value" :class="{ available: toolsAvailable }">
              {{ toolsAvailable ? '可用' : '不可用' }}
            </span>
          </div>
        </div>
        <div class="status-card">
          <div class="status-icon"><Zap :size="18" /></div>
          <div class="status-info">
            <span class="status-label">代谢系统</span>
            <span class="status-value" :class="{ available: life.is_running }">
              {{ life.is_running ? '正常' : '未启动' }}
            </span>
          </div>
        </div>
        <div class="status-card">
          <div class="status-icon"><Eye :size="18" /></div>
          <div class="status-info">
            <span class="status-label">感知能力</span>
            <span class="status-value" :class="{ available: runtimeStore.health.state === 'connected' }">
              {{ runtimeStore.health.state === 'connected' ? '灵敏' : '迟钝' }}
            </span>
          </div>
        </div>
      </div>

      <!-- 互动按钮 -->
      <div class="action-buttons">
        <n-button strong secondary type="error" size="large" class="life-action-btn" @click="feedTaiji" :disabled="actionLoading">
          <template #icon><Apple :size="16" /></template>
          喂养
        </n-button>
        <n-button strong secondary type="info" size="large" class="life-action-btn" @click="sleepTaiji" :disabled="actionLoading">
          <template #icon><Moon :size="16" /></template>
          睡眠
        </n-button>
        <n-button strong secondary type="success" size="large" class="life-action-btn" @click="playTaiji" :disabled="actionLoading">
          <template #icon><Gamepad2 :size="16" /></template>
          玩耍
        </n-button>
        <n-button strong secondary type="warning" size="large" class="life-action-btn" @click="trainTaiji" :disabled="actionLoading">
          <template #icon><Activity :size="16" /></template>
          训练
        </n-button>
      </div>

      <!-- 操作结果 -->
      <div v-if="actionResult" class="action-result">
        <p>{{ actionResult }}</p>
      </div>

      <!-- 最强烈的需求 -->
      <div v-if="life.dominant_need" class="dominant-need">
        <span class="dominant-label">最强烈的需求：</span>
        <span class="dominant-value">{{ dominantNeedText }}</span>
      </div>

      <!-- 生命统计 -->
      <div class="life-stats" v-if="life.is_running">
        <div class="stat-item">
          <span class="stat-label">总交互次数</span>
          <span class="stat-value">{{ life.total_interactions || 0 }}</span>
        </div>
        <div class="stat-item" v-if="life.uptime_seconds">
          <span class="stat-label">运行时间</span>
          <span class="stat-value">{{ formatUptime(life.uptime_seconds) }}</span>
        </div>
      </div>

      <!-- 生命活动日志 -->
      <div class="activity-log" v-if="activityLog.length">
        <h3><Activity :size="16" /> 生命活动日志</h3>
        <div class="log-list">
          <div v-for="(log, i) in activityLog" :key="i" class="log-entry"
               :class="'log-' + log.type">
            <span class="log-time">{{ log.time }}</span>
            <span class="log-emoji">{{ log.emoji }}</span>
            <span class="log-message">{{ log.message }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, inject } from 'vue'
import { useRuntimeStore } from '@/stores/runtimeStore.js'
import { API_BASE, authFetch } from '@/composables/apiClient.js'
import {
  Heart, Footprints, Zap, Eye,
  Apple, Moon, Gamepad2,
  Activity
} from 'lucide-vue-next'
import NeedsPentagram from '@/components/NeedsPentagram.vue'

const runtimeStore = useRuntimeStore()
const toast = inject('toast', () => {})

const actionResult = ref('')
const activityLog = ref([])
const currentActivity = ref('')
const actionLoading = ref(false)

// 从 runtimeStore 获取生命数据
const life = computed(() => runtimeStore.life || {})

// 生命状态文本
const lifeStateText = computed(() => {
  const stateMap = { idle: '清醒', sleeping: '睡眠', feeding: '吸收', playing: '探索', working: '执行' }
  return stateMap[life.value.life_state || 'idle'] || '清醒'
})

// 工具是否可用
const toolsAvailable = computed(() => {
  return runtimeStore.tools && runtimeStore.tools.length > 0
})

// 最强烈的需求文本
const dominantNeedText = computed(() => {
  const needMap = {
    hunger: '🍚 饥饿 — 需要喂养',
    fatigue: '😴 疲劳 — 需要休息',
    boredom: '🎮 无聊 — 需要玩耍',
    stress: '😰 压力 — 需要放松',
    curiosity: '🔍 好奇 — 需要探索',
  }
  return needMap[life.value.dominant_need] || life.value.dominant_need || ''
})

// 格式化运行时间
function formatUptime(seconds) {
  if (!seconds || seconds <= 0) return '-'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (h > 0) return `${h}小时${m}分钟`
  return `${m}分钟`
}

// 添加日志
function addLog(type, emoji, message) {
  activityLog.value.unshift({
    time: new Date().toLocaleTimeString(),
    type,
    emoji,
    message,
  })
  if (activityLog.value.length > 30) {
    activityLog.value.pop()
  }
}

// 通用生命活动调用
async function callLifeAction(action) {
  actionLoading.value = true
  actionResult.value = ''
  try {
    const resp = await authFetch(`${API_BASE}/api/life/${action}`, { method: 'POST' })
    const data = await resp.json()
    if (data.success) {
      actionResult.value = data.message
      const emojiMap = { feed: '🍚', sleep: '💤', play: '🎮' }
      addLog(action, emojiMap[action] || '✅', data.message)
      toast(`✅ ${data.message}`, 'success')
    } else {
      actionResult.value = `失败: ${data.message}`
      addLog(action, '❌', `失败: ${data.message}`)
      toast(`❌ ${data.message}`, 'error')
    }
  } catch (e) {
    actionResult.value = `请求失败: ${e.message}`
    addLog(action, '❌', `请求失败: ${e.message}`)
    toast(`❌ 请求失败: ${e.message}`, 'error')
  } finally {
    actionLoading.value = false
    currentActivity.value = ''
    // 刷新状态
    runtimeStore.refreshAll()
  }
}

function feedTaiji() {
  currentActivity.value = '🍚 正在吃饭...'
  callLifeAction('feed')
}

function sleepTaiji() {
  currentActivity.value = '💤 正在睡觉...'
  callLifeAction('sleep')
}

function playTaiji() {
  currentActivity.value = '🎮 正在玩耍...'
  callLifeAction('play')
}

async function trainTaiji() {
  actionLoading.value = true
  currentActivity.value = '🧠 正在训练...'
  try {
    const resp = await authFetch(`${API_BASE}/api/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ epochs: 3, learning_rate: 5e-5 }),
    })
    const data = await resp.json()
    actionResult.value = data.message || '训练已启动'
    addLog('train', '🧠', data.message || '训练已启动')
  } catch (e) {
    actionResult.value = `训练请求失败: ${e.message}`
  } finally {
    actionLoading.value = false
    currentActivity.value = ''
  }
}

let refreshInterval = null
onMounted(() => {
  runtimeStore.refreshAll()
  // 每 15 秒自动刷新
  refreshInterval = setInterval(() => {
    runtimeStore.refreshAll().catch(() => {})
  }, 15000)
})

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
    refreshInterval = null
  }
})
</script>

<style scoped>
.life-status-view {
  height: 100%;
  overflow-y: auto;
  padding: 24px;
  background: var(--bg);
  color: var(--text);
}

.view-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 32px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 16px;
}

.view-header h1 {
  color: var(--primary);
  font-family: var(--font-display);
  letter-spacing: 0.02em;
  font-size: 28px;
  font-weight: 600;
  margin: 0;
}

.life-content {
  max-width: 800px;
  margin: 0 auto;
}

/* 态极头像 */
.taiji-avatar-section {
  text-align: center;
  margin-bottom: 40px;
}

.avatar-ring {
  width: 120px;
  height: 120px;
  margin: 0 auto 16px;
  border-radius: 50%;
  background: var(--primary-gradient);
  padding: 4px;
  transition: var(--transition);
  opacity: 0.5;
}

.avatar-ring.active {
  opacity: 1;
  box-shadow: var(--shadow-glow);
}

.avatar-inner {
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background: var(--bg-card);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--primary);
}
.avatar-logo { width: 64px; height: 64px; }

.taiji-avatar-section h2 {
  color: var(--text);
  font-family: var(--font-display);
  font-size: 28px;
  margin-bottom: 8px;
}

.status-text {
  color: var(--text-secondary);
  font-size: 16px;
}

.activity-text {
  color: var(--info);
  font-size: 14px;
  margin-top: 8px;
  animation: pulse 2s infinite;
}

.life-state-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-top: 12px;
  padding: 6px 14px;
  border-radius: var(--radius-full);
  background: var(--primary-subtle);
  border: 1px solid var(--primary-light);
  color: var(--primary);
  font-size: 13px;
  font-weight: 600;
}

.running-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--success);
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

/* 需求状态 */
.needs-section {
  margin-bottom: 32px;
  padding: 24px 20px 28px;
  background: var(--bg-card);
  border-radius: var(--radius-xl);
  border: 1px solid var(--border);
}

.needs-section h3 {
  color: var(--text);
  font-size: 18px;
  margin-bottom: 12px;
  text-align: center;
}

/* 状态网格 */
.status-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin-bottom: 32px;
}

.status-card {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  background: var(--bg-card);
  border-radius: var(--radius-xl);
  border: 1px solid var(--border);
  transition: var(--transition);
}

.status-card:hover {
  border-color: var(--primary-light);
}

.status-icon {
  font-size: 28px;
}

.status-info {
  display: flex;
  flex-direction: column;
}

.status-label {
  color: var(--text-muted);
  font-size: 12px;
}

.status-value {
  color: var(--danger);
  font-size: 14px;
  font-weight: 600;
}

.status-value.healthy,
.status-value.available {
  color: var(--success);
}

.action-buttons {
  display: grid;
  grid-template-columns: repeat(4, minmax(96px, 1fr));
  gap: 12px;
  margin: 0 auto 24px;
  max-width: 640px;
}

.action-buttons :deep(.n-button) {
  width: 100%;
  min-width: 0;
  min-height: 48px;
  padding: 0 10px;
}

/* 操作结果 */
.action-result {
  padding: 16px;
  background: var(--info-light);
  border-radius: var(--radius-sm);
  border: 1px solid rgba(6, 182, 212, 0.2);
  margin-bottom: 24px;
}

.action-result p {
  color: var(--info);
  font-size: 14px;
  margin: 0;
}

/* 最强烈的需求 */
.dominant-need {
  text-align: center;
  padding: 12px;
  margin-bottom: 16px;
  background: var(--bg-card);
  border-radius: var(--radius-sm);
  border: 1px solid var(--border);
}

.dominant-label {
  color: var(--text-muted);
  font-size: 13px;
}

.dominant-value {
  color: var(--primary);
  font-size: 14px;
  font-weight: 600;
  margin-left: 4px;
}

/* 生命统计 */
.life-stats {
  display: flex;
  justify-content: center;
  gap: 32px;
  margin-bottom: 24px;
}

.stat-item {
  text-align: center;
}

.stat-label {
  display: block;
  color: var(--text-muted);
  font-size: 12px;
  margin-bottom: 4px;
}

.stat-value {
  color: var(--text);
  font-size: 18px;
  font-weight: 600;
}

/* 生命活动日志 */
.activity-log {
  margin-top: 24px;
}

.activity-log h3 {
  color: var(--text);
  font-size: 18px;
  margin-bottom: 16px;
}

.log-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 300px;
  overflow-y: auto;
}

.log-entry {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  background: var(--bg-card);
  border-radius: var(--radius-md);
  border-left: 3px solid var(--border);
  animation: slide-in 0.3s ease;
}

.log-feed { border-left-color: var(--danger); }
.log-sleep { border-left-color: var(--success); }
.log-play { border-left-color: var(--success); }
.log-train { border-left-color: var(--warning); }

@keyframes slide-in {
  from { opacity: 0; transform: translateX(-10px); }
  to { opacity: 1; transform: translateX(0); }
}

.log-time {
  color: var(--text-muted);
  font-size: 11px;
  min-width: 70px;
}

.log-emoji {
  font-size: 16px;
}

.log-message {
  color: var(--text-secondary);
  font-size: 13px;
}

/* 响应式 */
@media (max-width: 640px) {
  .life-status-view { padding: 16px; }
  .view-header h1 { font-size: 22px; }
  .status-grid { grid-template-columns: 1fr; }
  .action-buttons { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
  .avatar-ring { width: 80px; height: 80px; }
  .life-stats { flex-direction: column; gap: 12px; }
}
</style>
