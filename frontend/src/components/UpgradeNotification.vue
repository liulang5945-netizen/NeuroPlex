<template>
  <Transition name="upgrade-fade">
    <div v-if="visible" class="upgrade-overlay" @click.self="dismiss">
      <div class="upgrade-card">
        <!-- 标题 -->
        <div class="upgrade-header">
          <span class="upgrade-icon">🧠</span>
          <h3>态极模型升级建议</h3>
          <button class="close-btn" @click="dismiss">&times;</button>
        </div>

        <!-- 非升级状态：显示建议 -->
        <div v-if="upgradeState === 'idle'" class="upgrade-body">
          <div class="upgrade-info">
            <div class="info-row">
              <span class="info-label">当前模型</span>
              <span class="info-value">{{ suggestion.current_size || '125M' }}</span>
            </div>
            <div class="info-row">
              <span class="info-label">建议升级</span>
              <span class="info-value highlight">{{ suggestion.next_size || '350M' }}</span>
            </div>
            <div class="info-row">
              <span class="info-label">运行模式</span>
              <span class="info-value">{{ upgradeModeText }}</span>
            </div>
            <div class="info-row">
              <span class="info-label">硬件</span>
              <span class="info-value">
                内存 {{ suggestion.hardware?.ram_gb || '?' }}GB
                <template v-if="suggestion.hardware?.vram_gb">
                  / 显存 {{ suggestion.hardware.vram_gb }}GB
                </template>
              </span>
            </div>
          </div>

          <div v-if="suggestion.has_bottleneck" class="upgrade-reason">
            <strong>原因：</strong>{{ suggestion.bottleneck_reason || suggestion.message || '模型能力达到当前容量瓶颈' }}
          </div>

          <div class="upgrade-actions">
            <button class="btn-primary" @click="startUpgrade">
              🚀 立即升级
            </button>
            <button class="btn-secondary" @click="remindLater">
              ⏰ 5天后提醒
            </button>
            <button class="btn-ghost" @click="dismiss">
              暂时忽略
            </button>
          </div>
        </div>

        <!-- 升级进行中：显示进度 -->
        <div v-else-if="upgradeState === 'progress'" class="upgrade-body">
          <div class="progress-section">
            <div class="progress-bar-container">
              <div class="progress-bar" :style="{ width: progress + '%' }"></div>
            </div>
            <div class="progress-text">{{ progress }}%</div>
          </div>
          <div class="progress-message">{{ progressMessage }}</div>
        </div>

        <!-- 升级完成 -->
        <div v-else-if="upgradeState === 'done'" class="upgrade-body">
          <div class="upgrade-success">
            <span class="success-icon">✅</span>
            <p>{{ progressMessage || '升级完成！新模型已就绪。' }}</p>
          </div>
          <button class="btn-primary" @click="dismiss">好的</button>
        </div>

        <!-- 升级失败 -->
        <div v-else-if="upgradeState === 'error'" class="upgrade-body">
          <div class="upgrade-error">
            <span class="error-icon">❌</span>
            <p>{{ progressMessage || '升级失败' }}</p>
            <p class="error-detail" v-if="errorDetail">{{ errorDetail }}</p>
          </div>
          <button class="btn-secondary" @click="resetState">重试</button>
        </div>
      </div>
    </div>
  </Transition>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { API_BASE, authFetch } from '@/composables/apiClient.js'

const visible = ref(false)
const suggestion = ref({})
const upgradeState = ref('idle') // idle / progress / done / error
const progress = ref(0)
const progressMessage = ref('')
const errorDetail = ref('')

let checkTimer = null
let pollTimer = null

const upgradeModeText = computed(() => {
  const mode = suggestion.value.upgrade_mode
  if (mode === 'gpu') return 'GPU 加速'
  if (mode === 'cpu_quantized') return 'CPU 量化'
  return '未知'
})

// 检查是否应该提醒（5天后提醒）
function shouldShow() {
  const remindAt = localStorage.getItem('taiji_upgrade_remind_at')
  if (!remindAt) return true
  return Date.now() > parseInt(remindAt, 10)
}

// 检查升级建议
async function checkUpgrade() {
  try {
    const resp = await authFetch(`${API_BASE}/api/taiji_model/upgrade_check`)
    if (!resp.ok) return
    const data = await resp.json()

    if (data.can_upgrade && data.is_taiji && shouldShow()) {
      suggestion.value = data
      visible.value = true
    }
  } catch (e) {
    // 静默失败
  }
}

// 启动升级
async function startUpgrade() {
  upgradeState.value = 'progress'
  progress.value = 0
  progressMessage.value = '正在准备升级...'

  try {
    const resp = await authFetch(`${API_BASE}/api/taiji_model/upgrade`, { method: 'POST' })
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}))
      throw new Error(err.detail || `HTTP ${resp.status}`)
    }

    // 开始轮询进度
    pollTimer = setInterval(async () => {
      try {
        const progResp = await authFetch(`${API_BASE}/api/taiji_model/upgrade_progress`)
        if (!progResp.ok) return
        const progData = await progResp.json()

        progress.value = progData.progress || 0
        progressMessage.value = progData.message || ''

        if (progData.state === 'done') {
          clearInterval(pollTimer)
          pollTimer = null
          upgradeState.value = 'done'
          localStorage.removeItem('taiji_upgrade_remind_at')
        } else if (progData.state === 'error') {
          clearInterval(pollTimer)
          pollTimer = null
          upgradeState.value = 'error'
          errorDetail.value = progData.error || ''
          progressMessage.value = progData.message || '升级失败'
        }
      } catch (e) { console.debug('[UpgradeNotification] poll error:', e.message) }
    }, 2000)
  } catch (e) {
    upgradeState.value = 'error'
    progressMessage.value = e.message
  }
}

// 5天后提醒
function remindLater() {
  const fiveDaysMs = 5 * 24 * 60 * 60 * 1000
  localStorage.setItem('taiji_upgrade_remind_at', String(Date.now() + fiveDaysMs))
  visible.value = false
}

// 忽略本次
function dismiss() {
  visible.value = false
  if (upgradeState.value === 'done' || upgradeState.value === 'error') {
    upgradeState.value = 'idle'
  }
}

// 重置状态
function resetState() {
  upgradeState.value = 'idle'
  progress.value = 0
  progressMessage.value = ''
  errorDetail.value = ''
}

// 生命周期
onMounted(() => {
  // 启动后延迟 30 秒检查一次，之后每 5 分钟检查一次
  setTimeout(checkUpgrade, 30000)
  checkTimer = setInterval(checkUpgrade, 5 * 60 * 1000)
})

onUnmounted(() => {
  if (checkTimer) clearInterval(checkTimer)
  if (pollTimer) clearInterval(pollTimer)
})
</script>

<style scoped>
.upgrade-overlay {
  position: fixed;
  bottom: 24px;
  right: 24px;
  z-index: 9999;
  pointer-events: auto;
}

.upgrade-card {
  width: 380px;
  background: var(--bg-card, #1a1a2e);
  border: 1px solid var(--border, #333);
  border-radius: 16px;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
  overflow: hidden;
}

.upgrade-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 16px 20px;
  background: linear-gradient(135deg, var(--primary, #5b7a8a), var(--primary-hover, #4a6675));
  color: white;
}

.upgrade-icon {
  font-size: 1.5rem;
}

.upgrade-header h3 {
  margin: 0;
  flex: 1;
  font-size: 1rem;
  font-weight: 600;
}

.close-btn {
  background: none;
  border: none;
  color: white;
  font-size: 1.4rem;
  cursor: pointer;
  opacity: 0.7;
  padding: 0 4px;
}
.close-btn:hover { opacity: 1; }

.upgrade-body {
  padding: 20px;
}

.upgrade-info {
  display: grid;
  gap: 8px;
  margin-bottom: 16px;
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.9rem;
}

.info-label {
  color: var(--text-muted, #888);
}

.info-value {
  font-weight: 600;
  color: var(--text, #eee);
}

.info-value.highlight {
  color: var(--primary, #5b7a8a);
  font-size: 1.05em;
}

.upgrade-reason {
  padding: 10px 14px;
  background: var(--bg-muted, rgba(255,255,255,0.05));
  border-radius: 8px;
  font-size: 0.85rem;
  color: var(--text-secondary, #aaa);
  margin-bottom: 16px;
  border-left: 3px solid var(--primary, #5b7a8a);
}

.upgrade-actions {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.btn-primary {
  padding: 10px 20px;
  background: var(--primary-gradient, linear-gradient(135deg, #5b7a8a, #7a9aab));
  color: white;
  border: none;
  border-radius: 10px;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
}
.btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }

.btn-secondary {
  padding: 10px 20px;
  background: var(--bg-muted, rgba(255,255,255,0.08));
  color: var(--text, #eee);
  border: 1px solid var(--border, #444);
  border-radius: 10px;
  font-size: 0.9rem;
  cursor: pointer;
}
.btn-secondary:hover { background: var(--bg-card, rgba(255,255,255,0.12)); }

.btn-ghost {
  padding: 8px 16px;
  background: none;
  border: none;
  color: var(--text-muted, #888);
  font-size: 0.85rem;
  cursor: pointer;
}
.btn-ghost:hover { color: var(--text, #eee); }

/* 进度条 */
.progress-section {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.progress-bar-container {
  flex: 1;
  height: 8px;
  background: var(--bg-muted, rgba(255,255,255,0.1));
  border-radius: 4px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background: var(--primary-gradient, linear-gradient(90deg, #5b7a8a, #7a9aab));
  border-radius: 4px;
  transition: width 0.5s ease;
}

.progress-text {
  font-weight: 600;
  font-size: 0.95rem;
  color: var(--primary, #5b7a8a);
  min-width: 40px;
  text-align: right;
}

.progress-message {
  font-size: 0.85rem;
  color: var(--text-muted, #888);
}

/* 成功/失败 */
.upgrade-success, .upgrade-error {
  text-align: center;
  padding: 12px 0;
}

.success-icon, .error-icon {
  font-size: 2.5rem;
  display: block;
  margin-bottom: 8px;
}

.error-detail {
  font-size: 0.8rem;
  color: var(--text-muted, #888);
  margin-top: 4px;
}

/* 动画 */
.upgrade-fade-enter-active {
  transition: all 0.3s ease;
}
.upgrade-fade-leave-active {
  transition: all 0.2s ease;
}
.upgrade-fade-enter-from {
  opacity: 0;
  transform: translateY(20px);
}
.upgrade-fade-leave-to {
  opacity: 0;
  transform: translateY(10px);
}
</style>
