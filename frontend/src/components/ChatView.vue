<template>
  <main class="chat-workbench">
    <header class="chat-topbar">
      <div class="chat-title-block">
        <div class="title-row">
          <h1>{{ chatStore.currentSessionName || '态极对话' }}</h1>
        </div>
        <div class="life-strip">
          <span class="life-chip" :class="runtimeStore.life.life_state || 'idle'">
            <Activity :size="12" />
            {{ lifeStateText }}
          </span>
          <span class="metric-chip">精力 {{ energyPercent }}%</span>
          <span class="metric-chip">饱腹 {{ satietyPercent }}%</span>
          <span class="metric-chip">好奇 {{ curiosityPercent }}%</span>
          <span v-if="dominantNeedLabel" class="metric-chip dominant-need" title="最迫切需求">
            {{ dominantNeedLabel }} {{ dominantNeedValue }}%
          </span>
        </div>
      </div>
      <div class="topbar-actions">
        <span class="engine-pill">ReAct 引擎</span>
        <button class="icon-btn" @click="chatStore.clearCurrentChat()" :title="t('clear_chat')">
          <RotateCcw :size="15" />
        </button>
      </div>
    </header>

    <section class="conversation-shell">
      <div class="messages-area" ref="messagesArea">
        <div class="chat-thread">
          <div v-if="runtimeNotice" class="runtime-notice" :class="runtimeStore.connectionClass">
            <span class="runtime-notice-dot"></span>
            <div>
              <strong>{{ runtimeNotice.title }}</strong>
              <p>{{ runtimeNotice.message }}</p>
            </div>
          </div>

          <div v-if="chatStore.messages.length === 0" class="empty-state">
            <div class="empty-logo-wrap">
              <TaijiLogo :size="72" />
            </div>
            <h2 class="empty-title">今天要推进什么？</h2>
            <p class="empty-desc">把问题、文件或任务交给态极，它会按当前运行状态接管执行。</p>
            <div class="empty-hints">
              <div class="hint-card" v-for="hint in quickHints" :key="hint.text" @click="chatStore.chatInput = hint.text">
                <component :is="hint.icon" :size="15" />
                <span>{{ hint.text }}</span>
              </div>
            </div>
          </div>

          <div v-if="hasMoreMessages" class="load-more-row">
            <button class="load-more-btn" @click="showMore">
              显示更多消息（{{ messageLimit }}/{{ chatStore.messages.length }}）
            </button>
          </div>
          <article v-for="(msg, index) in displayedMessages" :key="msg.id"
            :class="['message-row', msg.role]"
            v-memo="[msg.id, msg.content, msg.role]">
            <div class="message-avatar">
              <User v-if="msg.role === 'user'" :size="15" />
              <Bot v-else :size="15" />
            </div>
            <div class="message-main">
              <div class="message-meta">
                <span>{{ msg.role === 'user' ? '你' : '态极' }}</span>
                <span class="mode-label">{{ msg.role === 'user' ? '用户输入' : '态极回应' }}</span>
              </div>
              <div class="bubble">
                <div v-if="msg.role === 'user'" class="text-content">{{ msg.content }}</div>
                <div v-else class="markdown-body" v-html="renderMarkdown(msg.content)" />
              </div>
              <div v-if="msg.role === 'assistant' && msg.content" class="msg-actions">
                <button class="msg-action-btn" @click="copyMsg(msg.content)" title="复制"><Copy :size="13" /></button>
                <button class="msg-action-btn" @click="chatStore.regenerateMessage(msg.id)" title="重新生成"><RotateCcw :size="13" /></button>
              </div>
            </div>
          </article>

          <article v-if="chatStore.isLoading" class="message-row assistant thinking-row">
            <div class="message-avatar breathing"><Bot :size="15" /></div>
            <div class="message-main">
              <div class="message-meta"><span>态极</span><span class="mode-label">{{ chatStore.isReceiving ? '正在回应' : '正在启动' }}</span></div>
              <div v-if="!chatStore.isReceiving" class="bubble loading-bubble">
                <span class="thinking-animation"><span class="think-dot"></span><span class="think-dot"></span><span class="think-dot"></span></span>
              </div>
            </div>
          </article>
        </div>
      </div>

      <footer class="composer-area">
        <div class="stop-container" v-if="chatStore.isReceiving">
          <button class="stop-btn" @click="chatStore.stopGeneration()">
            <Square :size="13" fill="currentColor" /> 中断执行
          </button>
        </div>

        <!-- 多模态工具栏 -->
        <div class="multimodal-toolbar">
          <button class="mm-btn" :class="{ active: isRecording }" @click="toggleVoice" disabled title="语音输入">
            <Mic :size="16" />
            <span class="mm-label">{{ isRecording ? '停止' : '语音' }}</span>
          </button>
          <label class="mm-btn" title="上传图片">
            <ImageIcon :size="16" />
            <span class="mm-label">图片</span>
            <input type="file" accept="image/*" @change="onImageSelect" style="display:none" />
          </label>
          <label class="mm-btn" title="上传视频">
            <Video :size="16" />
            <span class="mm-label">视频</span>
            <input type="file" accept="video/*" @change="onVideoSelect" style="display:none" />
          </label>
          <label class="mm-btn" title="上传文件">
            <FileText :size="16" />
            <span class="mm-label">文件</span>
            <input type="file" multiple @change="onFileSelect" style="display:none" />
          </label>
          <button class="mm-btn" @click="toggleCamera" disabled title="拍照/录像">
            <Camera :size="16" />
            <span class="mm-label">拍照</span>
          </button>
        </div>

        <div class="input-container">
          <textarea ref="inputRef" v-model="chatStore.chatInput"
            :placeholder="inputPlaceholder"
            rows="1" @keydown="onKeydown" />
          <button class="send-btn" :class="{ unavailable: !canSend }" :disabled="!canSend" @click="handleSend" title="发送">
            <Send :size="16" />
          </button>
        </div>
        <div class="composer-hint">Enter 发送 · Shift+Enter 换行 · 语音/图片/视频/文件</div>
      </footer>
    </section>
  </main>
</template>

<script setup>
defineOptions({ name: 'ChatView' })
import { ref, computed, watch, nextTick, onMounted, inject } from 'vue'
import { Activity, User, Bot, RotateCcw, Copy, Square, Send, Lightbulb, Code, BookOpen, Mic, Image as ImageIcon, Video, FileText, Camera } from 'lucide-vue-next'
import TaijiLogo from './TaijiLogo.vue'
import { useChatStore } from '@/stores/chatStore.js'
import { useAppStore } from '@/stores/appStore.js'
import { useRuntimeStore } from '@/stores/runtimeStore.js'
import { useMarkdown } from '@/composables/useMarkdown.js'
import { authFetch } from '@/composables/apiClient.js'

const chatStore = useChatStore()
const appStore = useAppStore()
const runtimeStore = useRuntimeStore()
const { renderMarkdown } = useMarkdown()
const toast = inject('toast', () => {})
const t = (key) => appStore.t(key)

const messagesArea = ref(null)
const inputRef = ref(null)
const engineModel = ref('agent')  // 统一使用 ReAct 引擎

const energyPercent = computed(() => Math.max(0, 100 - (runtimeStore.life.needs?.fatigue || 0)).toFixed(0))
const satietyPercent = computed(() => Math.max(0, 100 - (runtimeStore.life.needs?.hunger || 0)).toFixed(0))
const curiosityPercent = computed(() => Math.max(0, runtimeStore.life.needs?.curiosity || 0).toFixed(0))
const lifeStateText = computed(() => ({ idle: '清醒', sleeping: '睡眠', feeding: '吸收', playing: '探索', working: '执行' }[runtimeStore.life.life_state || 'idle'] || ''))
const needIcons = { hunger: '饿', fatigue: '累', boredom: '闷', stress: '压', curiosity: '奇' }
const dominantNeedKey = computed(() => runtimeStore.life.dominant_need || '')
const dominantNeedLabel = computed(() => needIcons[dominantNeedKey.value] || '')
const dominantNeedValue = computed(() => {
  const needs = runtimeStore.life.needs || {}
  return dominantNeedKey.value ? Math.round(needs[dominantNeedKey.value] || 0) : 0
})
const runtimeNotice = computed(() => runtimeStore.runtimeNotice)

const quickHints = [
  { icon: Lightbulb, text: '梳理一个复杂问题' },
  { icon: Code, text: '检查并改进一段代码' },
  { icon: BookOpen, text: '总结一份项目资料' },
]

const canSend = computed(() =>
  !!chatStore.chatInput.trim() && !chatStore.isLoading && runtimeStore.health.state === 'connected' && runtimeStore.health.modelLoaded
)

function scrollToBottom() { nextTick(() => { if (messagesArea.value) messagesArea.value.scrollTop = messagesArea.value.scrollHeight }) }
watch(() => chatStore.messages.length, scrollToBottom)
watch(() => chatStore.isReceiving, scrollToBottom)

const isRecording = ref(false)
const messagePageSize = 50
const messageLimit = ref(messagePageSize)
const displayedMessages = computed(() => {
  const msgs = chatStore.messages
  return msgs.length > messageLimit.value ? msgs.slice(-messageLimit.value) : msgs
})
const hasMoreMessages = computed(() => chatStore.messages.length > messageLimit.value)
function showMore() { messageLimit.value += messagePageSize }
const inputPlaceholder = computed(() => {
  if (isRecording.value) return '正在录音...'
  return '输入任务、问题或文件说明'
})

function handleSend() {
  if (!canSend.value) return
  chatStore.sendMessage(engineModel.value)
  scrollToBottom()
}
function onKeydown(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() } }

// 语音输入
async function toggleVoice() {
  if (isRecording.value) {
    isRecording.value = false
    // TODO: 停止录音并发送音频
    toast('语音功能开发中', 'info')
  } else {
    isRecording.value = true
    // TODO: 开始录音
    toast('语音功能开发中', 'info')
  }
}

// 图片上传
async function onImageSelect(e) {
  const file = e.target.files[0]
  if (!file) return
  const formData = new FormData()
  formData.append('file', file)
  try {
    const resp = await authFetch('/api/taiji/upload', { method: 'POST', body: formData })
    if (resp.ok) {
      const data = await resp.json()
      chatStore.chatInput += `[图片: ${data.filename}] `
      toast('图片已上传', 'success')
    } else {
      toast('图片上传失败', 'error')
    }
  } catch (e) {
    console.warn('[ChatView] image upload failed:', e.message)
    toast('图片上传失败', 'error')
  }
  e.target.value = ''
}

// 视频上传
async function onVideoSelect(e) {
  const file = e.target.files[0]
  if (!file) return
  const formData = new FormData()
  formData.append('file', file)
  try {
    const resp = await authFetch('/api/taiji/upload', { method: 'POST', body: formData })
    if (resp.ok) {
      const data = await resp.json()
      chatStore.chatInput += `[视频: ${data.filename}] `
      toast('视频已上传', 'success')
    } else {
      toast('视频上传失败', 'error')
    }
  } catch (e) {
    console.warn('[ChatView] video upload failed:', e.message)
    toast('视频上传失败', 'error')
  }
  e.target.value = ''
}

// 文件上传
async function onFileSelect(e) {
  const files = e.target.files
  if (!files.length) return
  for (const file of files) {
    const formData = new FormData()
    formData.append('file', file)
    try {
      const resp = await authFetch('/api/taiji/upload', { method: 'POST', body: formData })
      if (resp.ok) {
        const data = await resp.json()
        chatStore.chatInput += `[文件: ${data.filename}] `
      }
    } catch (e) { console.warn('[ChatView] file upload failed:', e.message) }
  }
  toast(`已上传 ${files.length} 个文件`, 'success')
  e.target.value = ''
}

// 拍照/录像
async function toggleCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    // TODO: 实现拍照/录像逻辑
    toast('摄像头功能开发中', 'info')
    stream.getTracks().forEach(t => t.stop())
  } catch (e) {
    console.warn('[ChatView] camera access denied:', e.message)
    toast('无法访问摄像头', 'error')
  }
}

async function copyMsg(content) { try { await navigator.clipboard.writeText(content); toast('已复制', 'success') } catch { toast('复制失败', 'error') } }

onMounted(scrollToBottom)
</script>

<style scoped>
.chat-workbench {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--bg);
  color: var(--text);
}

/* 顶栏 */
.chat-topbar {
  min-height: 60px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 11px 24px;
  border-bottom: 1px solid var(--border);
  background: var(--toolbar-bg);
  flex-shrink: 0;
  position: relative;
  z-index: 5;
}
.chat-title-block { min-width: 0; display: flex; flex-direction: column; gap: 6px; }
.title-row { display: flex; align-items: center; gap: 8px; color: var(--text); }
.title-row svg { color: var(--primary-hover); }
.title-row h1 {
  margin: 0;
  font-family: var(--font-display);
  letter-spacing: 0.02em;
  font-size: 16px;
  font-weight: 700;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  letter-spacing: 0;
}

.life-strip { display: flex; gap: 6px; flex-wrap: wrap; }
.life-chip, .metric-chip {
  display: inline-flex; align-items: center; gap: 4px; height: 22px; padding: 0 7px;
  border-radius: var(--radius-md); font-size: 11px; color: var(--text-muted);
  border: 1px solid var(--border); background: var(--bg-muted);
}
.life-chip {
  color: var(--primary);
  background: var(--primary-subtle);
  border-color: var(--primary-light);
  font-weight: 600;
}
.life-chip.sleeping { color: var(--success); }
.life-chip.feeding { color: var(--warning); }
.life-chip.playing { color: var(--success); }

.life-strip {
  animation: taijiBreathe 8s ease-in-out infinite;
}

.metric-chip.dominant-need {
  background: var(--danger-light);
  border-color: var(--danger);
  color: var(--danger);
  font-weight: 600;
  animation: critical-pulse 2s ease-in-out infinite;
}

@media (prefers-reduced-motion: reduce) {
  .life-strip { animation: none; }
  .metric-chip.dominant-need { animation: none; }
}

.topbar-actions { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
.engine-pill {
  display: inline-flex;
  align-items: center;
  height: 28px;
  padding: 0 9px;
  border: 1px solid var(--border);
  border-radius: var(--radius-full);
  color: var(--text-secondary);
  background: var(--bg-muted);
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
}
.mode-switch {
  display: flex; padding: 3px; border-radius: var(--radius-md); background: var(--bg-muted);
  border: 1px solid var(--border);
}
.mode-switch button, .icon-btn, .msg-action-btn, .send-btn, .stop-btn { font: inherit; }
.mode-switch button {
  height: 30px; padding: 0 12px; display: inline-flex; align-items: center; justify-content: center; gap: 5px;
  border: 0; border-radius: var(--radius-sm); background: transparent; color: var(--text-secondary); cursor: pointer; font-size: 12px;
}
.mode-switch button.active { background: var(--bg-card); color: var(--primary); box-shadow: var(--shadow-sm); }
.icon-btn {
  width: 32px; height: 32px; display: inline-flex; align-items: center; justify-content: center;
  border: 1px solid var(--border); border-radius: var(--radius-lg); color: var(--text-secondary);
  background: var(--bg-card); cursor: pointer; transition: var(--transition-fast);
}
.icon-btn:hover { color: var(--text); background: var(--bg-hover); border-color: var(--border-strong); }

/* 对话区 */
.conversation-shell { min-height: 0; flex: 1; display: flex; flex-direction: column; }
.messages-area { flex: 1; min-height: 0; overflow-y: auto; padding: 22px; }
.chat-thread { width: min(100%, 820px); margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }

/* 运行状态 */
.runtime-notice {
  display: flex; align-items: flex-start; gap: 10px; padding: 14px 16px;
  border-radius: var(--radius-xl); background: var(--bg-card); border: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
}
.runtime-notice-dot { width: 8px; height: 8px; margin-top: 5px; border-radius: 50%; background: var(--text-muted); flex-shrink: 0; }
.runtime-notice.loading .runtime-notice-dot, .runtime-notice.connecting .runtime-notice-dot { background: var(--warning); }
.runtime-notice.error .runtime-notice-dot { background: var(--danger); }
.runtime-notice.connected .runtime-notice-dot { background: var(--success); }
.runtime-notice strong { display: block; color: var(--text); font-size: 13px; font-weight: 650; }
.runtime-notice p { margin: 3px 0 0; color: var(--text-muted); font-size: 12px; line-height: 1.5; }

/* 空状态 */
.empty-state {
  min-height: 360px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  text-align: center;
  padding: 40px 16px;
}
.empty-logo-wrap {
  display: inline-flex; align-items: center; justify-content: center;
  width: 78px; height: 78px; border-radius: 26px;
  background: var(--primary-subtle);
  border: 1px solid var(--primary-light);
  margin-bottom: 8px;
}
.empty-title {
  margin: 0;
  font-size: 22px;
  font-weight: 750;
  letter-spacing: 0;
  color: var(--text);
}
.empty-desc { margin: 0; max-width: 420px; font-size: 13px; line-height: 1.65; color: var(--text-secondary); }

.empty-hints { display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin-top: 16px; }
.hint-card {
  display: flex;
  align-items: center;
  gap: 7px;
  min-height: 38px;
  padding: 9px 14px;
  border-radius: var(--radius-xl);
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  font-size: 13px;
  cursor: pointer;
  transition: var(--transition-fast);
  box-shadow: var(--shadow-sm);
}
.hint-card:hover {
  color: var(--primary-hover);
  border-color: var(--primary-light);
  background: var(--primary-subtle);
}

/* Load more */
.load-more-row { text-align: center; padding: 8px 0 4px; }
.load-more-btn {
  background: var(--bg-muted); border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md); color: var(--text-muted);
  font-size: 12px; padding: 6px 16px; cursor: pointer;
  transition: background-color 0.15s var(--ease), color 0.15s var(--ease);
}
.load-more-btn:hover { background: var(--bg-hover); color: var(--text-secondary); }

/* 消息 */
.message-row {
  display: flex; gap: 12px; align-items: flex-start;
  content-visibility: auto;
  contain-intrinsic-size: auto 80px;
}
.message-row.user { flex-direction: row-reverse; }
.message-row.user .message-main { align-items: flex-end; }
.message-avatar {
  width: 32px; height: 32px; display: flex; align-items: center; justify-content: center;
  border-radius: var(--radius-xl); flex-shrink: 0; margin-top: 2px;
}
.message-row.assistant .message-avatar { background: var(--primary-subtle); color: var(--primary); border: 1px solid var(--primary-light); }
.message-row.user .message-avatar { background: var(--bg-elevated); color: var(--text); border: 1px solid var(--border); }

.message-main { min-width: 0; flex: 1; display: flex; flex-direction: column; gap: 4px; }
.message-meta { display: flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600; color: var(--text); }
.message-row.user .message-meta { flex-direction: row-reverse; }
.mode-label { font-weight: 400; color: var(--text-muted); font-size: 11px; }

.bubble {
  max-width: min(100%, 760px);
  padding: 13px 15px;
  border-radius: var(--radius-xl);
  background: var(--bg-card);
  border: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
  line-height: 1.72;
  font-size: 14px;
}
.message-row.user .bubble { color: var(--text); background: var(--bg-elevated); border-color: var(--border-strong); }
.message-row.assistant .bubble { border-left: 2px solid var(--primary); }
.text-content { white-space: pre-wrap; word-break: break-word; }

.msg-actions { display: flex; gap: 4px; opacity: 0; transition: opacity 0.15s; margin-top: 4px; }
.message-row.user .msg-actions { justify-content: flex-end; }
.message-main:hover .msg-actions { opacity: 1; }
.msg-action-btn {
  width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;
  border: 1px solid var(--border); border-radius: var(--radius-sm); color: var(--text-muted);
  background: var(--bg-card); cursor: pointer; transition: var(--transition-fast);
}
.msg-action-btn:hover { color: var(--primary); background: var(--primary-subtle); }

/* 思考动画 */
.thinking-row .bubble { width: fit-content; }
.thinking-animation { display: flex; gap: 5px; align-items: center; padding: 4px 0; }
.think-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--primary); animation: dotBounce 1.2s ease-in-out infinite; }
.think-dot:nth-child(2) { animation-delay: 0.15s; }
.think-dot:nth-child(3) { animation-delay: 0.3s; }

/* 输入区 */
.composer-area {
  flex-shrink: 0; padding: 14px 22px 16px; border-top: 1px solid var(--border);
  background: var(--toolbar-bg);
  position: relative; z-index: 5;
}
.composer-area > * { width: min(100%, 820px); margin-left: auto; margin-right: auto; }

.composer-hint { text-align: center; font-size: 11px; color: var(--text-muted); margin-top: 6px; }

/* 多模态工具栏 */
.multimodal-toolbar {
  display: flex; align-items: center; justify-content: center; gap: 6px;
  margin-bottom: 8px; padding: 2px 0;
  flex-wrap: wrap;
}
.mm-btn {
  display: inline-flex; align-items: center; gap: 4px; height: 30px; padding: 0 10px;
  border: 1px solid var(--border); border-radius: var(--radius-full);
  background: var(--bg-card); color: var(--text-secondary);
  cursor: pointer; font-size: 12px; transition: var(--transition-fast);
}
.mm-btn:hover { color: var(--primary-hover); border-color: var(--primary-light); background: var(--primary-subtle); }
.mm-btn.active { color: var(--danger); border-color: var(--danger); background: var(--danger-light); }
.mm-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.mm-btn:disabled:hover { color: var(--text-secondary); border-color: var(--border); background: var(--bg-card); }
.mm-label { font-size: 11px; }

.stop-container { display: flex; justify-content: center; margin-bottom: 8px; }
.stop-btn {
  display: inline-flex; align-items: center; gap: 5px; height: 32px; padding: 0 16px;
  border: 1px solid var(--danger); border-radius: var(--radius-full); color: var(--danger);
  background: rgba(239,68,68,0.08); cursor: pointer; font-size: 12px; transition: var(--transition);
}
.stop-btn:hover { background: rgba(239,68,68,0.15); transform: scale(1.02); }

.input-container {
  display: grid; grid-template-columns: minmax(0, 1fr) 38px; align-items: end; gap: 8px;
  padding: 11px 12px 11px 14px; border: 1px solid var(--border); border-radius: var(--radius-md);
  background: var(--bg-input); transition: var(--transition-fast); box-shadow: var(--shadow-sm);
}
.input-container:focus-within { border-color: var(--primary); box-shadow: var(--shadow-glow); }

textarea {
  flex: 1; border: 0; outline: none; background: transparent; color: var(--text);
  font-family: var(--font); font-size: 14px; line-height: 1.65; resize: none;
  padding: 4px 0; min-height: 24px; max-height: 150px;
}
textarea::placeholder { color: var(--text-muted); }

.send-btn {
  width: 38px; height: 38px; display: flex; align-items: center; justify-content: center;
  border: 0; border-radius: var(--radius-sm); background: var(--primary); color: #ffffff;
  box-shadow: none; cursor: pointer; transition: var(--transition-fast);
}
.send-btn:hover:not(:disabled) { background: var(--primary-hover); }
.send-btn.unavailable { opacity: 0.4; }

/* Markdown */
.markdown-body { color: inherit; }
.markdown-body :deep(pre) {
  background: var(--bg-elevated); border: 1px solid var(--border); border-radius: var(--radius-sm);
  padding: 12px; overflow-x: auto; font-family: var(--font-mono); font-size: 13px; line-height: 1.5;
}
.markdown-body :deep(code) { font-family: var(--font-mono); font-size: 13.5px; background: var(--bg-elevated); padding: 2px 6px; border-radius: 6px; }
.markdown-body :deep(pre code) { background: transparent; padding: 0; }
.markdown-body :deep(p) { margin: 0 0 8px; }
.markdown-body :deep(p:last-child) { margin-bottom: 0; }
.markdown-body :deep(ul), .markdown-body :deep(ol) { padding-left: 20px; }
.markdown-body :deep(blockquote) { border-left: 3px solid var(--primary-light); padding-left: 12px; color: var(--text-secondary); }
.markdown-body :deep(img) {
  max-width: 100%; height: auto; border-radius: var(--radius-sm);
  margin: 8px 0; display: block; border: 1px solid var(--border);
  box-shadow: var(--shadow-sm);
}
.markdown-body :deep(a) { color: var(--primary); text-decoration: none; }
.markdown-body :deep(a:hover) { text-decoration: underline; }
.markdown-body :deep(table) { border-collapse: collapse; width: 100%; margin: 8px 0; }
.markdown-body :deep(th), .markdown-body :deep(td) { border: 1px solid var(--border); padding: 6px 10px; font-size: 13px; }
.markdown-body :deep(th) { background: var(--bg-muted); font-weight: 600; }
.markdown-body :deep(hr) { border: none; border-top: 1px solid var(--border); margin: 12px 0; }

@media (max-width: 760px) {
  .chat-topbar { padding: 10px 14px; flex-direction: column; align-items: stretch; }
  .topbar-actions { justify-content: space-between; }
  .engine-pill { flex: 1; justify-content: center; }
  .messages-area { padding: 14px 10px; }
  .composer-area { padding: 10px 12px 14px; }
  .empty-state { min-height: 300px; }
  .empty-hints { flex-direction: column; width: 100%; align-items: stretch; }
  .hint-card { justify-content: center; }
  .message-row { gap: 8px; }
  .message-avatar { width: 28px; height: 28px; }
  .mm-label { display: none; }
}
</style>
