<template>
  <aside class="sidebar" :style="{ width: width + 'px' }">
    <div class="sidebar-resize-handle" 
         :class="{ active: isResizing }"
         @mousedown="$emit('resize-start', $event)">
    </div>
    <div class="sidebar-header">
      <div class="sidebar-logo">
        <div class="logo-icon-wrap">
          <TaijiLogo :size="38" />
        </div>
        <div class="brand-copy">
          <h2>{{ t('title') }}</h2>
          <span>{{ runtimeStore.health.modelLoaded ? '在线' : '等待模型' }}</span>
        </div>
      </div>
      <div class="runtime-card" :class="runtimeStore.connectionClass" role="status" :aria-label="`运行状态: ${runtimeStore.connectionStatus}`">
        <div class="runtime-state">
          <span class="runtime-dot" aria-hidden="true"></span>
          <span class="runtime-label">{{ runtimeStore.connectionStatus }}</span>
        </div>
        <MemoryStatusBar class="memory-badge" />
      </div>
    </div>

    <button class="new-chat-btn" @click="handleNewChat" aria-label="新建对话">
      <Plus :size="15" />
      <span>{{ t('new_chat') }}</span>
    </button>

    <div class="session-list" role="list" aria-label="会话列表">
      <div class="section-label">对话</div>
      <!-- 加载骨架 -->
      <div v-if="!chatStore.sessionsLoaded && !chatStore.sessions.length" class="session-skeleton">
        <div v-for="n in 3" :key="'skel-'+n" class="skeleton-item" aria-hidden="true">
          <span class="skeleton-bar" />
        </div>
      </div>
      <div v-for="session in chatStore.sessions" :key="session.id"
        v-memo="[session.id, session.name, chatStore.currentSessionId]"
        role="listitem"
        :class="['session-item', { active: chatStore.currentSessionId === session.id }]"
        @click="openSession(session.id)"
        tabindex="0"
        @keydown.enter="openSession(session.id)">
        <span class="session-name">
          <MessageSquare :size="14" class="session-icon" aria-hidden="true" />
          {{ session.name }}
        </span>
        <button class="session-del-btn" @click.stop="chatStore.deleteSession(session.id)"
          :aria-label="`删除会话 ${session.name}`">
          <X :size="13" />
        </button>
      </div>
    </div>

    <div class="sidebar-footer">
      <div v-for="group in navGroups" :key="group.title" class="nav-group">
        <div class="section-label">{{ group.title }}</div>
        <RouterLink v-for="item in group.items" :key="item.path"
          class="settings-btn" :class="{ active: isActiveRoute(item.path) }" :to="item.path">
          <TaijiLogo v-if="item.icon === 'TaijiLogo'" :size="15" class="nav-icon" />
          <component v-else :is="item.icon" :size="15" class="nav-icon" aria-hidden="true" />
          <span class="nav-label">{{ item.label }}</span>
        </RouterLink>
      </div>
    </div>

    <div class="side-life-pulse" v-if="runtimeStore.life.is_running" @click="router.push('/life')" title="查看生命状态">
      <span class="slp-dot" :class="dominantNeedClass"></span>
      <span class="slp-label">{{ dominantNeedLabel }}</span>
      <span class="slp-value">{{ dominantNeedValue }}%</span>
    </div>
  </aside>
</template>

<script setup>
import { computed } from 'vue'
import { RouterLink, useRouter, useRoute } from 'vue-router'
import { Plus, MessageSquare, X, BookOpen, Zap, Cpu, Layout, Settings, Heart } from 'lucide-vue-next'
import { useChatStore } from '@/stores/chatStore.js'
import { useAppStore } from '@/stores/appStore.js'
import { useRuntimeStore } from '@/stores/runtimeStore.js'
import MemoryStatusBar from './MemoryStatusBar.vue'
import TaijiLogo from './TaijiLogo.vue'

const props = defineProps({
  width: { type: Number, default: 260 },
  isResizing: { type: Boolean, default: false },
})

defineEmits(['resize-start'])

const chatStore = useChatStore()
const appStore = useAppStore()
const runtimeStore = useRuntimeStore()
const router = useRouter()
const route = useRoute()
const t = (key) => appStore.t(key)

function isActiveRoute(path) { return route.path === path }
function handleNewChat() { chatStore.createNewSession(); router.push('/').catch(() => {}) }
function openSession(id) { chatStore.switchSession(id); router.push('/').catch(() => {}) }

const needIcons = { hunger: '饿', fatigue: '累', boredom: '闷', stress: '压', curiosity: '奇' }
const dominantNeedKey = computed(() => runtimeStore.life.dominant_need || '')
const dominantNeedLabel = computed(() => needIcons[dominantNeedKey.value] || '')
const dominantNeedValue = computed(() => {
  const needs = runtimeStore.life.needs || {}
  return dominantNeedKey.value ? Math.round(needs[dominantNeedKey.value] || 0) : 0
})
const dominantNeedClass = computed(() => dominantNeedKey.value)

const navGroups = computed(() => [
  { title: '工作台', items: [{ path: '/workspace', icon: Layout, label: 'IDE' }] },
  { title: '能力', items: [
    { path: '/agent', icon: Cpu, label: t('agent_config') },
    { path: '/kb', icon: BookOpen, label: t('kb_management') },
    { path: '/train', icon: Zap, label: t('fine_tuning') },
  ]},
  { title: '系统', items: [
    { path: '/life', icon: Heart, label: '生命状态' },
    { path: '/settings', icon: Settings, label: t('sys_settings') },
  ]},
])
</script>

<style scoped>
/* AppSidebar — 组件独有样式。通用 sidebar/session-item/settings-btn 等由 app.css 统一管理 */
.logo-icon-wrap {
  width: 44px; height: 44px;
  display: flex; align-items: center; justify-content: center;
  border-radius: 14px; flex-shrink: 0;
  transition: background-color 0.2s var(--ease), border-color 0.2s var(--ease);
}
.logo-icon-wrap:hover { background: var(--bg-hover); }
.brand-copy { min-width: 0; flex: 1; display: flex; flex-direction: column; gap: 2px; }
.brand-copy h2 {
  margin: 0; font-size: 15px; font-weight: 700; color: var(--text); line-height: 1.2;
  font-family: var(--font-display); letter-spacing: 0.02em;
}
.brand-copy span { color: var(--text-muted); font-size: 11px; line-height: 1.35; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

.runtime-card {
  display: flex; align-items: center; justify-content: space-between;
  gap: 8px; margin-top: 12px; padding: 9px 10px;
  border-radius: var(--radius-lg); background: var(--bg-muted); border: 1px solid var(--border-subtle);
}
.runtime-state { display: flex; align-items: center; gap: 7px; min-width: 0; }
.runtime-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--text-muted); flex-shrink: 0; }
.runtime-card.connected .runtime-dot { background: var(--success); box-shadow: 0 0 0 3px var(--success-light); }
.runtime-card.connected .runtime-dot { animation: taijiBreathe 4s ease-in-out infinite; }
.runtime-card.loading .runtime-dot, .runtime-card.connecting .runtime-dot { background: var(--warning); animation: pulse 1.5s infinite; }
.runtime-card.error .runtime-dot { background: var(--danger); }
.runtime-label { color: var(--text-secondary); font-size: 12px; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.memory-badge { flex-shrink: 0; }

.session-list { flex: 1; min-height: 0; overflow-y: auto; padding: 0 8px 10px; }
.section-label {
  padding: 9px 8px 5px; font-size: 10.5px; color: var(--text-muted);
  font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
}
.session-name {
  display: flex; align-items: center; gap: 7px; min-width: 0; font-size: 13px;
  color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.session-icon { flex-shrink: 0; color: var(--text-muted); }
.session-del-btn {
  width: 24px; height: 24px; display: flex; align-items: center; justify-content: center;
  border: 0; border-radius: var(--radius-sm); background: transparent; color: var(--text-muted);
  cursor: pointer; opacity: 0; flex-shrink: 0;
  transition: opacity 0.15s var(--ease), color 0.15s var(--ease);
}
.session-del-btn:hover { opacity: 1 !important; color: var(--danger); background: var(--danger-light); }

.nav-icon { flex-shrink: 0; }
.nav-label { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.session-skeleton { padding: 0 4px; }
.skeleton-item { padding: 6px 9px; margin-bottom: 4px; }
.skeleton-bar {
  display: block; height: 14px; border-radius: var(--radius-sm);
  background: linear-gradient(90deg, var(--bg-muted) 25%, var(--bg-hover) 50%, var(--bg-muted) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
}
@keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
</style>

<style>
/* 移动端响应式折叠（app.css 中无此规则，保留为 unscoped） */
@media (max-width: 768px) {
  .sidebar { width: 56px !important; min-width: 56px !important; }
  .sidebar-header { padding: 14px 8px 8px !important; }
  .sidebar-logo { justify-content: center; }
  .brand-copy, .runtime-label, .memory-badge, .section-label,
  .session-name, .nav-label, .runtime-card { display: none !important; }
  .new-chat-btn { width: 36px; height: 32px; padding: 0 !important; margin: 10px auto !important; font-size: 0 !important; }
  .session-list { padding: 0 6px 8px !important; }
  .session-item { width: 36px; height: 32px; min-height: 32px; justify-content: center !important; padding: 0 !important; }
  .session-del-btn { display: none !important; }
  .sidebar-footer { padding: 8px 6px 10px !important; }
  .settings-btn { width: 36px; min-height: 32px; justify-content: center !important; padding: 0 !important; }
}

/* Sidebar mini life pulse */
.side-life-pulse {
  display: flex; align-items: center; gap: 6px;
  padding: 6px 10px; margin: 0 6px 4px;
  border-radius: var(--radius-md); cursor: pointer;
  background: var(--bg-muted); border: 1px solid var(--border);
  transition: border-color 0.2s var(--ease);
}
.side-life-pulse:hover { border-color: var(--ink-muted); }
.slp-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; background: var(--ink-muted); }
.slp-dot.hunger    { background: var(--danger); animation: slp-pulse 2s infinite; }
.slp-dot.fatigue   { background: var(--ink-muted); }
.slp-dot.boredom   { background: var(--text-muted); }
.slp-dot.stress    { background: var(--danger); animation: slp-pulse 1.5s infinite; }
.slp-dot.curiosity { background: var(--success); }
@keyframes slp-pulse { 0%,100%{opacity:1} 50%{opacity:.55} }
.slp-label { font-size: 11px; color: var(--text-muted); min-width: 14px; }
.slp-value { font-size: 11px; font-weight: 600; color: var(--text-secondary); }
@media (prefers-reduced-motion: reduce) { .slp-dot.hunger, .slp-dot.stress { animation: none; } }
</style>
