<template>
  <aside class="sidebar" :style="{ width: width + 'px' }">
    <div class="sidebar-resize-handle"
         :class="{ active: isResizing }"
         @mousedown="$emit('resize-start', $event)">
    </div>

    <!-- Logo + 品牌 -->
    <div class="sidebar-header">
      <div class="sidebar-logo">
        <div class="logo-icon-wrap">
          <img src="/logo-taiji-ink.jpg" class="logo-img" alt="Seed">
        </div>
        <div class="brand-copy">
          <h2>{{ t('title') }}</h2>
          <span>{{ runtimeStore.health.modelLoaded ? '在线' : '等待模型' }}</span>
        </div>
      </div>

      <!-- 搜索框 -->
      <div class="search-field">
        <Search :size="16" aria-hidden="true" />
        <input :placeholder="t('search') || '搜索...'" aria-label="搜索" v-model="searchQuery">
        <span class="kbd">⌘K</span>
      </div>
    </div>

    <!-- 新建对话 -->
    <button class="new-chat-btn" @click="handleNewChat" aria-label="新建对话">
      <Plus :size="15" />
      <span>{{ t('new_chat') }}</span>
    </button>

    <!-- 会话列表 -->
    <div class="session-list" role="list" aria-label="会话列表">
      <div class="nav-section-label">对话</div>
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

    <!-- 导航分组 -->
    <nav class="nav-scroll" aria-label="主导航">
      <div v-for="group in navGroups" :key="group.title">
        <div class="nav-section-label">{{ group.title }}</div>
        <RouterLink v-for="item in group.items" :key="item.path"
          class="nav-item" :class="{ active: isActiveRoute(item.path) }" :to="item.path">
          <span class="nav-icon-wrap">
            <component :is="item.icon" :size="16" aria-hidden="true" />
          </span>
          <span class="nav-label">{{ item.label }}</span>
        </RouterLink>
      </div>
    </nav>

    <!-- 生命状态指示器 -->
    <div class="side-life-pulse" v-if="runtimeStore.life.is_running" @click="router.push('/life')" title="查看生命状态">
      <span class="slp-dot" :class="dominantNeedClass"></span>
      <span class="slp-label">{{ dominantNeedLabel }}</span>
      <span class="slp-value">{{ dominantNeedValue }}%</span>
    </div>
  </aside>
</template>

<script setup>
import { computed, ref } from 'vue'
import { RouterLink, useRouter, useRoute } from 'vue-router'
import { Plus, MessageSquare, X, Search, BookOpen, Zap, Cpu, Layout, Settings, Heart } from 'lucide-vue-next'
import { useChatStore } from '@/stores/chatStore.js'
import { useAppStore } from '@/stores/appStore.js'
import { useRuntimeStore } from '@/stores/runtimeStore.js'

const props = defineProps({
  width: { type: Number, default: 248 },
  isResizing: { type: Boolean, default: false },
})

defineEmits(['resize-start'])

const chatStore = useChatStore()
const appStore = useAppStore()
const runtimeStore = useRuntimeStore()
const router = useRouter()
const route = useRoute()
const t = (key) => appStore.t(key)
const searchQuery = ref('')

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
/* 组件独有样式。通用 sidebar/nav-item/session-item 等由 app.css 统一管理 */
.session-name {
  display: flex; align-items: center; gap: 7px; min-width: 0; font-size: 0.85rem;
  color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.session-icon { flex-shrink: 0; color: var(--text-muted); }
.session-item.active .session-name { color: var(--text); }
.session-del-btn {
  width: 24px; height: 24px; display: flex; align-items: center; justify-content: center;
  border: 0; border-radius: 6px; background: transparent; color: var(--text-muted);
  cursor: pointer; opacity: 0; flex-shrink: 0;
  transition: opacity 0.15s ease, color 0.15s ease;
}
.session-item:hover .session-del-btn { opacity: 0.6; }
.session-del-btn:hover { opacity: 1 !important; color: var(--danger); background: var(--danger-light); }

.session-skeleton { padding: 0 4px; }
.skeleton-item { padding: 6px 9px; margin-bottom: 4px; }
.skeleton-bar {
  display: block; height: 14px; border-radius: 6px;
  background: linear-gradient(90deg, var(--bg-muted) 25%, var(--bg-hover) 50%, var(--bg-muted) 75%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
}
@keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }

.sidebar-resize-handle {
  position: absolute; top: 0; right: -3px; width: 6px; height: 100%;
  cursor: col-resize; z-index: 20; transition: background-color 0.2s;
}
.sidebar-resize-handle:hover { background: var(--primary-light); }
.sidebar-resize-handle.active { background: var(--primary); }

/* 生命状态指示器 */
.side-life-pulse {
  display: flex; align-items: center; gap: 6px;
  padding: 6px 10px; margin: 0 10px 8px;
  border-radius: 10px; cursor: pointer;
  background: var(--bg-muted); border: 1px solid var(--border);
  transition: border-color 0.2s ease;
}
.side-life-pulse:hover { border-color: var(--primary); }
.slp-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; background: var(--text-muted); }
.slp-dot.hunger    { background: var(--danger); animation: slp-pulse 2s infinite; }
.slp-dot.fatigue   { background: var(--text-muted); }
.slp-dot.boredom   { background: var(--text-muted); }
.slp-dot.stress    { background: var(--danger); animation: slp-pulse 1.5s infinite; }
.slp-dot.curiosity { background: var(--success); }
@keyframes slp-pulse { 0%,100%{opacity:1} 50%{opacity:.55} }
.slp-label { font-size: 0.72rem; color: var(--text-muted); min-width: 14px; }
.slp-value { font-size: 0.72rem; font-weight: 600; color: var(--text-secondary); }
@media (prefers-reduced-motion: reduce) { .slp-dot.hunger, .slp-dot.stress { animation: none; } }
</style>

<style>
/* 移动端响应式折叠 */
@media (max-width: 768px) {
  .sidebar { width: 56px !important; min-width: 56px !important; }
  .sidebar-header { padding: 14px 8px 8px !important; }
  .sidebar-logo { justify-content: center; }
  .brand-copy, .search-field, .nav-section-label,
  .session-name, .nav-label, .side-life-pulse { display: none !important; }
  .new-chat-btn { width: 36px; height: 32px; padding: 0 !important; margin: 10px auto !important; font-size: 0 !important; }
  .session-list { padding: 0 6px 8px !important; }
  .session-item { width: 36px; height: 32px; min-height: 32px; justify-content: center !important; padding: 0 !important; }
  .session-del-btn { display: none !important; }
  .nav-item { width: 36px; min-height: 32px; justify-content: center !important; padding: 0 !important; }
  .nav-icon-wrap { margin: 0 auto; }
}
</style>
