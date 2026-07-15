<template>
  <n-config-provider :theme="naiveTheme" :theme-overrides="themeOverrides">
    <n-loading-bar-provider>
      <n-dialog-provider>
        <n-notification-provider>
          <n-message-provider>
            <div class="app-wrapper" @dragenter="onDragEnter" @dragleave="onDragLeave" @dragover="onDragOver" @drop="onDrop">
              <SplashScreen />
              <ToastManager ref="toastRef" />
              <ConfirmDialog ref="confirmRef" />
              <RuntimeExceptionCenter />

              <!-- === Sidebar === -->
              <AppSidebar 
                :width="sidebarWidth" 
                :is-resizing="isResizing"
                @resize-start="onSidebarResizeStart" 
              />

              <!-- === Router View === -->
              <div class="router-wrapper">
                <RouteErrorView v-if="routeError" :message="routeError" />
                <router-view v-else v-slot="{ Component, route }">
                  <keep-alive include="ChatView">
                    <component :is="Component" :key="route.path" />
                  </keep-alive>
                </router-view>
              </div>

              <div
                v-if="dragOver"
                class="global-drag-overlay"
                @click="clearDragState"
                @dragenter.stop
                @dragover.prevent
                @dragleave="onDragLeave"
                @drop.prevent.stop="onDrop"
              >
                <div class="drag-overlay-panel">
                  <UploadCloud :size="36" />
                  <p>{{ appStore.t('drop_release') }}</p>
                </div>
              </div>
            </div>
          </n-message-provider>
        </n-notification-provider>
      </n-dialog-provider>
    </n-loading-bar-provider>
  </n-config-provider>
</template>

<script setup>
import { ref, computed, onErrorCaptured, onMounted, onUnmounted, provide } from 'vue'
import { darkTheme, lightTheme } from 'naive-ui'
import ToastManager from './components/ToastManager.vue'
import ConfirmDialog from './components/ConfirmDialog.vue'
import RuntimeExceptionCenter from './components/RuntimeExceptionCenter.vue'
import AppSidebar from './components/AppSidebar.vue'
import RouteErrorView from './components/RouteErrorView.vue'
import SplashScreen from './components/SplashScreen.vue'
import { UploadCloud } from 'lucide-vue-next'
import { useAppStore } from './stores/appStore.js'
import { useChatStore } from './stores/chatStore.js'
import { useApi } from './composables/useApi.js'
import { API_BASE, authFetch } from './composables/apiClient.js'
import { loadCheckpoints, trainAbortController } from './composables/useTraining.js'
import router from './router'

const appStore = useAppStore()
const chatStore = useChatStore()
const routeError = ref('')

// Naive UI 主题
const naiveTheme = computed(() => {
  return appStore.resolvedTheme === 'light' ? lightTheme : darkTheme
})

const themeOverrides = computed(() => {
  const accent = appStore.currentAccent || '#1a1a1a'
  return {
    common: {
      primaryColor: accent,
      primaryColorHover: accent + 'cc',
      primaryColorPressed: accent + 'aa',
      primaryColorSuppl: accent + '88',
      borderRadius: '12px',
      borderRadiusSmall: '8px',
      fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", sans-serif',
    },
    Button: {
      borderRadiusMedium: '12px',
      borderRadiusSmall: '8px',
    },
    Input: {
      borderRadius: '12px',
    },
    Card: {
      borderRadius: '20px',
    },
    Dialog: {
      borderRadius: '20px',
    },
    Notification: {
      borderRadius: '12px',
    },
  }
})

// Toast & Confirm
const toastRef = ref(null)
const confirmRef = ref(null)
const toast = (msg, type = 'info') => { if (toastRef.value) toastRef.value.showToast(msg, type) }
const $confirm = (options) => confirmRef.value ? confirmRef.value.show(options) : Promise.resolve(false)
provide('toast', toast)
provide('$confirm', $confirm)

// API connection
const { startHealthCheck, stopHealthCheck } = useApi()

// 侧边栏宽度调整
const sidebarWidth = ref(parseInt(localStorage.getItem('taiji_sidebar_width') || '260'))
const isResizing = ref(false)

function onSidebarResizeStart(event) {
  event.preventDefault()
  isResizing.value = true
  const startX = event.clientX
  const startWidth = sidebarWidth.value
  
  function onMouseMove(e) {
    const newWidth = Math.min(400, Math.max(200, startWidth + (e.clientX - startX)))
    sidebarWidth.value = newWidth
    document.documentElement.style.setProperty('--sidebar-width', newWidth + 'px')
  }
  
  function onMouseUp() {
    isResizing.value = false
    localStorage.setItem('taiji_sidebar_width', sidebarWidth.value.toString())
    document.removeEventListener('mousemove', onMouseMove)
    document.removeEventListener('mouseup', onMouseUp)
  }
  
  document.addEventListener('mousemove', onMouseMove)
  document.addEventListener('mouseup', onMouseUp)
}

// 恢复侧边栏宽度
onMounted(() => {
  const savedWidth = localStorage.getItem('taiji_sidebar_width')
  if (savedWidth) {
    document.documentElement.style.setProperty('--sidebar-width', savedWidth + 'px')
  }
})

function onRouteError(error) {
  routeError.value = error?.message || '页面加载失败'
}

router.onError((error) => {
  onRouteError(error)
})

router.beforeEach(() => {
  routeError.value = ''
  clearDragState()
})

router.afterEach(() => {
  routeError.value = ''
})

onErrorCaptured((error) => {
  onRouteError(error)
  return false
})

// Drag
const dragOver = ref(false)
let dragCounter = 0
let dragResetTimer = null
function isFileDrag(event) {
  return Array.from(event.dataTransfer?.types || []).includes('Files')
}
function clearDragState() {
  dragCounter = 0
  dragOver.value = false
  if (dragResetTimer) {
    clearTimeout(dragResetTimer)
    dragResetTimer = null
  }
}
function scheduleDragReset() {
  if (dragResetTimer) clearTimeout(dragResetTimer)
  dragResetTimer = setTimeout(clearDragState, 1200)
}
const onDragEnter = (event) => {
  if (!isFileDrag(event)) return
  event.preventDefault()
  dragCounter++
  dragOver.value = true
  scheduleDragReset()
}
const onDragOver = (event) => {
  if (!isFileDrag(event)) return
  event.preventDefault()
  dragOver.value = true
  scheduleDragReset()
}
const onDragLeave = (event) => {
  if (!isFileDrag(event)) return
  dragCounter--
  if (dragCounter <= 0) clearDragState()
}
const onDrop = (event) => {
  if (isFileDrag(event)) event.preventDefault()
  clearDragState()
}

// Lifecycle
onMounted(async () => {
  window.addEventListener('blur', clearDragState)
  window.addEventListener('keyup', onGlobalKeyup)
  try {
    const r = await authFetch(`${API_BASE}/api/settings`);
    if (r.ok) {
      const saved = await r.json();
      if (saved && typeof saved === 'object') {
        for (const [key, value] of Object.entries(saved)) {
          const storageKey = `taiji_${key}`;
          if (value !== undefined && value !== null) {
            localStorage.setItem(storageKey, typeof value === 'string' ? value : JSON.stringify(value));
          }
        }
        appStore.restoreUISettings(saved);
      }
    }
  } catch (e) { /* 静默处理 */ }

  await chatStore.loadSessions()
  if (chatStore.sessions.length === 0) {
    chatStore.createNewSession()
  }

  startHealthCheck()
  loadCheckpoints()
})

onUnmounted(() => {
  window.removeEventListener('blur', clearDragState)
  window.removeEventListener('keyup', onGlobalKeyup)
  clearDragState()
  stopHealthCheck()
  if (trainAbortController) trainAbortController.abort()
})

function onGlobalKeyup(event) {
  if (event.key === 'Escape') clearDragState()
}
</script>

<style>
@import './assets/styles/index.css';

.drag-overlay-panel {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  color: var(--text);
}

.drag-overlay-panel p {
  margin: 0;
  font-size: 18px;
  font-weight: 650;
  color: var(--text);
}

/* 深色主题覆盖 */
.theme-dark .drag-overlay-panel {
  color: #e0e0e0;
}

.theme-dark .drag-overlay-panel p {
  color: #e0e0e0;
}
</style>
