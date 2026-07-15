<template>
  <Transition name="splash-fade">
    <div v-if="visible" class="splash-wrapper">
      <div class="splash-bg"></div>
      <div class="splash-content">
        <div class="splash-logo">
          <img src="/logo.svg?v=ink-20260624-8" alt="态极" />
        </div>
        <h1 class="splash-title">态 极</h1>
        <p class="splash-status">{{ statusText }}</p>
      </div>
    </div>
  </Transition>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRuntimeStore } from '@/stores/runtimeStore.js'

const runtimeStore = useRuntimeStore()
const visible = ref(true)
const statusText = ref('正在连接...')
let checkInterval = null

onMounted(() => {
  checkInterval = setInterval(() => {
    const state = runtimeStore.health?.state
    if (state === 'connected') {
      statusText.value = '就绪'
      setTimeout(() => { visible.value = false }, 300)
      clearInterval(checkInterval)
    } else if (state === 'loading') {
      statusText.value = '模型加载中...'
    } else if (state === 'downloading') {
      statusText.value = '模型下载中...'
    } else {
      statusText.value = '正在连接...'
    }
  }, 500)

  setTimeout(() => {
    visible.value = false
    clearInterval(checkInterval)
  }, 8000)
})

onUnmounted(() => {
  if (checkInterval) clearInterval(checkInterval)
})
</script>

<style scoped>
.splash-wrapper {
  position: fixed;
  inset: 0;
  z-index: 99999;
  display: flex;
  align-items: center;
  justify-content: center;
}

.splash-bg {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(circle at 50% 32%, rgba(26,26,26,0.04), transparent 32%),
    var(--bg-base);
}

.splash-bg::before {
  display: none;
}

.splash-content {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 14px;
  animation: contentIn 0.6s var(--ease);
}

/* Logo */
.splash-logo {
  width: 82px;
  height: 82px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 28px;
  background: var(--bg-muted);
  border: 1px solid var(--border-subtle);
  box-shadow: var(--shadow-md);
}

.splash-logo img {
  width: 64px;
  height: 64px;
}

/* 标题 */
.splash-title {
  margin: 0;
  font-size: 22px;
  font-weight: 750;
  letter-spacing: 0.08em;
  color: var(--text);
}

/* 状态文字 */
.splash-status {
  margin: 0;
  min-height: 18px;
  padding: 0 9px;
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--bg-muted);
  font-size: 12px;
  line-height: 18px;
  font-weight: 600;
  color: var(--text-muted);
  letter-spacing: 0;
}

/* 淡出 — 与整体动画系统一致 */
.splash-fade-leave-active {
  transition: opacity 0.4s var(--ease);
}
.splash-fade-leave-to {
  opacity: 0;
}
</style>
