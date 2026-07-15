<template>
  <section class="route-error-view">
    <div class="route-error-panel">
      <AlertTriangle :size="28" />
      <h2>页面加载失败</h2>
      <p>{{ message || '当前页面模块没有正确加载。请刷新客户端，或回到对话后重试。' }}</p>
      <div class="route-error-actions">
        <button class="primary-btn" @click="$router.push('/')">回到对话</button>
        <button class="btn-secondary" @click="$router.go(0)">刷新</button>
      </div>
    </div>
  </section>
</template>

<script setup>
import { AlertTriangle } from 'lucide-vue-next'

defineProps({
  message: { type: String, default: '' },
})
</script>

<style scoped>
.route-error-view {
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: var(--bg);
  color: var(--text);
}

.route-error-panel {
  width: min(460px, 100%);
  padding: 22px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--bg-card);
  box-shadow: var(--shadow-md);
  text-align: center;
}

.route-error-panel svg {
  color: var(--warning);
}

.route-error-panel h2 {
  margin: 12px 0 8px;
  font-size: 20px;
}

.route-error-panel p {
  margin: 0;
  color: var(--text-secondary);
  line-height: 1.6;
}

.route-error-actions {
  margin-top: 18px;
  display: flex;
  justify-content: center;
  gap: 10px;
}
</style>
