
/**
 * API 连接与健康检查 composable
 *
 * 运行时状态统一通过 runtimeStore 管理（单一数据源）。
 * 本模块负责 HTTP 健康检查逻辑和连接生命周期。
 */
import { ref } from 'vue';
import { useAppStore } from '../stores/appStore.js';
import { useRuntimeStore } from '../stores/runtimeStore.js';
import { API_BASE, authFetch } from './apiClient.js';

/**
 * 动态 API 基地址：
 * - 开发环境：Vite proxy 拦截 /api，使用空字符串（相对路径）
 * - 生产环境：自动使用当前页面的 host
 */
export { API_BASE, authFetch };

// === 内部模块级状态 ===
const retryCountdown = ref(0);
const downloadProgress = ref(null);
let healthCheckTimer = null;
let retryTimer = null;
let consecutiveFailures = 0;
let lastHealthyAt = 0;
let isChatReceiving = false;

/**
 * 模块级函数：设置聊天接收状态
 */
export function setChatReceiving(receiving) {
  isChatReceiving = receiving;
}

export function useApi() {
  const appStore = useAppStore();
  const runtimeStore = useRuntimeStore();
  const t = (key, params = {}) => appStore.t(key, params);

  function syncAppState(state, msg = '') {
    // 统一写入 runtimeStore（单一数据源）
    runtimeStore.syncHealth(state, msg, runtimeStore.health.modelLoaded);
  }

  /**
   * Check bootstrap status (public, no auth required).
   * Returns bootstrap data or null on failure.
   */
  async function checkBootstrap() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      const resp = await fetch(`${API_BASE}/api/runtime/bootstrap`, { signal: controller.signal });
      clearTimeout(timeoutId);

      if (!resp.ok) return null;
      const ctype = resp.headers.get('content-type') || '';
      if (!ctype.includes('application/json')) return null;
      return await resp.json();
    } catch (e) { console.debug('[useApi] bootstrap check failed:', e.message) }
      return null;
  }

  async function checkHealth() {
    const state = runtimeStore.health.state;
    const modelLoaded = runtimeStore.health.modelLoaded;

    try {
      // Step 1: Use public bootstrap endpoint to determine auth state
      const bootstrap = await checkBootstrap();
      if (bootstrap) {
        runtimeStore.applyBootstrap(bootstrap);
        if (bootstrap.need_login && !localStorage.getItem('jwt_token')) {
          // Auth required but no token — show login, don't poll protected endpoint
          syncAppState('connecting', '需要登录');
          consecutiveFailures = 0;
          return false;
        }
      }

      // Step 2: Fetch full runtime status (requires auth when auth is enabled)
      const controller = new AbortController();
      const timeout = 5000;
      const timeoutId = setTimeout(() => controller.abort(), timeout);
      const resp = await authFetch(`${API_BASE}/api/runtime/status`, { signal: controller.signal });
      clearTimeout(timeoutId);

      if (!resp.ok) {
        if (resp.status === 401) {
          syncAppState('connecting', '需要重新登录');
          return false;
        }
        handleHealthFailure(`后端返回错误 (HTTP ${resp.status})`);
        return false;
      }

      const ctype = resp.headers.get('content-type') || '';
      if (!ctype.includes('application/json')) {
        handleHealthFailure('本地服务正在启动，暂时还没有返回运行状态。');
        return false;
      }

      const data = await resp.json();
      runtimeStore.applyRuntimeStatus(data);
      const health = data.health || {};
      consecutiveFailures = 0;
      if (health.state === 'connected') {
        lastHealthyAt = Date.now();
        syncAppState('connected', health.message || '');
        downloadProgress.value = null;
        retryCountdown.value = 0;
        clearRetryTimer();
        return true;
      } else if (health.state === 'downloading') {
        downloadProgress.value = health.download || health;
        syncAppState('downloading', health.message || '模型正在下载...');
        return false;
      } else if (health.state === 'loading') {
        consecutiveFailures = 0;
        if (state !== 'connected') {
          syncAppState('loading', health.message || '模型正在加载中...');
        }
        return false;
      } else {
        handleHealthFailure(health.message || '后端报告错误');
        return false;
      }
    } catch (err) {
      const recentlyHealthy = lastHealthyAt && Date.now() - lastHealthyAt < 45000;
      if (state === 'loading') {
        consecutiveFailures = 0;
        return false;
      }
      if (recentlyHealthy) {
        consecutiveFailures = 0;
        runtimeStore.syncHealth(
          state === 'error' ? 'connecting' : state,
          '本地运行时短暂无响应，正在保持连接并自动重试...',
          modelLoaded
        );
        return false;
      }
      if (state === 'unknown' || state === 'connecting') {
        consecutiveFailures++;
        if (consecutiveFailures >= 5) {
          syncAppState('error', '无法连接到后端服务，请确认后端已启动');
        } else {
          syncAppState('connecting', '正在连接后端服务...');
        }
        return false;
      }
      if (isChatReceiving) return false;
      consecutiveFailures++;
      if (consecutiveFailures >= 8) {
        syncAppState('error', '本地运行时暂时不可用，正在等待服务恢复...');
      }
      return false;
    }
  }

  function handleHealthFailure(message) {
    const state = runtimeStore.health.state;
    const modelLoaded = runtimeStore.health.modelLoaded;
    const recentlyHealthy = lastHealthyAt && Date.now() - lastHealthyAt < 45000;
    if (recentlyHealthy || consecutiveFailures < 7) {
      consecutiveFailures++;
      runtimeStore.syncHealth('connecting', message || '正在等待本地运行时恢复...', modelLoaded);
      return;
    }
    syncAppState('error', message || '本地运行时暂时不可用');
  }

  function startHealthCheck() {
    stopHealthCheck();
    consecutiveFailures = 0;
    checkHealth().catch(() => {});
    healthCheckTimer = setInterval(async () => {
      try {
        const connected = await checkHealth();
        if (connected) {
          clearInterval(healthCheckTimer);
          healthCheckTimer = setInterval(() => {
            checkHealth().catch(() => {});
            runtimeStore.refreshAll().catch(() => {});
          }, 15000);
        }
      } catch (e) {
        console.warn('[healthCheck] 健康检查异常:', e);
      }
    }, 2000);
  }

  function stopHealthCheck() {
    if (healthCheckTimer) {
      clearInterval(healthCheckTimer);
      healthCheckTimer = null;
    }
    clearRetryTimer();
  }

  function clearRetryTimer() {
    if (retryTimer) { clearInterval(retryTimer); retryTimer = null; }
  }

  return {
    API_BASE,
    retryCountdown, downloadProgress,
    t, checkHealth, startHealthCheck, stopHealthCheck,
    setChatReceiving,
  };
}
