/**
 * 认证管理 composable
 * 提供登录、登出、Token 管理、自动附加认证头等功能
 *
 * 认证状态（authEnabled、username）由 runtimeStore 统一管理，
 * 此处只提供命令动作（login/logout/enable/disable）。
 */
import { ref, computed } from 'vue';
import { API_BASE, authFetch } from './apiClient.js';
import { useRuntimeStore } from '@/stores/runtimeStore.js';

const token = ref(localStorage.getItem('jwt_token') || '');

export function useAuth() {
  const runtimeStore = useRuntimeStore()

  // 从 runtimeStore 读取认证状态，不再自己轮询 /api/auth/status
  const authEnabled = computed(() => runtimeStore.auth?.enabled ?? false)
  const username = computed(() => runtimeStore.auth?.username ?? '')
  const authLoaded = computed(() => !!runtimeStore.runtimeSnapshot)
  const isAuthenticated = computed(() => !authEnabled.value || !!token.value);

  async function login(user, password) {
    const r = await fetch(`${API_BASE}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user, password })
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(d.detail || '登录失败');
    }
    const d = await r.json();
    token.value = d.token;
    localStorage.setItem('jwt_token', d.token);
    return d;
  }

  function logout() {
    token.value = '';
    localStorage.removeItem('jwt_token');
  }

  function getAuthHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (token.value) {
      headers['Authorization'] = `Bearer ${token.value}`;
    }
    return headers;
  }

  async function enableAuth(user, password) {
    const r = await authFetch(`${API_BASE}/api/auth/enable`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ username: user, password })
    });
    if (r.ok) {
      authEnabled.value = true;
      username.value = user;
    }
    return r.ok;
  }

  async function disableAuth() {
    const r = await authFetch(`${API_BASE}/api/auth/disable`, {
      method: 'POST',
      headers: getAuthHeaders()
    });
    if (r.ok) {
      authEnabled.value = false;
      logout();
    }
    return r.ok;
  }

  async function changePassword(oldPwd, newPwd) {
    const r = await authFetch(`${API_BASE}/api/auth/change_password`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({ old_password: oldPwd, new_password: newPwd })
    });
    return r.ok;
  }

  // 处理 401 响应（全局拦截）
  function handleAuthError(response) {
    if (response.status === 401) {
      logout();
      return true;
    }
    return false;
  }

  return {
    token,
    authEnabled,
    username,
    authLoaded,
    isAuthenticated,
    login,
    logout,
    getAuthHeaders,
    enableAuth,
    disableAuth,
    changePassword,
    handleAuthError,
  };
}
