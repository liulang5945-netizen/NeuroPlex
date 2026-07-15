<template>
  <div class="terminal-wrapper">
    <div class="terminal-toolbar">
      <span class="terminal-title"><TerminalIcon :size="14" /> Terminal</span>
      <div class="terminal-actions">
        <button class="btn-term" @click="reconnect" title="重连"><RefreshCw :size="14" /></button>
        <button class="btn-term" @click="clearTerminal" title="清屏"><Trash2 :size="14" /></button>
      </div>
    </div>
    <div ref="terminalContainer" class="terminal-container"></div>
    <div class="terminal-status" :class="connectionStatus">
      {{ statusText }}
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, computed, watch } from 'vue';
import { Terminal as TerminalIcon, RefreshCw, Trash2 } from 'lucide-vue-next';
import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import { WebLinksAddon } from '@xterm/addon-web-links';
import { useAppStore } from '../stores/appStore.js';
import { useRuntimeStore } from '../stores/runtimeStore.js';
import { API_BASE } from '../composables/apiClient.js';
import '@xterm/xterm/css/xterm.css';

const appStore = useAppStore();
const runtimeStore = useRuntimeStore();
const terminalContainer = ref(null);
const connectionStatus = ref('disconnected');
const connectionError = ref('');

const statusText = computed(() => {
  const map = { connected: '已连接', connecting: '连接中...', disconnected: '未连接' };
  const base = map[connectionStatus.value] || '';
  return connectionError.value ? `${base} - ${connectionError.value}` : base;
});

const darkTheme = {
  background: '#1e293b', foreground: '#e2e8f0', cursor: '#5b7a8a',
  selectionBackground: 'rgba(91,122,138,0.3)',
  black: '#1e293b', red: '#c75a5a', green: '#5a9e6f', yellow: '#c48a3f',
  blue: '#5b7a8a', magenta: '#8a7ab5', cyan: '#5a9eaa', white: '#e2e8f0',
};

const lightTheme = {
  background: '#f8fafc', foreground: '#2c3e50', cursor: '#5b7a8a',
  selectionBackground: 'rgba(91,122,138,0.2)',
  black: '#2c3e50', red: '#c75a5a', green: '#5a9e6f', yellow: '#c48a3f',
  blue: '#5b7a8a', magenta: '#8a7ab5', cyan: '#5a9eaa', white: '#f8fafc',
};

let term = null;
let fitAddon = null;
let ws = null;
let reconnectTimer = null;
let resizeObserver = null;
let reconnectAttempts = 0;
const MAX_RECONNECT = 8;
const BASE_DELAY = 2000;

function getWsUrl() {
  const token = localStorage.getItem('jwt_token') || '';
  let wsBase = '';

  if (API_BASE && API_BASE.length > 0) {
    // 生产环境：API_BASE 类似 http://192.168.1.100:8000
    try {
      const apiUrl = new URL(API_BASE);
      const protocol = apiUrl.protocol === 'https:' ? 'wss:' : 'ws:';
      wsBase = `${protocol}//${apiUrl.host}`;
    } catch (e) {
      const loc = window.location;
      const protocol = loc.protocol === 'https:' ? 'wss:' : 'ws:';
      wsBase = `${protocol}//${loc.hostname}:8000`;
    }
  } else {
    // 开发环境：直连后端 8000 端口
    wsBase = 'ws://127.0.0.1:8000';
  }

  const params = new URLSearchParams();
  if (token) params.set('token', token);
  const qs = params.toString();
  return `${wsBase}/ws/terminal${qs ? '?' + qs : ''}`;
}

function isDarkMode() {
  return document.documentElement.classList.contains('theme-dark') ||
    (!document.documentElement.classList.contains('theme-light') &&
     window.matchMedia('(prefers-color-scheme: dark)').matches)
}

function initTerminal() {
  if (!terminalContainer.value) return;

  term = new Terminal({
    cursorBlink: true,
    cursorStyle: 'bar',
    fontSize: 14,
    fontFamily: "'Cascadia Code', 'Fira Code', 'JetBrains Mono', Consolas, monospace",
    theme: isDarkMode() ? darkTheme : lightTheme,
    allowProposedApi: true,
    scrollback: 5000,
  });

  fitAddon = new FitAddon();
  term.loadAddon(fitAddon);
  term.loadAddon(new WebLinksAddon());

  term.open(terminalContainer.value);
  fitAddon.fit();

  // 输入转发到 WebSocket
  term.onData((data) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'input', data }));
    }
  });

  // 窗口大小变化时通知服务端
  resizeObserver = new ResizeObserver(() => {
    if (fitAddon) fitAddon.fit();
    if (ws && ws.readyState === WebSocket.OPEN && term) {
      ws.send(JSON.stringify({ type: 'resize', cols: term.cols, rows: term.rows }));
    }
  });
  resizeObserver.observe(terminalContainer.value);
}

function connectWs() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  if (runtimeStore.auth.enabled && !localStorage.getItem('jwt_token')) {
    connectionStatus.value = 'disconnected';
    connectionError.value = '需要登录后连接终端';
    runtimeStore.syncTerminal('disconnected', connectionError.value);
    term?.writeln('\r\n\x1b[33m终端需要登录认证，请在系统设置完成登录后点击重连\x1b[0m');
    return;
  }

  connectionStatus.value = 'connecting';
  connectionError.value = '';
  runtimeStore.syncTerminal('connecting');
  const url = getWsUrl();

  try {
    ws = new WebSocket(url);
  } catch (e) {
    connectionStatus.value = 'disconnected';
    connectionError.value = `创建失败: ${e.message}`;
    runtimeStore.syncTerminal('disconnected', connectionError.value);
    term?.writeln(`\r\n\x1b[31m无法创建 WebSocket 连接: ${e.message}\x1b[0m`);
    return;
  }

  ws.onopen = () => {
    connectionStatus.value = 'connected';
    connectionError.value = '';
    runtimeStore.syncTerminal('connected');
    reconnectAttempts = 0;
    if (term) {
      ws.send(JSON.stringify({ type: 'resize', cols: term.cols, rows: term.rows }));
    }
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === 'output' && term) {
        term.write(msg.data);
      } else if (msg.type === 'exit') {
        term?.writeln(`\r\n\x1b[33m进程已退出 (code: ${msg.code})\x1b[0m`);
        connectionStatus.value = 'disconnected';
      } else if (msg.type === 'error') {
        connectionError.value = normalizeTerminalError(msg.data);
        runtimeStore.syncTerminal('disconnected', connectionError.value);
        term?.writeln(`\r\n\x1b[31m${connectionError.value}\x1b[0m`);
      }
    } catch (e) {
      term?.write(event.data);
    }
  };

  ws.onerror = (e) => {
    connectionStatus.value = 'disconnected';
    connectionError.value = '连接异常，请检查后端服务是否运行';
    runtimeStore.syncTerminal('disconnected', connectionError.value);
  };

  ws.onclose = (event) => {
    connectionStatus.value = 'disconnected';
    clearTimeout(reconnectTimer);

    if (event.code === 4001) {
      connectionError.value = 'JWT token 缺失或已过期，请重新登录';
      runtimeStore.reportAuthExpired(connectionError.value);
      runtimeStore.syncTerminal('disconnected', connectionError.value);
      term?.writeln('\r\n\x1b[31m认证已过期：已清除失效 token，请重新登录后点击重连\x1b[0m');
      return;
    } else if (event.code === 4002) {
      connectionError.value = '终端数已达上限';
      runtimeStore.syncTerminal('disconnected', connectionError.value);
      term?.writeln('\r\n\x1b[31m终端并发数已达上限，请关闭其他终端后点击重连\x1b[0m');
      return;
    } else if (event.code === 1006) {
      connectionError.value = '服务未启动或网络不可达';
      runtimeStore.syncTerminal('disconnected', connectionError.value);
      term?.writeln('\r\n\x1b[31m终端通道暂不可用，正在等待本地服务恢复\x1b[0m');
    }

    if (reconnectAttempts >= MAX_RECONNECT) {
      term?.writeln('\r\n\x1b[31m连接失败次数过多，请检查后端服务是否运行，然后点击重连按钮\x1b[0m');
      connectionError.value = '重连次数用尽，请手动重连';
      runtimeStore.syncTerminal('disconnected', connectionError.value);
      return;
    }
    const delay = Math.min(BASE_DELAY * Math.pow(2, reconnectAttempts), 30000);
    reconnectAttempts++;
    connectionError.value = `第 ${reconnectAttempts}/${MAX_RECONNECT} 次重连，${Math.round(delay / 1000)}秒后...`;
    runtimeStore.syncTerminal('connecting', connectionError.value);
    reconnectTimer = setTimeout(() => {
      if (term) connectWs();
    }, delay);
  };
}

function normalizeTerminalError(message) {
  const text = String(message || '')
  if (/认证|token|JWT|Unauthorized/i.test(text)) {
    return 'JWT token 缺失或已过期，请重新登录'
  }
  return text || '终端连接异常'
}

async function reconnect() {
  if (ws) {
    ws.close();
    ws = null;
  }
  clearTimeout(reconnectTimer);
  reconnectAttempts = 0;
  if (term) term.clear();
  await runtimeStore.refreshAuth().catch(() => {});
  connectWs();
}

function clearTerminal() {
  if (term) term.clear();
}

watch(() => appStore.currentTheme, () => {
  if (term) term.options.theme = isDarkMode() ? darkTheme : lightTheme
})

onMounted(async () => {
  initTerminal();
  await runtimeStore.refreshAuth().catch(() => {});
  connectWs();
});

onBeforeUnmount(() => {
  clearTimeout(reconnectTimer);
  if (resizeObserver) { resizeObserver.disconnect(); resizeObserver = null; }
  if (ws) { ws.close(); ws = null; }
  if (term) { term.dispose(); term = null; }
});
</script>

<style scoped>
.terminal-wrapper { display: flex; flex-direction: column; height: 100%; min-height: 0; background: var(--bg-card); }
.terminal-toolbar { display: flex; align-items: center; justify-content: space-between; padding: 2px 10px; background: var(--bg); border-bottom: 1px solid var(--border); min-height: 24px; }
.terminal-title { font-size: 12px; color: var(--text-muted); font-weight: 500; display: flex; align-items: center; gap: 4px; }
.terminal-actions { display: flex; gap: 2px; }
.btn-term { background: transparent; border: 1px solid transparent; cursor: pointer; font-size: 13px; padding: 3px 6px; border-radius: 4px; transition: all 0.15s ease; }
.btn-term:hover { background: var(--bg-hover); border-color: var(--border); }
.terminal-container { flex: 1 1 0; min-height: 0; padding: 4px; overflow: hidden; }
.terminal-container :deep(.xterm) { height: 100%; }
.terminal-status { padding: 1px 10px; font-size: 11px; min-height: 18px; display: flex; align-items: center; gap: 4px; }
.terminal-status::before { content: ''; display: inline-block; width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.terminal-status.connected { color: var(--success); background: var(--success-light); }
.terminal-status.connected::before { background: var(--success); }
.terminal-status.connecting { color: var(--warning); background: var(--warning-light); }
.terminal-status.connecting::before { background: var(--warning); animation: termPulse 1.5s infinite; }
.terminal-status.disconnected { color: var(--danger); background: var(--danger-light); }
.terminal-status.disconnected::before { background: var(--danger); }
@keyframes termPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
</style>
