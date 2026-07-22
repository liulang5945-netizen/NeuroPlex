<template>
  <div class="agent-page">
    <div class="page-container">

      <!-- 1. 基本信息卡片 -->
      <section class="cfg-card">
        <div class="cfg-card-head">
          <div class="cfg-title">
            <h2>基本信息</h2>
            <span class="sub">定义智能体身份与归属</span>
          </div>
          <button class="btn btn-ghost sm" @click="refreshAgentRuntime" :disabled="runtimeStore.toolsLoading">
            <RefreshCw :size="14" :class="{ spin: runtimeStore.toolsLoading }" /> 刷新
          </button>
        </div>
        <div class="form-grid">
          <div class="form-field">
            <label class="form-label" for="agent-name">智能体名称<span class="req">*</span></label>
            <input class="input" id="agent-name" type="text" value="语言推理专家" readonly>
          </div>
          <div class="form-field">
            <label class="form-label" for="agent-domain">所属域<span class="req">*</span></label>
            <select class="select" id="agent-domain" disabled>
              <option value="language">语言</option>
              <option value="reasoning" selected>推理</option>
              <option value="code">代码</option>
              <option value="knowledge">知识</option>
              <option value="multi">多域</option>
            </select>
          </div>
          <div class="form-field full">
            <label class="form-label" for="agent-desc">描述</label>
            <textarea class="textarea" id="agent-desc" rows="3" readonly>融合语言域和推理域神经元，处理复杂问答</textarea>
          </div>
        </div>
      </section>

      <!-- 2. 参数配置 -->
      <section class="cfg-card">
        <div class="cfg-card-head">
          <div class="cfg-title">
            <h2>参数配置</h2>
            <span class="sub">系统提示策略 · 温度 · 迭代上限</span>
          </div>
          <button class="btn btn-primary sm" @click="saveAgentPrefs" :disabled="runtimeStore.toolsLoading">
            <svg class="ic-svg" viewBox="0 0 24 24"><path d="M5 12l4 4L19 7"/></svg>
            保存配置
          </button>
        </div>

        <div class="params-grid">
          <div class="form-field">
            <label class="form-label" for="ag-temp">
              <span>温度</span><span class="hint">0 - 2</span>
            </label>
            <input id="ag-temp" class="input" v-model.number="temperature" type="number" min="0" max="2" step="0.1" @change="saveAgentPrefs" />
          </div>
          <div class="form-field">
            <label class="form-label" for="ag-iter">
              <span>最大迭代</span><span class="hint">1 - 50</span>
            </label>
            <input id="ag-iter" class="input" v-model.number="maxIterations" type="number" min="1" max="50" @change="saveAgentPrefs" />
          </div>
        </div>

        <!-- 预设方案 -->
        <div class="preset-row">
          <span class="preset-label">预设方案</span>
          <div class="preset-chips">
            <button class="preset-chip" @click="temperature = 0.2; maxIterations = 5; saveAgentPrefs()">
              <span class="chip-name">精准</span>
              <span class="chip-meta">T 0.2 · 5 轮</span>
            </button>
            <button class="preset-chip" @click="temperature = 0.7; maxIterations = 10; saveAgentPrefs()">
              <span class="chip-name">均衡</span>
              <span class="chip-meta">T 0.7 · 10 轮</span>
            </button>
            <button class="preset-chip" @click="temperature = 1.2; maxIterations = 20; saveAgentPrefs()">
              <span class="chip-name">创意</span>
              <span class="chip-meta">T 1.2 · 20 轮</span>
            </button>
          </div>
        </div>

        <div class="action-row">
          <button class="btn btn-outline" @click="maxIterations = 10; temperature = 0.7; saveAgentPrefs()">重置默认</button>
        </div>
      </section>

      <!-- 3. 状态概览 -->
      <section class="overview-grid">
        <div class="ov-card" :class="runtimeStore.connectionClass">
          <span class="ov-ic">
            <svg class="ic-svg" viewBox="0 0 24 24"><path d="M5 12.55a11 11 0 0 1 14.08 0M1.42 9a16 16 0 0 1 21.16 0M8.53 16.11a6 6 0 0 1 6.95 0M12 20h.01"/></svg>
          </span>
          <div class="ov-text">
            <small class="ov-label">连接状态</small>
            <strong class="ov-value">
              <span class="ov-dot"></span>
              {{ runtimeStore.connectionStatus }}
            </strong>
          </div>
        </div>
        <div class="ov-card">
          <span class="ov-ic">
            <svg class="ic-svg" viewBox="0 0 24 24"><path d="M14.7 6.3a4 4 0 0 0-5.6 5.6L3 18v3h3l6.1-6.1a4 4 0 0 0 5.6-5.6l-2.1 2.1-2-2 2.1-2.1Z"/></svg>
          </span>
          <div class="ov-text">
            <small class="ov-label">工具数量</small>
            <strong class="ov-value">{{ runtimeStore.tools.length }}<span class="ov-unit"> 个可用</span></strong>
          </div>
        </div>
        <div class="ov-card">
          <span class="ov-ic">
            <svg class="ic-svg" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>
          </span>
          <div class="ov-text">
            <small class="ov-label">MCP 数量</small>
            <strong class="ov-value">{{ installedServers.filter(s => s.running).length }}<span class="ov-unit"> 个运行</span></strong>
          </div>
        </div>
        <div class="ov-card" v-if="runtimeStore.memoryAvailableGb">
          <span class="ov-ic">
            <svg class="ic-svg" viewBox="0 0 24 24"><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10v4M11 10v4M15 10v4"/></svg>
          </span>
          <div class="ov-text">
            <small class="ov-label">可用内存</small>
            <strong class="ov-value">{{ runtimeStore.memoryAvailableGb.toFixed(1) }}<span class="ov-unit"> GB</span></strong>
          </div>
        </div>
      </section>

      <!-- 4. 标签页 -->
      <div class="tabs" role="tablist">
        <button class="tab" :class="{ active: activeTab === 'tools' }" @click="activeTab = 'tools'">工具与插件</button>
        <button class="tab" :class="{ active: activeTab === 'installed' }" @click="activeTab = 'installed'; loadInstalled()">MCP 服务</button>
        <button class="tab" :class="{ active: activeTab === 'marketplace' }" @click="activeTab = 'marketplace'; loadMarketplace()">MCP 市场</button>
      </div>

      <!-- 工具与插件 tab -->
      <section v-if="activeTab === 'tools'" class="tab-panel">
        <div class="filter-row">
          <div class="search-field">
            <Search :size="15" />
            <input v-model="toolQuery" placeholder="搜索工具名或描述..." />
          </div>
          <select class="select" v-model="toolSource">
            <option value="">全部来源</option>
            <option v-for="s in toolSources" :key="s" :value="s">{{ s }}</option>
          </select>
        </div>
        <div class="tool-grid">
          <div v-for="tool in filteredTools" :key="tool.name" class="tool-card" :class="{ off: tool.enabled === false }">
            <div class="tool-head">
              <span class="tool-ic">
                <svg class="ic-svg" viewBox="0 0 24 24"><path d="M14.7 6.3a4 4 0 0 0-5.6 5.6L3 18v3h3l6.1-6.1a4 4 0 0 0 5.6-5.6l-2.1 2.1-2-2 2.1-2.1Z"/></svg>
              </span>
              <button class="toggle" :class="{ on: tool.enabled !== false }" :aria-pressed="tool.enabled !== false" aria-label="启用工具"></button>
            </div>
            <div class="tool-name">{{ tool.name }}</div>
            <div class="tool-desc">{{ tool.description || '暂无描述' }}</div>
          </div>
        </div>
        <div class="panel-foot">
          <span class="foot-hint">共 {{ filteredTools.length }} 个工具</span>
        </div>
        <div v-if="!filteredTools.length && !runtimeStore.toolsLoading" class="empty-msg">无匹配工具</div>
      </section>

      <!-- MCP 服务 tab -->
      <section v-if="activeTab === 'installed'" class="tab-panel">
        <div v-if="installedServers.length" class="mcp-table-wrap">
          <table class="mcp-table">
            <thead>
              <tr>
                <th>服务名称</th>
                <th>协议</th>
                <th>状态</th>
                <th>端点</th>
                <th style="width:110px">操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="server in installedServers" :key="server.id">
                <td>
                  <span class="mcp-name">
                    <span class="mcp-ic"><svg class="ic-svg" viewBox="0 0 24 24"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/></svg></span>
                    {{ server.name || server.id }}
                  </span>
                </td>
                <td><span class="mcp-protocol">{{ server.npm_package ? 'npm' : 'stdio' }}</span></td>
                <td><span class="status-chip" :class="server.running ? 'status-running' : 'status-stopped'">{{ server.running ? '运行中' : '已停止' }}</span></td>
                <td><span class="mcp-endpoint">{{ server.npm_package || server.id }}</span></td>
                <td>
                  <div class="row-actions">
                    <button v-if="!server.running" class="act-btn" @click="startServer(server.id)" aria-label="启动"><svg class="ic-svg" viewBox="0 0 24 24"><path d="M7 5l12 7-12 7Z"/></svg></button>
                    <button v-else class="act-btn" @click="stopServer(server.id)" aria-label="停止"><svg class="ic-svg" viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></svg></button>
                    <button v-if="server.running" class="act-btn" @click="restartServer(server.id)" aria-label="重启"><svg class="ic-svg" viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 3-6.7L3 8M3 3v5h5"/></svg></button>
                    <button class="act-btn" @click="uninstallServer(server.id)" aria-label="卸载"><svg class="ic-svg" viewBox="0 0 24 24"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg></button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="panel-foot" v-if="installedServers.length">
          <span class="foot-hint">{{ installedServers.filter(s => s.running).length }} 运行中 · {{ installedServers.filter(s => !s.running).length }} 已停止 · 共 {{ installedServers.length }} 个服务</span>
        </div>
        <div v-if="!installedServers.length" class="empty-msg">暂无已安装 MCP 服务</div>
      </section>

      <!-- MCP 市场 tab -->
      <section v-if="activeTab === 'marketplace'" class="tab-panel">
        <div class="filter-row">
          <div class="search-field">
            <Search :size="15" />
            <input v-model="mcpSearch" placeholder="搜索 MCP 服务..." @input="debounceSearch" />
          </div>
          <select class="select" v-model="mcpCategory" @change="loadMarketplace">
            <option value="">全部分类</option>
            <option v-for="c in categories" :key="c" :value="c">{{ c }}</option>
          </select>
          <button class="btn btn-outline sm" @click="loadMarketplace" :disabled="mcpLoading">
            <RefreshCw :size="13" :class="{ spin: mcpLoading }" /> 同步
          </button>
        </div>
        <div class="tool-grid">
          <div v-for="server in marketplaceServers" :key="server.id" class="tool-card" :class="{ off: !server.installed && !server.running }">
            <div class="tool-head">
              <span class="tool-ic">
                <svg class="ic-svg" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/></svg>
              </span>
              <span class="status-chip" :class="server.running ? 'status-running' : 'status-stopped'">
                {{ server.running ? '运行中' : server.installed ? '已安装' : '未安装' }}
              </span>
            </div>
            <div class="tool-name">{{ server.name }}</div>
            <div class="tool-desc">{{ server.description || '暂无描述' }}</div>
            <div class="mc-actions">
              <button v-if="!server.installed" class="mc-btn" @click="installServer(server.id)"><Download :size="12" /> 安装</button>
              <template v-else>
                <button v-if="!server.running" class="mc-btn" @click="startServer(server.id)"><Play :size="12" /> 启动</button>
                <button v-else class="mc-btn" @click="stopServer(server.id)"><Square :size="12" /> 停止</button>
                <button class="mc-btn" @click="uninstallServer(server.id)"><Trash2 :size="12" /> 卸载</button>
              </template>
            </div>
          </div>
        </div>
        <div class="panel-foot">
          <span class="foot-hint">共 {{ marketplaceServers.length }} 个服务</span>
        </div>
        <div v-if="!marketplaceServers.length && !mcpLoading" class="empty-msg">{{ mcpLoading ? '加载中...' : '无匹配结果' }}</div>
      </section>

    </div>
  </div>
</template>

<script setup>
import { computed, inject, onMounted, ref, watch } from 'vue'
import { RefreshCw, Play, Square, Trash2, Download, Search } from 'lucide-vue-next'
import { useRuntimeStore } from '../stores/runtimeStore.js'
import { API_BASE, authFetch } from '../composables/apiClient.js'

const runtimeStore = useRuntimeStore()
const toast = inject('toast', () => {})

const activeTab = ref('tools')
const toolQuery = ref('')
const toolSource = ref('')
const maxIterations = ref(10)
const temperature = ref(0.7)

const installedServers = ref([])
const marketplaceServers = ref([])
const categories = ref([])
const mcpSearch = ref('')
const mcpCategory = ref('')
const mcpLoading = ref(false)
const refreshing = ref(false)

const toolSources = computed(() => {
  const sources = new Set(runtimeStore.tools.map(t => t.source).filter(Boolean))
  return [...sources].sort()
})

const filteredTools = computed(() => {
  let list = runtimeStore.tools
  if (toolSource.value) list = list.filter(t => t.source === toolSource.value)
  if (toolQuery.value) {
    const q = toolQuery.value.toLowerCase()
    list = list.filter(t => t.name?.toLowerCase().includes(q) || t.description?.toLowerCase().includes(q))
  }
  return list
})

function sourceLabel(source) {
  const map = { mcp: 'MCP', local: '本地', core: '核心', builtin: '内置', plugin: '插件' }
  return map[source] || source || '本地'
}

function categoryLabel(tool) {
  if (tool.category) return tool.category
  if (tool.name?.includes('web') || tool.name?.includes('search')) return '网络'
  if (tool.name?.includes('file') || tool.name?.includes('read')) return '文件'
  return '通用'
}

function permissionLabel(tool) {
  if (tool.permission === 'dangerous') return '高权限'
  if (tool.permission === 'read') return '只读'
  return '标准'
}

function saveAgentPrefs() {
  localStorage.setItem('taiji_agent_max_iterations', String(maxIterations.value))
  localStorage.setItem('taiji_agent_temperature', String(temperature.value))
}

async function refreshAgentRuntime() {
  await runtimeStore.refreshTools()
  await loadInstalled()
  toast('已刷新', 'success')
}

async function loadInstalled() {
  try {
    const r = await authFetch(`${API_BASE}/api/mcp/installed`)
    if (r.ok) { const d = await r.json(); installedServers.value = d.servers || [] }
  } catch (e) { toast('加载已安装服务失败: ' + e.message, 'error') }
}

async function loadMarketplace() {
  mcpLoading.value = true
  try {
    const r = await authFetch(`${API_BASE}/api/mcp/marketplace?search=${mcpSearch.value}&category=${mcpCategory.value}`)
    if (r.ok) { const d = await r.json(); marketplaceServers.value = d.servers || []; categories.value = d.categories || [] }
  } catch (e) { toast('加载市场失败: ' + e.message, 'error') }
  mcpLoading.value = false
}

let searchTimer = null
function debounceSearch() {
  clearTimeout(searchTimer)
  searchTimer = setTimeout(loadMarketplace, 400)
}

async function startServer(id) {
  const s = installedServers.value.find(x => x.id === id) || marketplaceServers.value.find(x => x.id === id)
  if (s) s._starting = true
  try { await authFetch(`${API_BASE}/api/mcp/start/${id}`, { method: 'POST' }); await loadInstalled() } catch (e) { toast('启动服务失败: ' + e.message, 'error') }
  if (s) s._starting = false
}

async function stopServer(id) {
  try { await authFetch(`${API_BASE}/api/mcp/stop/${id}`, { method: 'POST' }); await loadInstalled() } catch (e) { toast('停止服务失败: ' + e.message, 'error') }
}

async function restartServer(id) {
  await stopServer(id)
  await startServer(id)
}

async function installServer(id) {
  const s = marketplaceServers.value.find(x => x.id === id)
  if (s) s._installing = true
  try { await authFetch(`${API_BASE}/api/mcp/install/${id}`, { method: 'POST' }); toast('已安装', 'success'); await loadInstalled() } catch (e) { toast('安装服务失败: ' + e.message, 'error') }
  if (s) s._installing = false
}

async function uninstallServer(id) {
  try { await authFetch(`${API_BASE}/api/mcp/uninstall/${id}`, { method: 'DELETE' }); toast('已卸载', 'success'); await loadInstalled() } catch (e) { toast('卸载服务失败: ' + e.message, 'error') }
}

onMounted(() => {
  maxIterations.value = Number(localStorage.getItem('taiji_agent_max_iterations')) || 10
  temperature.value = Number(localStorage.getItem('taiji_agent_temperature')) || 0.7
  refreshAgentRuntime()
})
</script>

<style scoped>
/* ═══ 智能体配置页 · 豆包设计 token 体系 ═══ */
.agent-page {
  height: 100%;
  overflow-y: auto;
  background: var(--background);
  color: var(--foreground);
}

/* 内容容器 */
.page-container {
  max-width: 800px;
  margin: 0 auto;
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 22px;
}

/* 按钮 */
.btn {
  height: 36px;
  padding: 0 15px;
  border-radius: 999px;
  border: 1px solid transparent;
  display: inline-flex;
  align-items: center;
  gap: 7px;
  font-size: 0.86rem;
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;
  transition: background 150ms ease, border-color 150ms ease, transform 120ms ease, color 150ms ease;
}
.btn:active { transform: translateY(1px); }
.btn.sm { height: 30px; padding: 0 12px; font-size: 0.8rem; }
.btn:disabled { opacity: 0.55; cursor: not-allowed; }
.btn-primary { background: var(--primary); color: var(--primary-foreground); }
.btn-primary:hover:not(:disabled) { background: color-mix(in srgb, var(--primary) 90%, var(--foreground)); }
.btn-ghost { background: var(--muted); color: var(--foreground); }
.btn-ghost:hover:not(:disabled) { background: color-mix(in srgb, var(--muted) 80%, var(--foreground) 12%); }
.btn-outline { background: var(--background); color: var(--foreground); border-color: var(--border); }
.btn-outline:hover:not(:disabled) { background: var(--muted); }

/* 配置卡片 */
.cfg-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 18px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.cfg-card:hover { border-color: color-mix(in srgb, var(--primary) 25%, var(--border)); }
.cfg-card-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.cfg-title { display: flex; flex-direction: column; gap: 3px; }
.cfg-card-head h2 { margin: 0; font-size: 1rem; font-weight: 600; color: var(--foreground); }
.cfg-card-head .sub { font-size: 0.76rem; color: var(--muted-foreground); }

/* 表单 */
.form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
.form-field { display: flex; flex-direction: column; gap: 6px; }
.form-field.full { grid-column: 1 / -1; }
.form-label {
  font-size: 0.8rem;
  font-weight: 500;
  color: var(--foreground);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.form-label .req { color: var(--destructive); margin-left: 2px; }
.form-label .hint { font-size: 0.7rem; font-weight: 400; color: var(--muted-foreground); }

.input, .textarea, .select {
  width: 100%;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--background);
  color: var(--foreground);
  padding: 9px 12px;
  font-size: 0.86rem;
  outline: none;
  transition: border-color 160ms ease, box-shadow 160ms ease;
}
.input::placeholder, .textarea::placeholder { color: var(--muted-foreground); }
.input:focus, .textarea:focus, .select:focus {
  border-color: var(--ring);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 18%, transparent);
}
.textarea { resize: vertical; min-height: 76px; line-height: 1.5; }
.select {
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%237f8d9f' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 10px center;
  padding-right: 32px;
}

/* 参数网格 */
.params-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 14px;
}

/* 预设方案 */
.preset-row { display: flex; flex-direction: column; gap: 10px; }
.preset-label { font-size: 0.8rem; font-weight: 500; color: var(--foreground); }
.preset-chips { display: flex; flex-wrap: wrap; gap: 10px; }
.preset-chip {
  flex: 1;
  min-width: 140px;
  display: flex;
  flex-direction: column;
  gap: 2px;
  align-items: flex-start;
  padding: 10px 14px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--card);
  color: var(--foreground);
  cursor: pointer;
  transition: border-color 160ms ease, background 160ms ease, transform 160ms ease;
}
.preset-chip:hover { border-color: var(--primary); background: var(--primary-subtle); transform: translateY(-1px); }
.chip-name { font-size: 0.84rem; font-weight: 600; }
.chip-meta { font-size: 0.7rem; color: var(--muted-foreground); font-family: var(--font-mono); }

/* 操作行 */
.action-row { display: flex; gap: 10px; }

/* 概览网格 */
.overview-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
  gap: 12px;
}
.ov-card {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 16px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--card);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.ov-card:hover { border-color: color-mix(in srgb, var(--primary) 25%, var(--border)); }
.ov-card.connected { border-color: var(--success); }
.ov-ic {
  width: 36px; height: 36px; border-radius: 10px;
  display: grid; place-items: center; flex: none;
  background: color-mix(in srgb, var(--primary) 14%, transparent);
  color: var(--primary);
}
.ov-card.connected .ov-ic { background: var(--success-light); color: var(--success); }
.ov-ic .ic-svg { width: 18px; height: 18px; stroke: currentColor; fill: none; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }
.ov-text { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.ov-label { font-size: 0.72rem; color: var(--muted-foreground); }
.ov-value { font-size: 0.92rem; font-weight: 600; color: var(--foreground); display: inline-flex; align-items: center; gap: 6px; }
.ov-unit { font-size: 0.74rem; font-weight: 400; color: var(--muted-foreground); }
.ov-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted-foreground); flex: none; }
.ov-card.connected .ov-dot { background: var(--success); }

/* 标签页 */
.tabs { display: flex; align-items: center; gap: 4px; border-bottom: 1px solid var(--border); padding: 0 4px; }
.tab {
  appearance: none;
  background: transparent;
  border: 0;
  padding: 10px 14px;
  font-size: 0.88rem;
  color: var(--muted-foreground);
  cursor: pointer;
  position: relative;
  transition: color 150ms ease;
}
.tab:hover { color: var(--foreground); }
.tab.active { color: var(--foreground); font-weight: 600; }
.tab.active::after {
  content: '';
  position: absolute;
  left: 10px; right: 10px; bottom: -1px;
  height: 2px; border-radius: 2px; background: var(--primary);
}
.tab-panel { display: flex; flex-direction: column; gap: 16px; }

/* 筛选 */
.filter-row { display: flex; gap: 8px; align-items: center; }
.search-field {
  flex: 1;
  min-width: 0;
  height: 38px;
  padding: 0 12px;
  border-radius: 10px;
  background: var(--muted);
  border: 1px solid transparent;
  display: flex;
  align-items: center;
  gap: 9px;
  color: var(--muted-foreground);
  transition: border-color 160ms ease, background 160ms ease;
}
.search-field:focus-within { border-color: var(--ring); background: var(--background); }
.search-field input {
  flex: 1; min-width: 0; border: 0; outline: none; background: transparent;
  font-size: 0.86rem; color: var(--foreground);
}
.search-field input::placeholder { color: var(--muted-foreground); }
.filter-row .select { width: 140px; height: 38px; flex: none; }

/* 工具网格 */
.tool-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}
.tool-card {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px;
  background: var(--card);
  display: flex;
  flex-direction: column;
  gap: 10px;
  transition: border-color 160ms ease, background 160ms ease, transform 160ms ease;
}
.tool-card:hover { border-color: color-mix(in srgb, var(--primary) 35%, var(--border)); transform: translateY(-1px); }
.tool-card.off { opacity: 0.65; }
.tool-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.tool-ic {
  width: 34px; height: 34px; border-radius: 9px;
  display: grid; place-items: center;
  background: color-mix(in srgb, var(--primary) 14%, transparent);
  color: var(--primary);
  flex: none;
}
.tool-card.off .tool-ic { background: var(--muted); color: var(--muted-foreground); }
.ic-svg { width: 18px; height: 18px; stroke: currentColor; fill: none; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }
.tool-name { font-size: 0.84rem; font-weight: 600; color: var(--foreground); word-break: break-word; }
.tool-desc { font-size: 0.74rem; color: var(--muted-foreground); line-height: 1.45; min-height: 2.2em; }

/* 开关 toggle */
.toggle {
  width: 38px; height: 22px; border-radius: 999px;
  background: var(--muted); border: 1px solid var(--border);
  position: relative; cursor: pointer;
  transition: background 160ms ease, border-color 160ms ease;
  flex: none; padding: 0;
}
.toggle::after {
  content: ''; position: absolute; top: 2px; left: 2px;
  width: 16px; height: 16px; border-radius: 50%;
  background: var(--background);
  box-shadow: 0 1px 2px rgba(0,0,0,0.18);
  transition: transform 180ms cubic-bezier(.4,0,.2,1);
}
.toggle.on { background: var(--primary); border-color: var(--primary); }
.toggle.on::after { transform: translateX(16px); background: var(--primary-foreground); box-shadow: none; }

/* 状态 chip */
.status-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 10px; border-radius: 999px;
  font-size: 0.74rem; font-weight: 500;
  white-space: nowrap;
}
.status-chip::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.status-running { background: color-mix(in srgb, var(--chart-2) 18%, transparent); color: var(--chart-2); }
.status-stopped { background: color-mix(in srgb, var(--muted-foreground) 16%, transparent); color: var(--muted-foreground); }

/* MCP 表格 */
.mcp-table-wrap { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; background: var(--card); }
.mcp-table { width: 100%; border-collapse: collapse; }
.mcp-table th {
  text-align: left;
  font: 600 0.7rem/1 var(--font-mono);
  letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--muted-foreground);
  padding: 11px 16px;
  border-bottom: 1px solid var(--border);
  background: var(--muted);
}
.mcp-table td {
  padding: 12px 16px;
  border-bottom: 1px solid color-mix(in srgb, var(--border) 55%, transparent);
  font-size: 0.84rem;
  color: var(--foreground);
  vertical-align: middle;
}
.mcp-table tbody tr:last-child td { border-bottom: 0; }
.mcp-table tbody tr { transition: background 120ms ease; }
.mcp-table tbody tr:hover { background: color-mix(in srgb, var(--accent) 18%, transparent); }
.mcp-name { display: flex; align-items: center; gap: 9px; font-weight: 500; }
.mcp-ic {
  width: 26px; height: 26px; border-radius: 7px;
  background: color-mix(in srgb, var(--primary) 14%, transparent);
  color: var(--primary);
  display: grid; place-items: center; flex: none;
}
.mcp-ic .ic-svg { width: 15px; height: 15px; }
.mcp-protocol { font-family: var(--font-mono); font-size: 0.76rem; color: var(--muted-foreground); }
.mcp-endpoint { font-family: var(--font-mono); font-size: 0.76rem; color: var(--muted-foreground); }

/* 行操作（图标） */
.row-actions { display: flex; align-items: center; gap: 4px; }
.act-btn {
  width: 28px; height: 28px; border: 0; border-radius: 8px;
  background: transparent; color: var(--muted-foreground);
  display: grid; place-items: center; cursor: pointer;
  transition: background 140ms ease, color 140ms ease;
}
.act-btn:hover { background: var(--muted); color: var(--foreground); }
.act-btn .ic-svg { width: 15px; height: 15px; stroke: currentColor; fill: none; stroke-width: 1.7; stroke-linecap: round; stroke-linejoin: round; }

/* 市场卡片操作按钮 */
.mc-actions { display: flex; gap: 6px; flex-wrap: wrap; }
.mc-btn {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 5px 12px; border: 1px solid var(--border); border-radius: 999px;
  background: var(--card); color: var(--muted-foreground);
  font-size: 0.74rem; cursor: pointer;
  transition: background 140ms ease, color 140ms ease, border-color 140ms ease;
}
.mc-btn:hover { background: var(--muted); color: var(--foreground); }

/* 底部 */
.panel-foot { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding-top: 4px; }
.foot-hint { font-size: 0.76rem; color: var(--muted-foreground); }

.empty-msg {
  padding: 36px; text-align: center;
  color: var(--muted-foreground); font-size: 0.84rem;
  border: 1px dashed color-mix(in srgb, var(--muted-foreground) 30%, transparent);
  border-radius: 12px;
}

.spin { animation: ag-spin 0.9s linear infinite; }
@keyframes ag-spin { to { transform: rotate(360deg); } }

@media (max-width: 880px) {
  .page-container { padding: 16px; }
  .overview-grid { grid-template-columns: 1fr; }
  .tool-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .form-grid { grid-template-columns: 1fr; }
  .filter-row { flex-wrap: wrap; }
  .filter-row .select { width: 100%; }
}
</style>
