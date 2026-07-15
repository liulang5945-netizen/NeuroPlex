<template>
  <div class="agent-page">
    <header class="agent-head">
      <div class="head-left">
        <h1>Agent 配置</h1>
        <span class="head-stat">{{ runtimeStore.tools.length }} 个工具 · {{ installedServers.length }} 个 MCP</span>
      </div>
      <button class="head-btn" @click="refreshAgentRuntime" :disabled="runtimeStore.toolsLoading">
        <RefreshCw :size="14" :class="{ spin: runtimeStore.toolsLoading }" /> 刷新
      </button>
    </header>

    <!-- 状态概览 -->
    <div class="overview-row">
      <div class="ov-card" :class="runtimeStore.connectionClass">
        <span class="ov-dot"></span>
        <div><strong>{{ runtimeStore.connectionStatus }}</strong><small>{{ runtimeStore.modelLifecycle.message }}</small></div>
      </div>
      <div class="ov-card">
        <span class="ov-num">{{ runtimeStore.tools.length }}</span>
        <span class="ov-label">工具</span>
      </div>
      <div class="ov-card">
        <span class="ov-num">{{ installedServers.length }}</span>
        <span class="ov-label">MCP</span>
      </div>
      <div class="ov-card" v-if="runtimeStore.memoryAvailableGb">
        <span class="ov-num">{{ runtimeStore.memoryAvailableGb.toFixed(1) }}</span>
        <span class="ov-label">GB 可用</span>
      </div>
    </div>

    <!-- 参数 -->
    <div class="params-row">
      <label class="param"><span>最大迭代</span><input v-model.number="maxIterations" type="number" min="1" max="50" @change="saveAgentPrefs" /></label>
      <label class="param"><span>温度</span><input v-model.number="temperature" type="number" min="0" max="2" step="0.1" @change="saveAgentPrefs" /></label>
    </div>

    <!-- Tab 切换 -->
    <div class="tabs">
      <button :class="{ active: activeTab === 'tools' }" @click="activeTab = 'tools'">工具 ({{ filteredTools.length }})</button>
      <button :class="{ active: activeTab === 'installed' }" @click="activeTab = 'installed'; loadInstalled()">MCP 运行</button>
      <button :class="{ active: activeTab === 'marketplace' }" @click="activeTab = 'marketplace'; loadMarketplace()">MCP 市场</button>
    </div>

    <!-- 工具列表 -->
    <section v-if="activeTab === 'tools'" class="tab-body">
      <div class="filter-row">
        <input v-model="toolQuery" placeholder="搜索工具名或描述..." />
        <select v-model="toolSource">
          <option value="">全部来源</option>
          <option v-for="s in toolSources" :key="s" :value="s">{{ s }}</option>
        </select>
      </div>
      <div class="tool-grid">
        <div v-for="tool in filteredTools" :key="tool.name" class="tool-card">
          <div class="tc-head">
            <span class="tc-badge" :class="tool.source || 'local'">{{ sourceLabel(tool.source) }}</span>
            <span class="tc-status" :class="{ on: tool.enabled !== false }">{{ tool.enabled === false ? '停用' : '可用' }}</span>
          </div>
          <h4>{{ tool.name }}</h4>
          <p>{{ tool.description || '暂无描述' }}</p>
          <div class="tc-meta">
            <span>{{ categoryLabel(tool) }}</span>
            <span>{{ permissionLabel(tool) }}</span>
          </div>
        </div>
      </div>
      <div v-if="!filteredTools.length && !runtimeStore.toolsLoading" class="empty-msg">无匹配工具</div>
    </section>

    <!-- MCP 运行 -->
    <section v-if="activeTab === 'installed'" class="tab-body">
      <div v-for="server in installedServers" :key="server.id" class="mcp-card">
        <div class="mc-head">
          <div class="mc-info">
            <h4>{{ server.name || server.id }}</h4>
            <small>{{ server.npm_package || server.id }}</small>
          </div>
          <span class="mc-status" :class="{ running: server.running }">{{ server.running ? '运行中' : '已停止' }}</span>
        </div>
        <p class="mc-desc">{{ server.description || '暂无描述' }}</p>
        <div v-if="server.runtime_info?.tools?.length" class="mc-tools">
          <span v-for="t in server.runtime_info.tools.slice(0, 8)" :key="t.name">{{ t.name }}</span>
          <span v-if="server.runtime_info.tools.length > 8">+{{ server.runtime_info.tools.length - 8 }}</span>
        </div>
        <div class="mc-actions">
          <button v-if="!server.running" class="act-btn green" @click="startServer(server.id)"><Play :size="12" /> 启动</button>
          <button v-else class="act-btn yellow" @click="stopServer(server.id)"><Square :size="12" /> 停止</button>
          <button v-if="server.running" class="act-btn" @click="restartServer(server.id)"><RefreshCw :size="12" /> 重启</button>
          <button class="act-btn red" @click="uninstallServer(server.id)"><Trash2 :size="12" /> 卸载</button>
        </div>
      </div>
      <div v-if="!installedServers.length" class="empty-msg">暂无已安装 MCP 服务</div>
    </section>

    <!-- MCP 市场 -->
    <section v-if="activeTab === 'marketplace'" class="tab-body">
      <div class="filter-row">
        <input v-model="mcpSearch" placeholder="搜索 MCP 服务..." @input="debounceSearch" />
        <select v-model="mcpCategory" @change="loadMarketplace">
          <option value="">全部分类</option>
          <option v-for="c in categories" :key="c" :value="c">{{ c }}</option>
        </select>
        <button class="act-btn" @click="refreshMarketplace" :disabled="refreshing"><RefreshCw :size="12" :class="{ spin: refreshing }" /> 同步</button>
      </div>
      <div class="tool-grid">
        <div v-for="server in marketplaceServers" :key="server.id" class="tool-card mc">
          <div class="tc-head">
            <span class="tc-badge">{{ server.category || '未分类' }}</span>
            <span class="tc-status" :class="{ on: server.running }">{{ server.running ? '运行中' : server.installed ? '已安装' : '' }}</span>
          </div>
          <h4>{{ server.name }}</h4>
          <p>{{ server.description || '暂无描述' }}</p>
          <div class="tc-meta">
            <span>{{ server.npm_package }}</span>
            <span>{{ server.tools_count || '?' }} 工具</span>
          </div>
          <div class="mc-actions">
            <button v-if="!server.installed" class="act-btn blue" @click="installServer(server.id)"><Download :size="12" /> 安装</button>
            <template v-else>
              <button v-if="!server.running" class="act-btn green" @click="startServer(server.id)"><Play :size="12" /> 启动</button>
              <button v-else class="act-btn yellow" @click="stopServer(server.id)"><Square :size="12" /> 停止</button>
              <button class="act-btn red" @click="uninstallServer(server.id)"><Trash2 :size="12" /> 卸载</button>
            </template>
          </div>
        </div>
      </div>
      <div v-if="!marketplaceServers.length && !mcpLoading" class="empty-msg">{{ mcpLoading ? '加载中...' : '无匹配结果' }}</div>
    </section>
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
.agent-page {
  height: 100%; overflow-y: auto; padding: 20px 24px;
  display: flex; flex-direction: column; gap: 16px;
  background: var(--bg); color: var(--text);
}

/* 头部 */
.agent-head { display: flex; justify-content: space-between; align-items: center; }
.head-left h1 { margin: 0; font-size: 20px; font-weight: 650; color: var(--text); }
.head-stat { font-size: 12px; color: var(--text-muted); }
.head-btn {
  display: flex; align-items: center; gap: 5px; padding: 7px 14px;
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  background: var(--bg-card); color: var(--text-secondary); font-size: 13px;
  cursor: pointer; transition: var(--transition);
}
.head-btn:hover { background: var(--bg-hover); }

/* 概览 */
.overview-row { display: flex; gap: 10px; flex-wrap: wrap; }
.ov-card {
  display: flex; align-items: center; gap: 10px; padding: 12px 16px;
  border: 1px solid var(--border); border-radius: var(--radius-md);
  background: var(--bg-card); flex: 1; min-width: 140px;
}
.ov-card.connected { border-color: var(--success); }
.ov-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--text-muted); flex-shrink: 0; }
.ov-card.connected .ov-dot { background: var(--success); }
.ov-card strong { display: block; font-size: 13px; color: var(--text); }
.ov-card small { display: block; font-size: 11px; color: var(--text-muted); margin-top: 2px; }
.ov-num { font-size: 22px; font-weight: 700; color: var(--text); }
.ov-label { font-size: 12px; color: var(--text-muted); }

/* 参数 */
.params-row { display: flex; gap: 12px; }
.param {
  display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-secondary);
}
.param input {
  width: 70px; height: 30px; padding: 0 8px; border: 1px solid var(--border);
  border-radius: var(--radius-sm); background: var(--bg-card); color: var(--text);
  font-size: 13px; outline: none;
}
.param input:focus { border-color: var(--primary); }

/* Tabs */
.tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--border); padding-bottom: 0; }
.tabs button {
  display: flex; align-items: center; gap: 5px; padding: 8px 14px;
  border: 0; border-radius: var(--radius-sm) var(--radius-sm) 0 0;
  background: transparent; color: var(--text-muted); font-size: 13px;
  cursor: pointer; transition: var(--transition); border-bottom: 2px solid transparent;
  margin-bottom: -1px;
}
.tabs button:hover { color: var(--text-secondary); }
.tabs button.active { color: var(--primary); border-bottom-color: var(--primary); background: var(--bg-card); }

.tab-body { flex: 1; min-height: 0; }

/* 筛选 */
.filter-row { display: flex; gap: 8px; margin-bottom: 12px; }
.filter-row input, .filter-row select {
  height: 34px; padding: 0 10px; border: 1px solid var(--border);
  border-radius: var(--radius-sm); background: var(--bg-card); color: var(--text);
  font-size: 13px; outline: none;
}
.filter-row input { flex: 1; min-width: 0; }
.filter-row input:focus { border-color: var(--primary); }
.filter-row select { width: 130px; }

/* 工具网格 */
.tool-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 10px;
}
.tool-card {
  padding: 14px; border: 1px solid var(--border); border-radius: var(--radius-md);
  background: var(--bg-card); transition: var(--transition);
}
.tool-card:hover { border-color: var(--primary-light); background: var(--primary-subtle); }
.tc-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.tc-badge {
  padding: 2px 8px; border-radius: 999px; font-size: 11px;
  background: var(--bg-muted); color: var(--text-muted); border: 1px solid var(--border);
}
.tc-status { padding: 2px 8px; border-radius: 999px; font-size: 11px; background: var(--bg-muted); color: var(--text-muted); }
.tc-status.on { color: var(--success); background: var(--success-light); }
.tool-card h4, .tool-card h4 { margin: 0 0 4px; font-size: 14px; font-weight: 600; color: var(--text); word-break: break-word; }
.tool-card p { margin: 0 0 8px; font-size: 12px; color: var(--text-secondary); line-height: 1.5; min-height: 36px; }
.tc-meta { display: flex; gap: 4px; flex-wrap: wrap; }
.tc-meta span { padding: 1px 6px; border-radius: 999px; font-size: 10px; background: var(--bg-muted); color: var(--text-muted); }

/* MCP 卡片 */
.mcp-card {
  padding: 14px; border: 1px solid var(--border); border-radius: var(--radius-md);
  background: var(--bg-card); transition: var(--transition);
}
.mc-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; }
.mc-info { flex: 1; min-width: 0; }
.mc-info h4 { margin: 0; font-size: 14px; font-weight: 600; color: var(--text); }
.mc-info small { display: block; margin-top: 2px; font-size: 11px; color: var(--text-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.mc-status { padding: 3px 10px; border-radius: 999px; font-size: 11px; background: var(--bg-muted); color: var(--text-muted); white-space: nowrap; }
.mc-status.running { color: var(--success); background: var(--success-light); }
.mc-desc { margin: 8px 0; font-size: 12px; color: var(--text-secondary); line-height: 1.5; min-height: 36px; }
.mc-tools { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 10px; }
.mc-tools span { padding: 2px 7px; border-radius: 999px; font-size: 10px; background: var(--bg-muted); color: var(--text-muted); border: 1px solid var(--border); }
.mc-actions { display: flex; gap: 6px; flex-wrap: wrap; }

/* 操作按钮 */
.act-btn {
  display: inline-flex; align-items: center; gap: 4px; padding: 5px 12px;
  border: 1px solid var(--border); border-radius: var(--radius-sm);
  background: var(--bg-card); color: var(--text-secondary); font-size: 12px;
  cursor: pointer; transition: var(--transition);
}
.act-btn:hover { background: var(--bg-hover); }
.act-btn.green { color: var(--success); border-color: var(--success); }
.act-btn.green:hover { background: var(--success-light); }
.act-btn.yellow { color: var(--warning); border-color: var(--warning); }
.act-btn.yellow:hover { background: var(--warning-light); }
.act-btn.red { color: var(--danger); border-color: var(--danger); }
.act-btn.red:hover { background: var(--danger-light); }

.empty-msg { padding: 32px; text-align: center; color: var(--text-muted); font-size: 13px; border: 1px dashed var(--border); border-radius: var(--radius-md); }

.spin { animation: spin 0.9s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

@media (max-width: 760px) {
  .agent-page { padding: 12px; }
  .overview-row { flex-direction: column; }
  .tool-grid { grid-template-columns: 1fr; }
}
</style>
