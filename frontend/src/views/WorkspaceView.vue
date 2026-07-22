<template>
  <div class="workspace-view">
    <!-- 顶栏 -->
    <header class="topbar">
      <div>
        <div class="topbar-title">IDE 工作区</div>
        <div class="topbar-sub">态极脚本与配置编辑</div>
      </div>
      <div class="topbar-spacer"></div>
      <button class="btn btn-primary">
        <svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        运行
      </button>
      <button class="btn btn-outline">保存</button>
    </header>

    <!-- 主体工作区 -->
    <div class="workspace-body">
      <div class="ide-layout" :style="{ gridTemplateColumns: sidebarWidth + 'px minmax(0, 1fr) 260px' }">
        <!-- 左栏：文件树 -->
        <div class="panel panel-left">
          <div class="panel-header">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>
            项目文件
          </div>
          <div class="panel-body">
            <div v-if="!fileTree.length" class="tree-empty">空工作台</div>
            <template v-for="node in fileTree" :key="node.path">
              <div
                class="tree-item"
                :class="{ 'tree-folder': node.type === 'directory' }"
                :style="{ paddingLeft: (node.depth * 18 + 8) + 'px' }"
                @click="handleTreeClick(node)"
                @contextmenu.prevent="showContextMenu($event, node)"
              >
                <component :is="node.type === 'directory' ? (expandedDirs.has(node.path) ? FolderOpen : Folder) : getFileIcon(node.name)" :size="14" class="tree-icon" />
                <span class="tree-label">{{ node.name }}</span>
              </div>
            </template>
          </div>
          <div class="resize-col" @mousedown="startResize"></div>
        </div>

        <!-- 中栏：编辑器 + 终端 -->
        <div class="panel panel-center">
          <div class="editor-area">
            <MonacoEditor ref="monacoEditor" class="monaco-container" />
            <!-- 终端 -->
            <Transition name="term-slide">
              <div v-if="showTerminal" class="ide-terminal" :style="{ height: terminalHeight + 'px' }">
                <div class="resize-row" @mousedown="startTerminalResize"></div>
                <WebTerminal ref="webTerminal" />
              </div>
            </Transition>
          </div>
        </div>

        <!-- 右栏：属性与检查器 -->
        <div class="panel panel-right">
          <div class="panel-header">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 3"/></svg>
            属性
          </div>
          <div class="panel-body">
            <div v-if="currentFile" class="prop-group">
              <div class="prop-group-title">文件信息</div>
              <div class="prop-row"><span class="prop-label">文件名</span><span class="prop-value">{{ currentFile.name }}</span></div>
              <div class="prop-row"><span class="prop-label">路径</span><span class="prop-value prop-truncate" :title="currentFile.path">{{ currentFile.path }}</span></div>
              <div class="prop-row"><span class="prop-label">大小</span><span class="prop-value">{{ formatFileSize(currentFile.content?.length || 0) }}</span></div>
              <div class="prop-row"><span class="prop-label">编码</span><span class="prop-value">UTF-8</span></div>
              <div class="prop-row"><span class="prop-label">行数</span><span class="prop-value">{{ countLines(currentFile.content) }}</span></div>
              <div class="prop-row"><span class="prop-label">类型</span><span class="prop-value">{{ currentFile.language?.toUpperCase() || '—' }}</span></div>
            </div>
            <div v-else class="prop-empty">未打开文件</div>
            <div class="prop-group">
              <div class="prop-group-title">态极检查器</div>
              <div class="inspector-item"><span class="inspector-dot ok"></span><span class="inspector-text">YAML 语法校验</span><span class="inspector-meta">通过</span></div>
              <div class="inspector-item"><span class="inspector-dot info"></span><span class="inspector-text">配置完整性</span><span class="inspector-meta">6/6 节</span></div>
              <div class="inspector-item"><span class="inspector-dot ok"></span><span class="inspector-text">分布式策略对齐</span><span class="inspector-meta">128 GPU</span></div>
            </div>
            <div class="prop-group">
              <div class="prop-group-title">快捷操作</div>
              <button class="quick-btn"><svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>运行全部测试</button>
              <button class="quick-btn"><svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>导出配置</button>
              <button class="quick-btn"><svg class="icon-sm" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>在终端中打开</button>
            </div>
          </div>
        </div>
      </div>

      <!-- 底部状态栏 -->
      <div class="status-bar">
        <span v-if="monacoEditor?.activeTab">Ln {{ monacoEditor.cursorLine }}, Col {{ monacoEditor.cursorCol }}</span>
        <span v-else>—</span>
        <span>UTF-8</span>
        <span>{{ monacoEditor?.language?.toUpperCase() || '—' }}</span>
        <span v-if="monacoEditor?.isDirty" class="status-dirty">● 未保存</span>
        <span class="status-spacer"></span>
        <span class="status-dot"></span>
        <span>神经元同步中</span>
      </div>
    </div>

    <!-- 对话框 -->
    <input ref="folderPicker" type="file" webkitdirectory directory style="display:none" @change="onFolderSelected" />
    <div v-if="inputDialog.visible" class="dlg-overlay" @click.self="cancelInputDialog">
      <div class="dlg-box">
        <h3>{{ inputDialog.title }}</h3>
        <input v-model="inputDialog.value" class="dlg-input" :placeholder="inputDialog.placeholder" @keydown.enter="confirmInputDialog" @keydown.escape="cancelInputDialog" />
        <div class="dlg-actions">
          <button class="dlg-btn primary" @click="confirmInputDialog">确认</button>
          <button class="dlg-btn" @click="cancelInputDialog">取消</button>
        </div>
      </div>
    </div>
    <div v-if="showPathDialog" class="dlg-overlay" @click.self="showPathDialog = false">
      <div class="dlg-box">
        <h3>切换项目路径</h3>
        <div class="quick-paths" v-if="quickPaths.length">
          <button v-for="qp in quickPaths" :key="qp.path" class="qp-btn" @click="newPathInput = qp.path">
            <FolderOpen :size="11" /> {{ qp.label }}
          </button>
        </div>
        <input v-model="newPathInput" class="dlg-input" placeholder="输入完整路径" @keydown.enter="applyNewPath" />
        <div class="dlg-actions">
          <button class="dlg-btn primary" @click="applyNewPath">切换</button>
          <button class="dlg-btn" @click="showPathDialog = false">取消</button>
        </div>
        <p v-if="pathDialogError" class="dlg-error">{{ pathDialogError }}</p>
      </div>
    </div>

    <!-- 右键菜单 -->
    <div v-if="contextMenu.visible" class="ctx-menu" :style="{ left: contextMenu.x + 'px', top: contextMenu.y + 'px' }" @click="contextMenu.visible = false">
      <div class="ctx-item" @click="openInEditor"><Edit3 :size="13" /> 打开</div>
      <div class="ctx-sep"></div>
      <div class="ctx-item" @click="renameItem"><Edit2 :size="13" /> 重命名</div>
      <div class="ctx-sep"></div>
      <div class="ctx-item danger" @click="deleteItem"><Trash2 :size="13" /> 删除</div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted, inject } from 'vue';
import { useApi } from '../composables/useApi.js';
import { API_BASE, authFetch } from '../composables/apiClient.js';
import { RefreshCw, FilePlus, FolderPlus, Terminal, FolderOpen, Folder, FileCode, FileText, Image as ImageIcon, Database, Edit3, Edit2, Trash2 } from 'lucide-vue-next';
import MonacoEditor from '../components/MonacoEditor.vue';
import WebTerminal from '../components/WebTerminal.vue';

const { t } = useApi();
const toast = inject('toast');
const $confirm = inject('$confirm');

const workspacePath = ref('');
const fileTree = ref([]);
const expandedDirs = reactive(new Set());
const showTerminal = ref(false);
const sidebarWidth = ref(220);
const terminalHeight = ref(280);
const monacoEditor = ref(null);
const folderPicker = ref(null);
const contextMenu = ref({ visible: false, x: 0, y: 0, node: null });
const inputDialog = ref({ visible: false, title: '', value: '', placeholder: '', resolve: null });
const showPathDialog = ref(false);
const newPathInput = ref('');
const pathDialogError = ref('');
const quickPaths = ref([]);

// 当前激活文件（从 MonacoEditor 读取）
const currentFile = computed(() => {
  if (!monacoEditor.value?.openTabs || !monacoEditor.value?.activeTab) return null;
  return monacoEditor.value.openTabs.find(t => t.path === monacoEditor.value.activeTab) || null;
});

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function countLines(content) {
  if (!content) return 0;
  return content.split('\n').length;
}

function showInputDialog(title, placeholder = '') {
  return new Promise((resolve) => {
    inputDialog.value = { visible: true, title, value: '', placeholder, resolve };
  });
}
function confirmInputDialog() {
  const val = inputDialog.value.value.trim();
  inputDialog.value.visible = false;
  if (inputDialog.value.resolve) inputDialog.value.resolve(val || null);
}
function cancelInputDialog() {
  inputDialog.value.visible = false;
  if (inputDialog.value.resolve) inputDialog.value.resolve(null);
}

function openFolderPicker() {
  if (folderPicker.value) { folderPicker.value.value = ''; folderPicker.value.click(); }
}

function onFolderSelected(event) {
  const files = event.target.files;
  if (!files || files.length) return;
  showPathDialog.value = true;
}

async function applyNewPath() {
  const path = newPathInput.value.trim();
  if (!path) { pathDialogError.value = '请输入路径'; return; }
  pathDialogError.value = '';
  try {
    const r = await authFetch(`${API_BASE}/api/workspace/path`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    const data = await r.json();
    if (r.ok && data.status === 'ok') {
      workspacePath.value = data.path;
      showPathDialog.value = false;
      newPathInput.value = '';
      expandedDirs.clear();
      loadTree();
      toast('路径已切换', 'success');
    } else {
      pathDialogError.value = data.detail || '路径设置失败';
    }
  } catch (e) { pathDialogError.value = e.message; }
}

async function loadWorkspacePath() {
  try {
    const r = await authFetch(`${API_BASE}/api/workspace/path`);
    if (r.ok) { const d = await r.json(); workspacePath.value = d.path || ''; }
  } catch (e) { toast('加载工作路径失败: ' + e.message, 'error') }
}

async function loadTree() {
  try {
    const r = await authFetch(`${API_BASE}/api/workspace/tree`);
    if (r.ok) { const d = await r.json(); fileTree.value = flattenTree(d.tree || [], 0); }
  } catch (e) { toast('加载文件树失败: ' + e.message, 'error') }
}

function flattenTree(nodes, depth) {
  let result = [];
  for (const node of nodes) {
    node.depth = depth;
    result.push(node);
    if (node.type === 'directory' && expandedDirs.has(node.path) && node.children) {
      result = result.concat(flattenTree(node.children, depth + 1));
    }
  }
  return result;
}

function handleTreeClick(node) {
  if (node.type === 'directory') {
    if (expandedDirs.has(node.path)) expandedDirs.delete(node.path);
    else expandedDirs.add(node.path);
    loadTree();
  } else {
    if (monacoEditor.value) monacoEditor.value.openFile(node.path);
  }
}

function getFileIcon(name) {
  const ext = name.split('.').pop()?.toLowerCase();
  const map = { py: FileCode, js: FileCode, ts: FileCode, json: FileCode, html: FileCode, css: FileCode, vue: FileCode, md: FileText, txt: FileText, sh: Terminal, sql: Database, png: ImageIcon, jpg: ImageIcon };
  return map[ext] || FileText;
}

function showContextMenu(e, node) { contextMenu.value = { visible: true, x: e.clientX, y: e.clientY, node }; }
function openInEditor() { if (contextMenu.value.node?.type === 'file' && monacoEditor.value) monacoEditor.value.openFile(contextMenu.value.node.path); }
async function renameItem() { toast('重命名功能开发中', 'info'); }
async function deleteItem() {
  const node = contextMenu.value.node;
  if (!node) return;
  const ok = await $confirm({ title: '删除确认', message: `确定删除 ${node.name}？`, type: 'danger' });
  if (!ok) return;
  try { const r = await authFetch(`${API_BASE}/api/workspace/delete/${node.path}`, { method: 'DELETE' }); if (r.ok) loadTree(); } catch (e) { toast('删除失败: ' + e.message, 'error') }
}
async function createNewFile() {
  const name = await showInputDialog('文件名:');
  if (!name) return;
  try { await authFetch(`${API_BASE}/api/workspace/file`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, content: '' }) }); loadTree(); if (monacoEditor.value) monacoEditor.value.openFile(name); } catch (e) { toast('创建文件失败: ' + e.message, 'error') }
}
async function createNewFolder() {
  const name = await showInputDialog('文件夹名:');
  if (!name) return;
  try { await authFetch(`${API_BASE}/api/workspace/file`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: name + '/.gitkeep', content: '' }) }); loadTree(); } catch (e) { toast('创建文件夹失败: ' + e.message, 'error') }
}

let resizing = false;
function startResize(e) {
  resizing = true;
  const startX = e.clientX, startW = sidebarWidth.value;
  const onMove = (ev) => { sidebarWidth.value = Math.max(120, Math.min(400, startW + ev.clientX - startX)); };
  const onUp = () => { resizing = false; document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}

function startTerminalResize(e) {
  const startY = e.clientY, startH = terminalHeight.value;
  const onMove = (ev) => { terminalHeight.value = Math.max(80, Math.min(500, startH - (ev.clientY - startY))); };
  const onUp = () => { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}

function closeCtx() { contextMenu.value.visible = false; }
onMounted(() => {
  document.addEventListener('click', closeCtx);
  loadWorkspacePath();
  loadTree();
});
onUnmounted(() => { document.removeEventListener('click', closeCtx); });
</script>

<style scoped>
.workspace-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  width: 100%;
  overflow: hidden;
  background: var(--background);
  color: var(--foreground);
}

/* ── 顶栏 ── */
.topbar {
  height: 52px;
  flex: none;
  padding: 0 18px;
  display: flex;
  align-items: center;
  gap: 12px;
  border-bottom: 1px solid var(--border);
}
.topbar-title { font-size: 0.92rem; font-weight: 600; }
.topbar-sub { font-size: 0.72rem; color: var(--muted-foreground); margin-top: 1px; }
.topbar-spacer { flex: 1; }

/* ── 按钮 ── */
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
  transition: background 150ms ease, border-color 150ms ease, transform 120ms ease, color 150ms ease;
  cursor: pointer;
  font-family: inherit;
}
.btn:active { transform: translateY(1px); }
.btn:focus-visible { outline: 2px solid var(--ring); outline-offset: 2px; }
.btn-primary { background: var(--primary); color: var(--primary-foreground); }
.btn-primary:hover { background: color-mix(in srgb, var(--primary) 90%, var(--foreground)); }
.btn-outline { background: var(--background); color: var(--foreground); border-color: var(--border); }
.btn-outline:hover { background: var(--muted); }
.icon-sm { width: 15px; height: 15px; flex: none; }

/* ── 主体工作区 ── */
.workspace-body {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

/* ── 三栏 IDE 布局 ── */
.ide-layout {
  flex: 1;
  min-height: 0;
  display: grid;
}
.panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
  position: relative;
}
.panel-left { border-right: 1px solid var(--border); }
.panel-center { border-right: 1px solid var(--border); }
.panel-right { border-left: 1px solid var(--border); }

.panel-header {
  padding: 12px 14px;
  font-size: 0.74rem;
  font-weight: 600;
  color: var(--muted-foreground);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--border);
  flex: none;
  display: flex;
  align-items: center;
  gap: 6px;
}
.panel-body {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

/* ── 文件树 ── */
.tree-empty {
  text-align: center;
  padding: 24px 12px;
  color: var(--muted-foreground);
  font-size: 0.8rem;
}
.tree-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 8px;
  border-radius: 6px;
  font-size: 0.84rem;
  cursor: pointer;
  transition: background 120ms ease;
  color: var(--foreground);
}
.tree-item:hover { background: var(--muted); }
.tree-folder { font-weight: 600; }
.tree-icon {
  width: 16px;
  height: 16px;
  flex: none;
  color: var(--muted-foreground);
}
.tree-label {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.resize-col {
  position: absolute;
  top: 0;
  right: -2px;
  width: 4px;
  height: 100%;
  cursor: col-resize;
  z-index: 10;
  transition: background 0.15s;
}
.resize-col:hover { background: var(--primary); }

/* ── 编辑器区域 ── */
.editor-area {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.monaco-container {
  flex: 1;
  min-height: 0;
  width: 100%;
}
.editor-area :deep(.monaco-wrapper) {
  height: 100% !important;
  min-height: 0 !important;
}
.editor-area :deep(.monaco-editor-container) {
  flex: 1 !important;
  min-height: 0 !important;
  height: auto !important;
}

/* ── 终端 ── */
.ide-terminal {
  flex-shrink: 0;
  border-top: 1px solid var(--border);
  background: var(--card);
  position: relative;
}
.resize-row {
  position: absolute;
  top: -3px;
  left: 0;
  right: 0;
  height: 6px;
  cursor: row-resize;
  z-index: 10;
  transition: background 0.15s;
}
.resize-row:hover { background: var(--primary); }

/* 终端过渡动画 */
.term-slide-enter-active,
.term-slide-leave-active {
  transition: transform 0.2s ease;
}
.term-slide-enter-from,
.term-slide-leave-to {
  transform: translateY(100%);
}

/* ── 底部状态栏 ── */
.status-bar {
  height: 28px;
  flex: none;
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 14px;
  gap: 16px;
  font-size: 0.74rem;
  color: var(--muted-foreground);
  background: var(--card);
}
.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--chart-2);
  animation: pulse 2s infinite;
}
.status-spacer { flex: 1; }
.status-dirty { color: var(--chart-4); }
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

/* ── 右栏属性面板 ── */
.prop-group { margin-bottom: 14px; }
.prop-group-title {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--muted-foreground);
  letter-spacing: 0.04em;
  text-transform: uppercase;
  margin-bottom: 6px;
  padding: 0 4px;
}
.prop-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 5px 8px;
  border-radius: 6px;
  font-size: 0.82rem;
}
.prop-row:hover { background: var(--muted); }
.prop-label { color: var(--muted-foreground); }
.prop-value { font-weight: 500; font-variant-numeric: tabular-nums; }
.prop-truncate {
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-align: right;
}
.prop-empty {
  text-align: center;
  padding: 20px 12px;
  color: var(--muted-foreground);
  font-size: 0.8rem;
}

/* ── 检查器 ── */
.inspector-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 6px;
  font-size: 0.81rem;
  cursor: pointer;
  transition: background 120ms ease;
}
.inspector-item:hover { background: var(--muted); }
.inspector-dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.inspector-dot.ok { background: var(--chart-2); }
.inspector-dot.warn { background: var(--chart-4); }
.inspector-dot.info { background: var(--chart-1); }
.inspector-text { flex: 1; min-width: 0; }
.inspector-meta { font-size: 0.72rem; color: var(--muted-foreground); }

/* ── 快捷按钮 ── */
.quick-btn {
  display: flex;
  align-items: center;
  gap: 7px;
  width: 100%;
  padding: 7px 10px;
  border-radius: 6px;
  border: 0;
  background: transparent;
  color: var(--foreground);
  font-size: 0.82rem;
  cursor: pointer;
  text-align: left;
  transition: background 120ms ease;
  font-family: inherit;
}
.quick-btn:hover { background: var(--muted); }
.quick-btn .icon-sm {
  width: 15px;
  height: 15px;
  color: var(--muted-foreground);
}

/* ── 对话框 ── */
.dlg-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10000;
  backdrop-filter: blur(4px);
  -webkit-backdrop-filter: blur(4px);
}
.dlg-box {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  min-width: 380px;
  max-width: 90vw;
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.4);
}
.dlg-box h3 {
  margin: 0 0 12px;
  font-size: 15px;
  color: var(--foreground);
}
.dlg-input {
  width: 100%;
  padding: 9px 12px;
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) * 0.6);
  background: var(--background);
  color: var(--foreground);
  font-family: inherit;
  font-size: 13px;
  outline: none;
  margin-bottom: 12px;
  transition: border-color 0.2s;
  box-sizing: border-box;
}
.dlg-input:focus { border-color: var(--primary); }
.dlg-actions { display: flex; gap: 8px; justify-content: flex-end; }
.dlg-btn {
  padding: 7px 16px;
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) * 0.6);
  background: transparent;
  color: var(--muted-foreground);
  font-size: 13px;
  cursor: pointer;
  font-family: inherit;
  transition: background 150ms ease, color 150ms ease;
}
.dlg-btn:hover { background: var(--muted); color: var(--foreground); }
.dlg-btn.primary {
  border: 0;
  background: var(--primary);
  color: var(--primary-foreground);
}
.dlg-btn.primary:hover {
  background: color-mix(in srgb, var(--primary) 90%, var(--foreground));
}
.dlg-error {
  color: var(--destructive, #ef4444);
  font-size: 12px;
  margin: 8px 0 0;
}

.quick-paths {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-bottom: 12px;
}
.qp-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  font-size: 12px;
  color: var(--muted-foreground);
  background: var(--muted);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) * 0.5);
  cursor: pointer;
  transition: background 150ms ease, color 150ms ease, border-color 150ms ease;
  font-family: inherit;
}
.qp-btn:hover {
  background: color-mix(in srgb, var(--primary) 12%, var(--background));
  color: var(--primary);
  border-color: color-mix(in srgb, var(--primary) 30%, var(--border));
}

/* ── 右键菜单 ── */
.ctx-menu {
  position: fixed;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 4px;
  z-index: 9999;
  min-width: 140px;
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
}
.ctx-item {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 6px 12px;
  font-size: 12px;
  color: var(--foreground);
  cursor: pointer;
  border-radius: 8px;
  transition: background 0.1s;
}
.ctx-item:hover { background: var(--muted); }
.ctx-item.danger:hover {
  background: rgba(239, 68, 68, 0.1);
  color: var(--destructive, #ef4444);
}
.ctx-sep {
  height: 1px;
  background: var(--border);
  margin: 3px 8px;
}

/* 响应式 */
@media (max-width: 880px) {
  .ide-layout { grid-template-columns: 1fr !important; }
  .panel-left, .panel-right { display: none; }
}
</style>
