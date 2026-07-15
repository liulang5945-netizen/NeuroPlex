<template>
  <div class="ide">
    <!-- 工具栏 -->
    <div class="ide-toolbar">
      <div class="toolbar-left">
        <span class="toolbar-title">工作台</span>
        <span class="toolbar-path" v-if="workspacePath">{{ workspacePath }}</span>
        <button class="tb-btn" @click="openFolderPicker" title="切换路径"><FolderOpen :size="13" /></button>
      </div>
      <div class="toolbar-right">
        <button class="tb-btn" @click="refreshTree" title="刷新"><RefreshCw :size="13" /></button>
        <button class="tb-btn" @click="createNewFile" title="新建文件"><FilePlus :size="13" /></button>
        <button class="tb-btn" @click="createNewFolder" title="新建文件夹"><FolderPlus :size="13" /></button>
        <button class="tb-btn" :class="{ on: showTerminal }" @click="showTerminal = !showTerminal" title="终端"><Terminal :size="13" /></button>
      </div>
    </div>

    <!-- 主体 -->
    <div class="ide-body">
      <!-- 文件树 -->
      <div class="ide-sidebar" :style="{ width: sidebarWidth + 'px' }">
        <div class="sidebar-head">
          <FolderOpen :size="13" />
          <span>文件</span>
        </div>
        <div class="sidebar-tree">
          <div v-if="!fileTree.length" class="tree-empty">空工作台</div>
          <div v-for="node in fileTree" :key="node.path"
            class="tree-item" :class="{ dir: node.type === 'directory' }"
            :style="{ paddingLeft: (node.depth * 14 + 8) + 'px' }"
            @click="handleTreeClick(node)"
            @contextmenu.prevent="showContextMenu($event, node)">
            <component :is="node.type === 'directory' ? (expandedDirs.has(node.path) ? FolderOpen : Folder) : getFileIcon(node.name)" :size="13" class="tree-ico" />
            <span class="tree-name">{{ node.name }}</span>
          </div>
        </div>
        <div class="resize-col" @mousedown="startResize"></div>
      </div>

      <!-- 编辑器 + 终端 右侧区域 -->
      <div class="ide-main">
        <!-- 编辑器 -->
        <div class="ide-editor">
          <MonacoEditor ref="monacoEditor" />
        </div>

        <!-- 终端 -->
        <Transition name="term-slide">
          <div v-if="showTerminal" class="ide-terminal" :style="{ height: terminalHeight + 'px' }">
            <div class="resize-row" @mousedown="startTerminalResize"></div>
            <WebTerminal ref="webTerminal" />
          </div>
        </Transition>
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
import { ref, reactive, onMounted, onUnmounted, inject } from 'vue';
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
.ide {
  display: flex;
  flex-direction: column;
  height: 100%;
  width: 100%;
  overflow: hidden;
  background: var(--bg);
}

/* ── 工具栏 ── */
.ide-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 40px;
  padding: 0 14px;
  background: var(--toolbar-bg);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.toolbar-left, .toolbar-right { display: flex; align-items: center; gap: 6px; }
.toolbar-left { color: var(--text-muted); min-width: 0; }
.toolbar-title { font-size: 13px; font-weight: 600; color: var(--text); }
.toolbar-path {
  font-size: 11px; color: var(--text-muted); font-family: var(--font-mono);
  max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  padding: 2px 8px; background: var(--bg-muted); border-radius: var(--radius-sm);
}
.tb-btn {
  width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;
  border: 0; border-radius: var(--radius-sm); background: transparent; color: var(--text-muted);
  cursor: pointer; transition: var(--transition-fast);
}
.tb-btn:hover { background: var(--bg-hover); color: var(--text); }
.tb-btn.on { background: var(--primary-subtle); color: var(--primary); }

/* ── 主体 flex 布局 ── */
.ide-body {
  flex: 1;
  display: flex;
  min-height: 0;
  overflow: hidden;
}

/* ── 文件树侧边栏 ── */
.ide-sidebar {
  display: flex;
  flex-direction: column;
  background: var(--sidebar-bg);
  border-right: 1px solid var(--border);
  position: relative;
  min-width: 120px;
  flex-shrink: 0;
}
.sidebar-head {
  display: flex; align-items: center; gap: 5px; padding: 8px 10px;
  font-size: 11px; font-weight: 600; color: var(--text-muted);
  border-bottom: 1px solid var(--border-subtle);
  text-transform: uppercase; letter-spacing: 0.04em;
}
.sidebar-tree {
  flex: 1;
  overflow-y: auto;
  padding: 4px 0;
}
.tree-empty { text-align: center; padding: 24px 12px; color: var(--text-muted); font-size: 12px; }
.tree-item {
  display: flex; align-items: center; gap: 5px; padding: 4px 10px; cursor: pointer;
  font-size: 12.5px; color: var(--text-secondary); white-space: nowrap;
  transition: background 0.1s; border-left: 2px solid transparent;
}
.tree-item:hover { background: var(--bg-hover); color: var(--text); }
.tree-item.dir { font-weight: 500; color: var(--text); }
.tree-ico { flex-shrink: 0; width: 14px; text-align: center; color: var(--text-muted); }
.tree-name { overflow: hidden; text-overflow: ellipsis; }
.resize-col {
  position: absolute; top: 0; right: -2px; width: 4px; height: 100%;
  cursor: col-resize; z-index: 10; transition: background 0.15s;
}
.resize-col:hover { background: var(--primary); }

/* ── 右侧主区域（编辑器 + 终端） ── */
.ide-main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  background: var(--editor-bg);
  position: relative;
  overflow: hidden;
}

/* ── 编辑器 ── */
.ide-editor {
  flex: 1;
  min-height: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.ide-editor :deep(.monaco-wrapper) {
  height: 100% !important;
  min-height: 0 !important;
}
.ide-editor :deep(.monaco-editor-container) {
  flex: 1 !important;
  min-height: 0 !important;
  height: auto !important;
}

/* ── 终端 ── */
.ide-terminal {
  flex-shrink: 0;
  border-top: 1px solid var(--border);
  background: var(--terminal-bg);
  position: relative;
}
.resize-row {
  position: absolute; top: -3px; left: 0; right: 0; height: 6px;
  cursor: row-resize; z-index: 10; transition: background 0.15s;
}
.resize-row:hover { background: var(--primary); }

/* 终端过渡动画 */
.term-slide-enter-active,
.term-slide-leave-active {
  transition: transform 0.2s var(--ease);
}
.term-slide-enter-from,
.term-slide-leave-to {
  transform: translateY(100%);
}

/* ── 对话框 ── */
.dlg-overlay {
  position: fixed; inset: 0; background: rgba(0,0,0,0.5);
  display: flex; align-items: center; justify-content: center;
  z-index: 10000; backdrop-filter: blur(4px); -webkit-backdrop-filter: blur(4px);
}
.dlg-box {
  background: var(--glass-bg-thick); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
  border: 1px solid var(--glass-border); border-radius: var(--radius-lg); padding: 24px;
  min-width: 380px; max-width: 90vw; box-shadow: 0 16px 48px rgba(0,0,0,0.4);
}
.dlg-box h3 { margin: 0 0 12px; font-size: 15px; color: var(--text); }
.dlg-input {
  width: 100%; padding: 9px 12px; border: 1px solid var(--border); border-radius: var(--radius-sm);
  background: var(--bg-input); color: var(--text); font-family: var(--font); font-size: 13px;
  outline: none; margin-bottom: 12px; transition: border-color 0.2s;
}
.dlg-input:focus { border-color: var(--primary); }
.dlg-actions { display: flex; gap: 8px; justify-content: flex-end; }
.dlg-btn {
  padding: 7px 16px; border: 1px solid var(--border); border-radius: var(--radius-sm);
  background: transparent; color: var(--text-secondary); font-size: 13px; cursor: pointer;
  font-family: var(--font); transition: var(--transition-fast);
}
.dlg-btn:hover { background: var(--bg-hover); color: var(--text); }
.dlg-btn.primary { border: 0; background: var(--primary-gradient); color: white; }
.dlg-btn.primary:hover { box-shadow: 0 2px 12px var(--border-strong); }
.dlg-error { color: var(--danger); font-size: 12px; margin: 8px 0 0; }

.quick-paths { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 12px; }
.qp-btn {
  display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px;
  font-size: 12px; color: var(--text-secondary); background: var(--bg-muted);
  border: 1px solid var(--border); border-radius: var(--radius-sm); cursor: pointer;
  transition: var(--transition-fast);
}
.qp-btn:hover { background: var(--primary-subtle); color: var(--primary); border-color: var(--primary-light); }

/* ── 右键菜单 ── */
.ctx-menu {
  position: fixed; background: var(--glass-bg); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--glass-border); border-radius: var(--radius-md); padding: 4px;
  z-index: 9999; min-width: 140px; box-shadow: 0 8px 28px rgba(0,0,0,0.35);
}
.ctx-item {
  display: flex; align-items: center; gap: 7px; padding: 6px 12px; font-size: 12px;
  color: var(--text); cursor: pointer; border-radius: 8px; transition: background 0.1s;
}
.ctx-item:hover { background: var(--bg-hover); }
.ctx-item.danger:hover { background: rgba(239,68,68,0.1); color: var(--danger); }
.ctx-sep { height: 1px; background: var(--border); margin: 3px 8px; }
</style>
