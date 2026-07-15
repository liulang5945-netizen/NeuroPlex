<template>
  <div class="monaco-wrapper">
    <div class="monaco-toolbar">
      <div class="monaco-tabs">
        <button
          v-for="tab in openTabs"
          :key="tab.path"
          :class="['monaco-tab', { active: tab.path === activeTab }]"
          @click="switchTab(tab.path)"
        >
          <span class="tab-icon">{{ getFileIcon(tab.path) }}</span>
          <span class="tab-name">{{ tab.name }}</span>
          <span class="tab-close" @click.stop="closeTab(tab.path)">×</span>
        </button>
      </div>
      <div class="monaco-actions">
        <select v-model="language" class="lang-select" @change="updateLanguage" :title="languageLabel">
          <option v-for="lang in languages" :key="lang.id" :value="lang.id">{{ lang.label }}</option>
        </select>
        <button class="btn-action" @click="saveFile" title="保存"><Save :size="14" /></button>
      </div>
    </div>
    <div class="monaco-loading" v-if="editorLoading">
      <div class="loading-spinner"></div>
      <span>编辑器加载中...</span>
    </div>
    <div class="monaco-error" v-if="editorError">
      <span class="error-icon">⚠️</span>
      <span>{{ editorError }}</span>
      <button class="btn-action" @click="initMonaco" title="重试">重试</button>
    </div>
    <div ref="editorContainer" class="monaco-editor-container" v-show="!editorError && !editorFallback"></div>
    <!-- Fallback: 简单文本编辑器 -->
    <div v-if="editorFallback" class="fallback-editor">
      <textarea
        ref="fallbackTextarea"
        v-model="fallbackContent"
        class="fallback-textarea"
        :placeholder="'Monaco Editor 加载失败，已切换到简易编辑器。\n您可以在此编辑文件内容，按 Ctrl+S 保存。'"
        spellcheck="false"
        @input="onFallbackInput"
      ></textarea>
    </div>
    <div class="monaco-statusbar">
      <span>{{ language }}</span>
      <span>行 {{ cursorLine }}, 列 {{ cursorCol }}</span>
      <span v-if="isDirty" class="dirty-indicator">● 未保存</span>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue';
import { Save } from 'lucide-vue-next';
import { useApi } from '../composables/useApi.js';
import { API_BASE, authFetch } from '../composables/apiClient.js';
import { useAppStore } from '../stores/appStore.js';

const { t } = useApi();
const appStore = useAppStore();
const props = defineProps({ workspacePath: { type: String, default: '' } });
const emit = defineEmits(['saved']);

const editorContainer = ref(null);
const editorLoading = ref(true);
const editorError = ref('');
const editorFallback = ref(false);
const fallbackTextarea = ref(null);
const fallbackContent = ref('');
let editor = null;
let monaco = null;

const activeTab = ref('');
const openTabs = ref([]);
const isDirty = ref(false);
const cursorLine = ref(1);
const cursorCol = ref(1);
const language = ref('plaintext');

const languages = [
  { id: 'python', label: 'Python' },
  { id: 'javascript', label: 'JavaScript' },
  { id: 'typescript', label: 'TypeScript' },
  { id: 'html', label: 'HTML' },
  { id: 'css', label: 'CSS' },
  { id: 'json', label: 'JSON' },
  { id: 'java', label: 'Java' },
  { id: 'go', label: 'Go' },
  { id: 'rust', label: 'Rust' },
  { id: 'cpp', label: 'C++' },
  { id: 'csharp', label: 'C#' },
  { id: 'sql', label: 'SQL' },
  { id: 'markdown', label: 'Markdown' },
  { id: 'yaml', label: 'YAML' },
  { id: 'xml', label: 'XML' },
  { id: 'shell', label: 'Shell' },
  { id: 'plaintext', label: '纯文本' },
];

const languageLabel = computed(() => languages.find(item => item.id === language.value)?.label || language.value);

const extToLang = {
  '.py': 'python', '.js': 'javascript', '.ts': 'typescript', '.jsx': 'javascript',
  '.tsx': 'typescript', '.html': 'html', '.htm': 'html', '.css': 'css',
  '.json': 'json', '.java': 'java', '.go': 'go', '.rs': 'rust',
  '.c': 'c', '.cpp': 'cpp', '.h': 'c', '.hpp': 'cpp', '.cs': 'csharp',
  '.sql': 'sql', '.md': 'markdown', '.yml': 'yaml', '.yaml': 'yaml',
  '.xml': 'xml', '.sh': 'shell', '.bat': 'shell', '.vue': 'html',
};

const fileIcons = {
  py: '🐍', js: '📜', ts: '📘', json: '📋', html: '🌐', css: '🎨',
  java: '☕', go: '🐹', rs: '🦀', cpp: '⚙️', md: '📝', vue: '💚',
};

function getFileIcon(path) {
  const ext = path.split('.').pop()?.toLowerCase();
  return fileIcons[ext] || '📄';
}

function detectLanguage(filename) {
  const ext = '.' + filename.split('.').pop()?.toLowerCase();
  return extToLang[ext] || 'plaintext';
}

async function initMonaco() {
  editorLoading.value = true;
  editorError.value = '';

  try {
    // 确保容器已挂载且有尺寸
    if (!editorContainer.value) {
      editorError.value = '编辑器容器未就绪';
      editorLoading.value = false;
      return;
    }

    const loader = await import('@monaco-editor/loader');
    monaco = await loader.default.init();

    const container = editorContainer.value;
    // 确保容器有明确的尺寸（防止 flex 子元素高度为 0）
    if (container.offsetHeight < 100) {
      container.style.height = '100%';
      container.style.minHeight = '300px';
    }

    editor = monaco.editor.create(container, {
      value: '',
      language: 'plaintext',
      theme: isDarkMode() ? 'vs-dark' : 'vs',
      automaticLayout: true,
      fontSize: 14,
      fontFamily: "'Cascadia Code', 'Fira Code', 'JetBrains Mono', Consolas, monospace",
      lineNumbers: 'on',
      minimap: { enabled: true },
      scrollBeyondLastLine: false,
      wordWrap: 'on',
      tabSize: 4,
      renderWhitespace: 'selection',
      bracketPairColorization: { enabled: true },
      suggest: { showKeywords: true, showSnippets: true },
      quickSuggestions: true,
      formatOnPaste: true,
      // 关键：防止大面积空白
      folding: true,
      padding: { top: 8, bottom: 8 },
    });

    editor.onDidChangeCursorPosition((e) => {
      cursorLine.value = e.position.lineNumber;
      cursorCol.value = e.position.column;
    });

    editor.onDidChangeModelContent(() => {
      isDirty.value = true;
    });

    // Ctrl+S 保存
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
      saveFile();
    });

    // 强制重新布局，修复初始化时尺寸计算错误
    requestAnimationFrame(() => {
      if (editor) editor.layout();
    });

    editorLoading.value = false;
    editorError.value = '';
  } catch (err) {
    console.error('Monaco Editor 初始化失败，切换到简易编辑器:', err);
    editorError.value = '';
    editorLoading.value = false;
    editorFallback.value = true;
  }
}

function onFallbackInput() {
  isDirty.value = true;
}

async function openFile(filePath) {
  const existing = openTabs.value.find(t => t.path === filePath);
  if (existing) {
    switchTab(filePath);
    return;
  }

  try {
    const r = await authFetch(`${API_BASE}/api/workspace/file?name=${encodeURIComponent(filePath)}`);
    if (r.ok) {
      const data = await r.json();
      const name = filePath.split('/').pop() || filePath.split('\\').pop();
      const lang = detectLanguage(name);

      openTabs.value.push({ path: filePath, name, content: data.content || '', language: lang });

      if (!activeTab.value) {
        switchTab(filePath);
      }
    }
  } catch (e) {
    console.error('打开文件失败:', e);
  }
}

function switchTab(filePath) {
  // 保存当前标签内容
  if (activeTab.value) {
    if (editor) {
      const oldTab = openTabs.value.find(t => t.path === activeTab.value);
      if (oldTab) oldTab.content = editor.getValue();
    } else if (editorFallback.value) {
      const oldTab = openTabs.value.find(t => t.path === activeTab.value);
      if (oldTab) oldTab.content = fallbackContent.value;
    }
  }

  activeTab.value = filePath;
  const tab = openTabs.value.find(t => t.path === filePath);
  if (tab) {
    if (editor) {
      editor.setValue(tab.content);
      language.value = tab.language;
      const model = editor.getModel();
      if (monaco && model) {
        monaco.editor.setModelLanguage(model, tab.language);
      }
    } else if (editorFallback.value) {
      fallbackContent.value = tab.content;
      language.value = tab.language;
    }
    isDirty.value = false;
  }
}

function closeTab(filePath) {
  const idx = openTabs.value.findIndex(t => t.path === filePath);
  if (idx === -1) return;
  openTabs.value.splice(idx, 1);

  if (activeTab.value === filePath) {
    if (openTabs.value.length > 0) {
      const next = openTabs.value[Math.min(idx, openTabs.value.length - 1)];
      switchTab(next.path);
    } else {
      activeTab.value = '';
      if (editor) editor.setValue('');
    }
  }
}

function updateLanguage() {
  if (!editor || !monaco) return;
  const model = editor.getModel();
  if (model) monaco.editor.setModelLanguage(model, language.value);
  const tab = openTabs.value.find(t => t.path === activeTab.value);
  if (tab) tab.language = language.value;
}

async function saveFile() {
  if (!activeTab.value) return;
  const content = editor ? editor.getValue() : fallbackContent.value;
  if (!content && content !== '') return;
  try {
    const r = await authFetch(`${API_BASE}/api/workspace/file`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: activeTab.value, content }),
    });
    if (r.ok) {
      isDirty.value = false;
      const tab = openTabs.value.find(t => t.path === activeTab.value);
      if (tab) tab.content = content;
      emit('saved', activeTab.value);
    }
  } catch (e) {
    console.error('保存失败:', e);
  }
}

function setTheme(isDark) {
  if (monaco && editor) {
    monaco.editor.setTheme(isDark ? 'vs-dark' : 'vs');
  }
}

function isDarkMode() {
  return document.documentElement.classList.contains('theme-dark') ||
    (!document.documentElement.classList.contains('theme-light') &&
     window.matchMedia('(prefers-color-scheme: dark)').matches)
}

watch(() => appStore.currentTheme, () => {
  if (monaco && editor) monaco.editor.setTheme(isDarkMode() ? 'vs-dark' : 'vs')
})

onMounted(async () => {
  // 等待容器有实际尺寸后再初始化 Monaco（防止刷新后布局偏移）
  await nextTick();
  const tryInit = () => {
    if (editorContainer.value && editorContainer.value.offsetWidth > 0 && editorContainer.value.offsetHeight > 0) {
      initMonaco();
    } else {
      // 容器还没有尺寸，用 ResizeObserver 等待
      const observer = new ResizeObserver((entries) => {
        for (const entry of entries) {
          if (entry.contentRect.width > 0 && entry.contentRect.height > 0) {
            observer.disconnect();
            initMonaco();
            return;
          }
        }
      });
      if (editorContainer.value) {
        observer.observe(editorContainer.value);
      }
      // 安全兜底：2秒后无论如何尝试初始化
      setTimeout(() => {
        observer.disconnect();
        if (!editor) initMonaco();
      }, 2000);
    }
  };
  requestAnimationFrame(() => requestAnimationFrame(tryInit));
});

onBeforeUnmount(() => {
  if (editor) editor.dispose();
});

defineExpose({ openFile, saveFile, setTheme });
</script>

<style scoped>
.monaco-wrapper {
  display: flex; flex-direction: column;
  height: 100%; min-height: 0;
  background: var(--editor-bg, var(--bg));
  overflow: hidden;
}

/* 工具栏 */
.monaco-toolbar {
  display: flex; align-items: center; justify-content: space-between;
  background: var(--toolbar-bg, var(--bg-surface));
  border-bottom: 1px solid var(--border);
  padding: 0 10px;
  height: 36px; flex-shrink: 0;
}
.monaco-tabs {
  display: flex; flex: 1; min-width: 0;
  overflow-x: auto; overflow-y: hidden; gap: 1px;
  scrollbar-width: none;
}
.monaco-tabs::-webkit-scrollbar { display: none; }
.monaco-tab {
  display: flex; align-items: center; gap: 4px;
  min-width: 0; max-width: 180px; height: 34px;
  padding: 0 10px; background: transparent; border: none;
  color: var(--text-muted); cursor: pointer;
  font-size: 12px; white-space: nowrap;
  border-bottom: 2px solid transparent;
  transition: all 0.12s ease;
  flex-shrink: 0;
}
.monaco-tab:hover { background: var(--bg-hover); color: var(--text-secondary); }
.monaco-tab.active {
  color: var(--text); background: var(--bg-elevated);
  border-bottom-color: var(--primary); font-weight: 500;
}
.tab-icon { font-size: 12px; flex-shrink: 0; }
.tab-name { overflow: hidden; text-overflow: ellipsis; }
.tab-close {
  font-size: 14px; margin-left: 4px; opacity: 0;
  width: 16px; height: 16px; display: flex; align-items: center; justify-content: center;
  border-radius: 3px; transition: all 0.1s ease; flex-shrink: 0;
}
.monaco-tab:hover .tab-close { opacity: 0.4; }
.tab-close:hover { opacity: 1 !important; background: rgba(239,68,68,0.1); color: var(--danger); }

/* 右侧操作区 */
.monaco-actions {
  display: flex; align-items: center; gap: 6px;
  flex-shrink: 0; margin-left: 8px;
}
.lang-select {
  width: 100px; height: 26px;
  background: var(--bg-elevated); color: var(--text-secondary);
  border: 1px solid var(--border); padding: 0 6px;
  border-radius: var(--radius-sm);
  font-size: 11px; outline: none;
}
.lang-select:focus { border-color: var(--primary); }
.btn-action {
  width: 26px; height: 26px;
  display: inline-flex; align-items: center; justify-content: center;
  background: transparent; border: 1px solid transparent;
  cursor: pointer; padding: 0;
  border-radius: var(--radius-sm);
  transition: all 0.12s ease; color: var(--text-secondary);
}
.btn-action:hover { background: var(--bg-hover); border-color: var(--border); }

/* 编辑器容器 */
.monaco-editor-container {
  flex: 1; min-height: 0; height: 100%;
}

/* 状态栏 */
.monaco-statusbar {
  display: flex; gap: 16px; padding: 2px 12px;
  background: var(--toolbar-bg, var(--bg-surface));
  border-top: 1px solid var(--border);
  font-size: 11px; color: var(--text-muted);
  height: 22px; align-items: center; flex-shrink: 0;
}
.dirty-indicator { color: var(--warning); }

/* 加载/错误 */
.monaco-loading {
  display: flex; align-items: center; justify-content: center; gap: 8px;
  flex: 1; color: var(--text-muted); font-size: 13px; padding: 40px;
}
.loading-spinner {
  width: 16px; height: 16px;
  border: 2px solid var(--border); border-top-color: var(--primary);
  border-radius: 50%; animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.monaco-error {
  display: flex; align-items: center; justify-content: center; gap: 8px;
  padding: 12px; color: var(--danger); font-size: 13px;
  background: var(--danger-light); border-bottom: 1px solid var(--border);
}
.error-icon { font-size: 14px; }

/* Fallback 编辑器 */
.fallback-editor { flex: 1; display: flex; min-height: 200px; }
.fallback-textarea {
  flex: 1; width: 100%; padding: 12px 16px;
  border: none; outline: none; resize: none;
  font-family: var(--font-mono); font-size: 14px; line-height: 1.6; tab-size: 4;
  background: var(--editor-bg, var(--bg)); color: var(--text);
}
.fallback-textarea::placeholder { color: var(--text-muted); opacity: 0.5; }

@media (max-width: 720px) {
  .monaco-toolbar { padding: 0 6px; }
  .lang-select { width: 80px; }
  .monaco-tab { max-width: 120px; }
}
</style>
