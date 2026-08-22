<template>
  <section class="dedicated-view">
    <div class="kb-page">
      <!-- 页面标题 -->
      <div class="page-header">
        <h1>知识库管理</h1>
        <p class="subtitle">管理Seed的领域知识源，为检索增强与共振推理提供语料支撑。</p>
      </div>

      <!-- 标签页 -->
      <div class="tabs" role="tablist">
        <button class="tab" :class="{ active: activeTab === 'files' }" role="tab" :aria-selected="activeTab === 'files'" @click="activeTab = 'files'">
          <svg class="tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/></svg>
          文件管理
        </button>
        <button class="tab" :class="{ active: activeTab === 'config' }" role="tab" :aria-selected="activeTab === 'config'" @click="activeTab = 'config'">
          <svg class="tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
          检索配置
        </button>
        <button class="tab" :class="{ active: activeTab === 'test' }" role="tab" :aria-selected="activeTab === 'test'" @click="activeTab = 'test'">
          <svg class="tab-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
          检索测试
        </button>
      </div>

      <!-- ══ Tab 1：文件管理 ══ -->
      <div v-if="activeTab === 'files'" class="tab-panel">
        <!-- 工具栏 -->
        <div class="toolbar">
          <div class="kb-search-wrap">
            <div class="search" style="height: 38px;">
              <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
              <input v-model="fileFilter" placeholder="搜索文件名或路径..." aria-label="搜索文件">
            </div>
          </div>
          <button class="btn btn-primary" @click="kbUploadRef?.triggerBrowse?.()">
            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>
            上传文件
          </button>
          <button class="btn btn-outline">
            <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M7 12h10M11 18h2"/></svg>
            批量操作
          </button>
          <span class="toolbar-meta">共 {{ kbFiles.length }} 个文件 · 已索引 {{ kbStats?.doc_count ?? kbFiles.length }}</span>
        </div>

        <!-- 文件表格 -->
        <div class="panel kb-panel">
          <div class="head">
            <h2>知识库文件</h2>
            <span class="sub">· 默认知识库</span>
            <span class="spacer"></span>
            <button class="btn btn-ghost btn-sm">
              <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5h16l-6 7v5l-4 2v-7L4 5Z"/></svg>
              筛选
            </button>
          </div>
          <table v-if="filteredFiles(kbFiles).length">
            <thead>
              <tr>
                <th style="width:36px"><input class="cb" type="checkbox" aria-label="全选"></th>
                <th>文件名</th>
                <th>类型</th>
                <th>大小</th>
                <th>状态</th>
                <th>更新时间</th>
                <th style="width:96px"></th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="f in filteredFiles(kbFiles)" :key="f">
                <td><input class="cb" type="checkbox" :aria-label="String(f)"></td>
                <td>
                  <span class="fname">
                    <span class="file-ic" :class="fileBadge(f)">{{ (fileExt(f) || 'file').slice(0, 4).toUpperCase() }}</span>
                    <span class="meta-stack">
                      <span class="name">{{ f }}</span>
                      <span class="path">—</span>
                    </span>
                  </span>
                </td>
                <td><span class="type-tag">{{ fileTypeName(f) }}</span></td>
                <td class="size-cell">—</td>
                <td><span class="status-chip ok">已索引</span></td>
                <td class="time-cell">—</td>
                <td>
                  <span class="row-actions">
                    <button class="act" aria-label="预览" title="预览">
                      <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
                    </button>
                    <button class="act" aria-label="删除" title="删除文件" @click="deleteKBFile(f)">
                      <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></svg>
                    </button>
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
          <div v-else class="empty-hint">知识库为空，上传文件开始构建知识库</div>
        </div>
      </div>

      <!-- ══ Tab 2：检索配置 ══ -->
      <div v-if="activeTab === 'config'" class="tab-panel">
        <div class="panel kb-panel config-panel">
          <div class="head">
            <h2>检索参数</h2>
            <span class="sub">· 影响召回与排序质量</span>
          </div>
          <div class="config-body">
            <div class="config-form">
              <!-- Top-K 召回数量 -->
              <div class="form-row">
                <div class="row-head">
                  <label>Top-K 召回数量</label>
                  <span class="val">{{ ragConfig.candidate_k }}</span>
                </div>
                <input type="range" class="slider" min="1" max="20" step="1" v-model.number="ragConfig.candidate_k" @change="saveRagConfig">
                <div class="slider-marks"><span>1</span><span>10</span><span>20</span></div>
                <span class="hint">每次检索返回的相关片段数量，值越大召回越广但耗时增加。</span>
              </div>

              <!-- 相似度阈值（静态展示，script 中无对应变量） -->
              <div class="form-row">
                <div class="row-head">
                  <label>相似度阈值</label>
                  <span class="val">0.70</span>
                </div>
                <input type="range" class="slider" min="0" max="1" step="0.01" value="0.7" disabled>
                <div class="slider-marks"><span>0.0</span><span>0.5</span><span>1.0</span></div>
                <span class="hint">低于该分数的片段将被过滤，提高阈值可减少噪声但可能漏召回。</span>
              </div>

              <!-- 重排序开关 -->
              <div class="switch-row">
                <div class="switch-label">
                  <label>启用重排序</label>
                  <span class="hint">对召回结果进行二次精排，提升前列结果准确率。</span>
                </div>
                <input type="checkbox" class="toggle" v-model="ragConfig.enable_reranker" @change="saveRagConfig">
              </div>

              <!-- Embedding 模型（静态展示） -->
              <div class="form-row">
                <div class="row-head">
                  <label>Embedding 模型</label>
                </div>
                <select class="select" disabled>
                  <option value="doubao-emb-v2" selected>Seed Embedding v2（默认 · 1024 维）</option>
                  <option value="doubao-emb-large">Seed Embedding Large（1536 维）</option>
                  <option value="bge-m3">BGE-M3（多语言）</option>
                  <option value="m3e-base">M3E-Base（轻量）</option>
                </select>
                <span class="hint">切换模型后需对全部文件重新索引，预计耗时数分钟。</span>
              </div>

              <div class="form-actions">
                <button class="btn btn-primary" @click="saveRagConfig">保存配置</button>
                <button class="btn btn-outline" @click="Object.assign(ragConfig, { enable_hybrid: true, enable_reranker: true, enable_query_rewrite: false, candidate_k: 20 }); saveRagConfig()">恢复默认</button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- ══ Tab 3：检索测试 ══ -->
      <div v-if="activeTab === 'test'" class="tab-panel">
        <div class="test-bar">
          <div class="kb-search-wrap">
            <div class="search" style="height: 40px;">
              <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
              <input v-model="kbSearchQuery" placeholder="输入查询内容进行检索测试..." aria-label="检索查询" @keydown.enter="searchKB">
            </div>
          </div>
          <button class="btn btn-primary" :disabled="kbSearching" @click="searchKB">
            <svg class="icon" :class="{ spin: kbSearching }" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
            检索
          </button>
        </div>

        <div v-if="kbSearched" class="test-meta">
          <span>Top-K = {{ ragConfig.candidate_k }}</span>
          <span class="dot"></span>
          <span>相似度阈值 0.70</span>
          <span class="dot"></span>
          <span>命中 {{ kbResults.length }} 条</span>
        </div>

        <div v-if="kbResults.length" class="result-list">
          <div v-for="(r, i) in kbResults" :key="i" class="result-item">
            <div class="result-head">
              <span v-if="r.score != null" class="result-score">{{ Number(r.score).toFixed(2) }}</span>
              <span class="result-name">{{ r.source || r.filename || r.title || ('片段 #' + (i + 1)) }}</span>
              <span v-if="r.path" class="result-source">{{ r.path }}</span>
            </div>
            <p class="result-snippet">{{ r.content || r.text || r }}</p>
          </div>
        </div>
        <div v-else-if="kbSearched" class="empty-hint">未找到相关结果，尝试更换关键词</div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { ref } from 'vue';
import { BookOpen, Upload, BarChart3, Search, FolderOpen, Settings as SettingsIcon, Trash2, FileText } from 'lucide-vue-next';
import FileUploadQueue from '../components/FileUploadQueue.vue';
import { useApi } from '../composables/useApi.js';
import { API_BASE, authFetch } from '../composables/apiClient.js';
const { t } = useApi();
const kbUploadRef = ref(null);
const kbStats = ref(null);
const kbSearchQuery = ref('');
const kbResults = ref([]);
const kbSearched = ref(false);
const kbSearching = ref(false);
const kbFiles = ref([]);
const ragConfig = ref({ enable_hybrid: true, enable_reranker: true, enable_query_rewrite: false, candidate_k: 20 });
const ragStatus = ref(null);
const loadRagConfig = async () => { try { const r = await authFetch(`${API_BASE}/api/rag/config`); if (r.ok) { const d = await r.json(); if (d.config) ragConfig.value = { ...ragConfig.value, ...d.config }; } } catch (e) {} };
const loadRagStatus = async () => { try { const r = await authFetch(`${API_BASE}/api/rag/status`); if (r.ok) { const d = await r.json(); if (d.status === 'ok') ragStatus.value = d; } } catch (e) {} };
const saveRagConfig = async () => { try { await authFetch(`${API_BASE}/api/rag/config`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(ragConfig.value) }); } catch (e) {} };
loadRagConfig(); loadRagStatus();
const loadKBStats = async () => { try { const r = await authFetch(`${API_BASE}/api/rag/stats`); if (r.ok) kbStats.value = await r.json(); } catch (e) {} };
const loadKBFiles = async () => { try { const r = await authFetch(`${API_BASE}/api/rag/files`); if (r.ok) { const d = await r.json(); kbFiles.value = d.files || []; } } catch (e) {} };
const searchKB = async () => { if (!kbSearchQuery.value.trim()) return; kbSearched.value = true; kbSearching.value = true; try { const r = await authFetch(`${API_BASE}/api/rag/search`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: kbSearchQuery.value, top_k: 5 }) }); const d = await r.json(); kbResults.value = d.results || []; } catch (e) { kbResults.value = []; } finally { kbSearching.value = false; } };
const clearKB = async () => { try { await authFetch(`${API_BASE}/api/rag/clear`, { method: 'DELETE' }); kbStats.value = null; kbFiles.value = []; } catch (e) {} };
const deleteKBFile = async (filename) => { try { await authFetch(`${API_BASE}/api/rag/file/${encodeURIComponent(filename)}`, { method: 'DELETE' }); loadKBFiles(); loadKBStats(); loadRagStatus(); } catch (e) {} };
loadKBStats(); loadKBFiles();

// ── 标签页与文件过滤（纯 UI 状态，不影响业务逻辑） ──
const activeTab = ref('files');
const fileFilter = ref('');
const fileExt = (name) => { const m = String(name).match(/\.(\w+)$/); return m ? m[1].toLowerCase() : ''; };
const fileBadge = (name) => { const e = fileExt(name); if (e === 'pdf') return 'pdf'; if (['md', 'markdown'].includes(e)) return 'md'; if (['json', 'jsonl'].includes(e)) return 'json'; if (e === 'txt') return 'txt'; return 'doc'; };
const fileTypeName = (name) => { const e = fileExt(name); const m = { pdf: 'PDF', md: 'Markdown', markdown: 'Markdown', json: 'JSON', jsonl: 'JSON', txt: 'TXT', csv: 'CSV', html: 'HTML', htm: 'HTML', docx: 'Word', doc: 'Word', pptx: 'PPT', xlsx: 'Excel', xls: 'Excel', py: 'Python', js: 'JavaScript', ts: 'TypeScript', yaml: 'YAML', yml: 'YAML' }; return m[e] || (e ? e.toUpperCase() : 'FILE'); };
const filteredFiles = (files) => { if (!fileFilter.value) return files; const q = fileFilter.value.toLowerCase(); return files.filter(f => String(f).toLowerCase().includes(q)); };
</script>

<style scoped>
/* ===== 内容区 ===== */
.dedicated-view { padding: 0; }

.kb-page {
  max-width: 800px;
  margin: 0 auto;
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 0;
}

/* ===== 页头 ===== */
.page-header { margin-bottom: 22px; }
.page-header h1 {
  margin: 0 0 6px;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--foreground);
}
.page-header .subtitle {
  margin: 0;
  color: var(--muted-foreground);
  font-size: 0.9rem;
  line-height: 1.5;
}

/* ===== 标签页 ===== */
.tabs {
  display: flex;
  gap: 4px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 22px;
}
.tab {
  border: 0;
  background: transparent;
  color: var(--muted-foreground);
  padding: 10px 18px;
  font-size: 0.9rem;
  font-weight: 500;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  transition: color .15s ease, border-color .15s ease;
  display: inline-flex;
  align-items: center;
  gap: 7px;
}
.tab:hover { color: var(--foreground); }
.tab.active {
  color: var(--primary);
  border-bottom-color: var(--primary);
  font-weight: 600;
}
.tab .tab-icon {
  width: 15px;
  height: 15px;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
  flex: none;
}

.tab-panel { animation: kb-fade-in .22s ease; }
@keyframes kb-fade-in {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

/* ===== 工具栏 ===== */
.toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}
.kb-search-wrap { flex: 1; min-width: 220px; }
.toolbar-meta {
  font-size: 0.78rem;
  color: var(--muted-foreground);
}

/* ===== 搜索框 ===== */
.search {
  display: flex;
  align-items: center;
  gap: 9px;
  height: 40px;
  padding: 0 12px;
  border: 1px solid transparent;
  border-radius: 10px;
  background: var(--muted);
  color: var(--foreground);
  transition: background .16s ease, border-color .16s ease, box-shadow .16s ease;
}
.search:focus-within {
  background: var(--card);
  border-color: var(--ring);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 18%, transparent);
}
.search input {
  flex: 1;
  min-width: 0;
  border: 0;
  outline: none;
  background: transparent;
  font-size: 0.86rem;
  color: inherit;
}
.search input::placeholder { color: var(--muted-foreground); }
.search .icon {
  width: 17px;
  height: 17px;
  flex: none;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-linejoin: round;
  color: var(--muted-foreground);
}

/* ===== 按钮 ===== */
.btn {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 36px;
  padding: 0 15px;
  border-radius: 999px;
  border: 1px solid transparent;
  font-size: 0.86rem;
  font-weight: 500;
  cursor: pointer;
  transition: background 150ms ease, border-color 150ms ease, transform 120ms ease, color 150ms ease;
}
.btn:active { transform: translateY(1px); }
.btn:focus-visible { outline: 2px solid var(--ring); outline-offset: 2px; }
.btn-primary {
  background: var(--primary);
  color: var(--primary-foreground);
}
.btn-primary:hover {
  background: color-mix(in srgb, var(--primary) 90%, var(--foreground));
}
.btn-outline {
  background: var(--background);
  color: var(--foreground);
  border-color: var(--border);
}
.btn-outline:hover { background: var(--muted); }
.btn-ghost {
  background: var(--muted);
  color: var(--foreground);
}
.btn-ghost:hover {
  background: color-mix(in srgb, var(--muted) 80%, var(--foreground) 12%);
}
.btn-sm {
  height: 32px;
  padding: 0 13px;
  font-size: 0.8rem;
}
.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.btn .icon {
  width: 15px;
  height: 15px;
  flex: none;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}

/* ===== 面板 / 表格 ===== */
.panel {
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--card);
  overflow: hidden;
}
.panel.kb-panel { max-width: none; }
.panel .head {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 16px;
  border-bottom: 1px solid var(--border);
}
.panel .head h2 {
  margin: 0;
  font-size: 15px;
  font-weight: 600;
  color: var(--foreground);
}
.panel .head .sub {
  margin-left: 2px;
  font-size: 12px;
  color: var(--muted-foreground);
}
.spacer { flex: 1; }

table {
  width: 100%;
  border-collapse: collapse;
}
th {
  text-align: left;
  font: 600 11px/1 var(--font-mono);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted-foreground);
  padding: 11px 16px;
  border-bottom: 1px solid var(--border);
}
td {
  padding: 13px 16px;
  border-bottom: 1px solid color-mix(in srgb, var(--border) 60%, transparent);
  font-size: 13px;
  color: var(--foreground);
}
tr:last-child td { border-bottom: 0; }
tbody tr { transition: background .12s ease; }
tbody tr:hover { background: color-mix(in srgb, var(--accent) 16%, transparent); }

/* 文件类型徽标 */
.file-ic {
  width: 30px;
  height: 30px;
  flex: none;
  border-radius: 8px;
  display: grid;
  place-items: center;
  font: 600 0.64rem var(--font-mono);
  letter-spacing: 0.04em;
  color: var(--primary-foreground);
}
.file-ic.pdf { background: color-mix(in srgb, var(--destructive) 78%, var(--foreground)); }
.file-ic.md { background: var(--chart-1); }
.file-ic.txt { background: color-mix(in srgb, var(--muted-foreground) 88%, var(--foreground)); }
.file-ic.json { background: var(--chart-3); }
.file-ic.doc { background: var(--muted-foreground); }

.fname { display: flex; align-items: center; gap: 11px; }
.fname .meta-stack { display: flex; flex-direction: column; min-width: 0; }
.fname .name {
  font-size: 0.86rem;
  font-weight: 500;
  line-height: 1.3;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 260px;
}
.fname .path {
  font-size: 0.7rem;
  color: var(--muted-foreground);
  margin-top: 1px;
}

.type-tag {
  display: inline-block;
  padding: 2px 9px;
  border-radius: 6px;
  background: var(--muted);
  color: var(--muted-foreground);
  font: 600 0.68rem var(--font-mono);
  letter-spacing: 0.03em;
}
.size-cell { font-variant-numeric: tabular-nums; }
.time-cell {
  color: var(--muted-foreground);
  font-size: 0.8rem;
}

/* 状态 chip */
.status-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px;
  border-radius: 999px;
  font: 500 12px var(--font-sans);
}
.status-chip::before {
  content: "";
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
}
.status-chip.ok {
  background: color-mix(in srgb, var(--chart-2) 18%, transparent);
  color: var(--chart-2);
}
.status-chip.run {
  background: color-mix(in srgb, var(--chart-1) 16%, transparent);
  color: var(--chart-1);
}
.status-chip.run::before {
  animation: pulse-dot 1.4s ease-in-out infinite;
}
.status-chip.fail {
  background: color-mix(in srgb, var(--destructive) 16%, transparent);
  color: var(--destructive);
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.35; transform: scale(0.6); }
}

/* 行操作按钮 */
.row-actions { display: inline-flex; gap: 2px; }
.act {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: var(--muted-foreground);
  cursor: pointer;
  transition: background .14s ease, color .14s ease;
}
.act:hover {
  background: var(--muted);
  color: var(--foreground);
}
.act .icon {
  width: 15px;
  height: 15px;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.8;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.cb {
  width: 16px;
  height: 16px;
  accent-color: var(--primary);
  cursor: pointer;
}

/* ===== 检索配置 ===== */
.config-panel { max-width: 640px; }
.config-body {
  padding: 22px 18px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.config-form {
  display: flex;
  flex-direction: column;
  gap: 24px;
  max-width: 580px;
}
.form-row {
  display: flex;
  flex-direction: column;
  gap: 9px;
}
.row-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
}
.row-head label {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--foreground);
}
.row-head .val {
  font: 600 0.85rem var(--font-mono);
  color: var(--primary);
}
.hint {
  font-size: 0.76rem;
  color: var(--muted-foreground);
  line-height: 1.5;
}

/* 滑块 */
.slider {
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 6px;
  border-radius: 999px;
  background: var(--muted);
  outline: none;
  cursor: pointer;
}
.slider:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--primary);
  border: 3px solid var(--background);
  box-shadow: 0 0 0 1px var(--border);
  cursor: pointer;
  transition: box-shadow .15s ease;
}
.slider::-webkit-slider-thumb:hover {
  box-shadow: 0 0 0 1px var(--primary), 0 0 0 5px color-mix(in srgb, var(--ring) 22%, transparent);
}
.slider::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--primary);
  border: 3px solid var(--background);
  cursor: pointer;
}
.slider-marks {
  display: flex;
  justify-content: space-between;
  font-size: 0.7rem;
  color: var(--muted-foreground);
  margin-top: 2px;
}

/* 下拉选择 */
.select {
  height: 38px;
  padding: 0 32px 0 12px;
  border-radius: calc(var(--radius) * 0.5);
  border: 1px solid var(--border);
  background: var(--background);
  color: var(--foreground);
  font-size: 0.86rem;
  cursor: pointer;
  appearance: none;
  -webkit-appearance: none;
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%237f8d9f' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'><path d='m6 9 6 6 6-6'/></svg>");
  background-repeat: no-repeat;
  background-position: right 10px center;
}
.select:focus {
  outline: none;
  border-color: var(--ring);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 18%, transparent);
}
.select:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* 开关 */
.switch-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}
.switch-label {
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-width: 0;
}
.switch-label label {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--foreground);
}
.toggle {
  appearance: none;
  -webkit-appearance: none;
  position: relative;
  width: 40px;
  height: 22px;
  border-radius: 999px;
  background: var(--muted);
  border: 1px solid var(--border);
  cursor: pointer;
  flex: none;
  margin: 0;
  transition: background .18s ease, border-color .18s ease;
}
.toggle::after {
  content: "";
  position: absolute;
  top: 1px;
  left: 1px;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--background);
  border: 1px solid var(--border);
  transition: transform .18s ease, background .18s ease, border-color .18s ease;
}
.toggle:checked {
  background: var(--primary);
  border-color: var(--primary);
}
.toggle:checked::after {
  transform: translateX(18px);
  background: var(--primary-foreground);
  border-color: var(--primary);
}

.form-actions {
  display: flex;
  gap: 10px;
  padding-top: 6px;
}

/* ===== 检索测试 ===== */
.test-bar {
  display: flex;
  gap: 10px;
  margin-bottom: 18px;
  align-items: center;
}
.test-bar .kb-search-wrap { flex: 1; }
.test-meta {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
  font-size: 0.78rem;
  color: var(--muted-foreground);
}
.test-meta .dot {
  width: 3px;
  height: 3px;
  border-radius: 50%;
  background: currentColor;
  opacity: 0.6;
}

.result-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.result-item {
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) * 0.6);
  padding: 14px 16px;
  background: var(--card);
  transition: border-color .15s ease, background .15s ease;
}
.result-item:hover {
  border-color: color-mix(in srgb, var(--primary) 38%, var(--border));
  background: color-mix(in srgb, var(--accent) 16%, var(--card));
}
.result-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 9px;
  flex-wrap: wrap;
}
.result-score {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 9px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--chart-2) 16%, transparent);
  color: var(--chart-2);
  font: 600 0.74rem var(--font-mono);
}
.result-score::before {
  content: "";
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: currentColor;
}
.result-name {
  font-weight: 600;
  font-size: 0.88rem;
  color: var(--foreground);
}
.result-source {
  font-size: 0.72rem;
  color: var(--muted-foreground);
}
.result-snippet {
  margin: 0;
  color: var(--muted-foreground);
  font-size: 0.84rem;
  line-height: 1.65;
  border-left: 2px solid var(--border);
  padding: 2px 0 2px 11px;
}

/* ===== 空状态 ===== */
.empty-hint {
  text-align: center;
  padding: 40px 16px;
  color: var(--muted-foreground);
  font-size: 0.86rem;
}

/* ===== 加载旋转 ===== */
.spin {
  animation: kb-spin 0.8s linear infinite;
}
@keyframes kb-spin {
  to { transform: rotate(360deg); }
}

/* ===== 响应式 ===== */
@media (max-width: 880px) {
  .kb-page { max-width: 100%; padding: 16px; }
  .tab { padding: 10px 12px; }
  .toolbar { gap: 8px; }
}
</style>
