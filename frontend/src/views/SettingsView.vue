<template>
  <section class="dedicated-view settings-view">
    <!-- 顶栏：居中标题 + 副标题 -->
    <header class="settings-topbar">
      <span style="width:0"></span>
      <div class="topbar-title-wrap">
        <span class="topbar-title">设置</span>
        <span class="topbar-sub">Seed系统配置与偏好</span>
      </div>
      <span class="topbar-spacer"></span>
    </header>

    <!-- 滚动内容区 -->
    <div class="settings-scroll">
      <div class="settings-wrap">
        <div class="settings-layout">

          <!-- ═══ 左侧：设置导航 ═══ -->
          <nav class="settings-nav" aria-label="设置导航">
            <button
              class="sn-item"
              :class="{ active: activeSection === 'general' }"
              @click="activeSection = 'general'"
            >
              <svg class="sn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 3v2.5M12 18.5V21M21 12h-2.5M5.5 12H3M18.5 5.5l-1.8 1.8M7.3 16.7l-1.8 1.8M18.5 18.5l-1.8-1.8M7.3 7.3 5.5 5.5"/></svg>
              通用设置
            </button>
            <button
              class="sn-item"
              :class="{ active: activeSection === 'neuron' }"
              @click="activeSection = 'neuron'"
            >
              <svg class="sn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="7" r="2.2"/><circle cx="6" cy="17" r="2.2"/><circle cx="18" cy="12" r="2.2"/><circle cx="12" cy="4" r="1.5"/><circle cx="12" cy="20" r="1.5"/><path d="M8 7.5 12 5M8 16.5 12 19M8 16l7-4M8 8l7 3"/></svg>
              神经元设置
            </button>
            <button
              class="sn-item"
              :class="{ active: activeSection === 'privacy' }"
              @click="activeSection = 'privacy'"
            >
              <svg class="sn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
              数据与隐私
            </button>
            <button
              class="sn-item"
              :class="{ active: activeSection === 'about' }"
              @click="activeSection = 'about'"
            >
              <svg class="sn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/></svg>
              关于
            </button>
          </nav>

          <!-- ═══ 右侧：设置内容 ═══ -->
          <div class="settings-content">

            <!-- ── 1. 通用设置 ── -->
            <section v-if="activeSection === 'general'" class="settings-section">
              <h2>通用设置</h2>

              <!-- 外观主题 -->
              <div class="setting-row setting-row--col setting-row--first">
                <span class="setting-label">外观主题</span>
                <div class="theme-previews">
                  <div
                    v-for="th in appStore.themes"
                    :key="th.id"
                    class="theme-preview-card"
                    :class="{ active: appStore.currentTheme === th.id }"
                    :title="th.desc"
                    @click="appStore.setTheme(th.id)"
                  >
                    <span class="theme-swatch" :style="{ background: th.gradient }"></span>
                    <span class="theme-name">{{ th.name }}</span>
                  </div>
                </div>
              </div>

              <!-- 默认语言 -->
              <div class="setting-row">
                <div class="setting-left">
                  <span class="setting-label">默认语言</span>
                  <p class="setting-desc">界面与交互的显示语言</p>
                </div>
                <div class="setting-right">
                  <select aria-label="默认语言">
                    <option value="zh-CN" selected>简体中文</option>
                    <option value="zh-TW">繁體中文</option>
                    <option value="en">English</option>
                    <option value="ja">日本語</option>
                    <option value="ko">한국어</option>
                  </select>
                </div>
              </div>

              <!-- 时区 -->
              <div class="setting-row">
                <div class="setting-left">
                  <span class="setting-label">时区</span>
                  <p class="setting-desc">用于定时任务、日志时间戳等</p>
                </div>
                <div class="setting-right">
                  <select aria-label="时区">
                    <option value="Asia/Shanghai" selected>Asia/Shanghai (UTC+8)</option>
                    <option value="Asia/Tokyo">Asia/Tokyo (UTC+9)</option>
                    <option value="Asia/Seoul">Asia/Seoul (UTC+9)</option>
                    <option value="Asia/Singapore">Asia/Singapore (UTC+8)</option>
                    <option value="America/Los_Angeles">America/Los_Angeles (UTC-8)</option>
                    <option value="America/New_York">America/New_York (UTC-5)</option>
                    <option value="Europe/London">Europe/London (UTC+0)</option>
                    <option value="Europe/Berlin">Europe/Berlin (UTC+1)</option>
                  </select>
                </div>
              </div>

              <!-- 界面密度 -->
              <div class="setting-row setting-row--last">
                <div class="setting-left">
                  <span class="setting-label">界面密度</span>
                  <p class="setting-desc">调整元素间距与信息密度</p>
                </div>
                <div class="setting-right">
                  <div class="radio-group" role="radiogroup" aria-label="界面密度">
                    <label class="radio-chip">
                      <input type="radio" name="density" value="compact">
                      <span class="rc-label">紧凑</span>
                    </label>
                    <label class="radio-chip">
                      <input type="radio" name="density" value="default" checked>
                      <span class="rc-label">默认</span>
                    </label>
                    <label class="radio-chip">
                      <input type="radio" name="density" value="comfortable">
                      <span class="rc-label">宽松</span>
                    </label>
                  </div>
                </div>
              </div>
            </section>

            <!-- ── 2. 神经元设置 ── -->
            <section v-else-if="activeSection === 'neuron'" class="settings-section">
              <h2>神经元设置</h2>

              <!-- 共振阈值 -->
              <div class="setting-row setting-row--first">
                <div class="setting-left">
                  <span class="setting-label">共振阈值</span>
                  <p class="setting-desc">神经元激活的同步判定阈值（越高越严谨）</p>
                </div>
                <div class="setting-right">
                  <div class="range-wrap">
                    <input type="range" min="0" max="1" step="0.01" value="0.72" aria-label="共振阈值" />
                    <span class="range-value">0.72</span>
                  </div>
                </div>
              </div>

              <!-- 响应超时 -->
              <div class="setting-row">
                <div class="setting-left">
                  <span class="setting-label">响应超时</span>
                  <p class="setting-desc">神经元链路最大等待时间（毫秒）</p>
                </div>
                <div class="setting-right">
                  <input type="number" min="10" max="10000" value="100" aria-label="响应超时" />
                </div>
              </div>

              <!-- 自动巩固 -->
              <div class="setting-row">
                <div class="setting-left">
                  <span class="setting-label">自动巩固</span>
                  <p class="setting-desc">高频突触自动强化并写入持久记忆</p>
                </div>
                <div class="setting-right">
                  <label class="toggle" aria-label="自动巩固开关">
                    <input type="checkbox" checked />
                    <span class="track"><span class="thumb"></span></span>
                  </label>
                </div>
              </div>

              <!-- 睡眠模式 -->
              <div class="setting-row setting-row--last">
                <div class="setting-left">
                  <span class="setting-label">睡眠模式</span>
                  <p class="setting-desc">非活跃时段暂停后台共振计算以节省算力，每日首次唤醒需约 2 秒预热</p>
                </div>
                <div class="setting-right">
                  <label class="toggle" aria-label="睡眠模式开关">
                    <input type="checkbox" />
                    <span class="track"><span class="thumb"></span></span>
                  </label>
                </div>
              </div>
            </section>

            <!-- ── 3. 数据与隐私 ── -->
            <section v-else-if="activeSection === 'privacy'" class="settings-section">
              <h2>数据与隐私</h2>

              <!-- 对话保留 -->
              <div class="setting-row setting-row--first">
                <div class="setting-left">
                  <span class="setting-label">对话保留</span>
                  <p class="setting-desc">历史对话的自动保留时长</p>
                </div>
                <div class="setting-right">
                  <select aria-label="对话保留">
                    <option value="30">30 天</option>
                    <option value="90" selected>90 天</option>
                    <option value="180">180 天</option>
                    <option value="365">365 天</option>
                    <option value="forever">永久保留</option>
                  </select>
                </div>
              </div>

              <!-- 自动清理 -->
              <div class="setting-row">
                <div class="setting-left">
                  <span class="setting-label">自动清理</span>
                  <p class="setting-desc">过期对话与缓存文件到达保留期限后自动移除</p>
                </div>
                <div class="setting-right">
                  <label class="toggle" aria-label="自动清理开关">
                    <input type="checkbox" checked />
                    <span class="track"><span class="thumb"></span></span>
                  </label>
                </div>
              </div>

              <!-- 导出数据 -->
              <div class="setting-row">
                <div class="setting-left">
                  <span class="setting-label">导出数据</span>
                  <p class="setting-desc">导出所有对话记录、配置快照与神经元状态</p>
                </div>
                <div class="setting-right">
                  <button class="btn-sm btn-outline" @click="toast('占位功能：导出数据尚未接入', 'info')">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/></svg>
                    导出
                  </button>
                </div>
              </div>

              <!-- 危险区 -->
              <div class="setting-row setting-row--last">
                <div class="danger-zone">
                  <h3>
                    <svg class="dz-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"/><path d="M12 9v4M12 17h.01"/></svg>
                    危险操作
                  </h3>
                  <p>重置将清除所有本地配置、神经元状态与缓存文件，此操作不可撤销。建议先导出数据再进行重置。</p>
                  <button class="btn-destructive" @click="toast('占位功能：重置尚未接入', 'warning')">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M9.5 9.5h5v6h-5z"/><path d="M14.5 9.5 13 7h-2l-1.5 2.5"/></svg>
                    重置Seed
                  </button>
                </div>
              </div>
            </section>

            <!-- ── 4. 关于 ── -->
            <section v-else-if="activeSection === 'about'" class="settings-section">
              <h2>关于</h2>

              <!-- 版本元信息 -->
              <div class="setting-row setting-row--first">
                <div class="about-meta">
                  <div class="meta-line">
                    <span style="font-weight:600;">Seed神经元系统</span>
                    <span class="meta-tag">v{{ appVersion }}</span>
                  </div>
                  <div class="meta-line meta-line--muted">
                    共振场引擎
                    <span class="meta-tag">ResonanceField v3.0</span>
                  </div>
                  <div class="meta-line meta-line--muted">
                    渲染后端
                    <span class="meta-tag">WASM+GPU</span>
                  </div>
                </div>
              </div>

              <!-- 开源许可 -->
              <div class="setting-row setting-row--last">
                <div class="setting-left">
                  <span class="setting-label">开源许可</span>
                  <p class="setting-desc">查看本系统使用的第三方组件许可协议</p>
                </div>
                <div class="setting-right">
                  <button class="btn-sm btn-ghost" @click="toast('占位功能：许可详情尚未接入', 'info')">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/><path d="M12 18v-6"/><path d="M9 15h6"/></svg>
                    查看许可
                  </button>
                </div>
              </div>
            </section>

          </div><!-- /.settings-content -->
        </div><!-- /.settings-layout -->
      </div><!-- /.settings-wrap -->
    </div><!-- /.settings-scroll -->
  </section>
</template>

<script setup>
import { ref, inject } from 'vue';
import { Settings as SettingsIcon, Palette, MessageSquareText, RefreshCw, Image as ImageIcon } from 'lucide-vue-next';
import { useAppStore } from '../stores/appStore.js';
import { API_BASE, authFetch } from '../composables/apiClient.js';

const toast = inject('toast');
const appStore = useAppStore();
const t = (key, params) => appStore.t(key, params);

// 左侧设置导航当前激活分区（v-if 切换右侧内容）
const activeSection = ref('general');

const defaultPrompt = '你是一个全能助手。请直接、简洁地回答问题。如果遇到错误或不知道的情况，请直接说明，无需冗长地道歉。';
const systemPrompt = ref(localStorage.getItem('taiji_system_prompt') || defaultPrompt);
const appVersion = ref('1.0.0');
const updateChecking = ref(false);
const updateAvailable = ref(false);
const updateMsg = ref('');

const onBgImageSelect = (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    appStore.setBgImage(ev.target.result);
    toast('✅ 背景图已设置', 'success');
  };
  reader.readAsDataURL(file);
};

const saveSettings = async () => {
  localStorage.setItem('taiji_system_prompt', systemPrompt.value);
  await authFetch(`${API_BASE}/api/settings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      theme: appStore.currentTheme,
      accent: appStore.currentAccent,
      system_prompt: systemPrompt.value,
    }),
  });
  toast('✅ 设置已保存', 'success');
};

const checkUpdate = async () => {
  updateChecking.value = true; updateMsg.value = '';
  try {
    const r = await authFetch(`${API_BASE}/api/system/check_update`, { method: 'POST' });
    const d = await r.json();
    updateAvailable.value = d.has_update;
    updateMsg.value = d.has_update ? `${t('update_available')} v${d.version}` : (d.message || '已是最新版本');
  } catch (e) { updateMsg.value = `❌ ${e.message}`; }
  finally { updateChecking.value = false; }
};

const applyUpdate = async () => {
  try { await authFetch(`${API_BASE}/api/system/apply_update`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) }); } catch (e) {}
};

// Load app version
(async () => {
  try {
    const r = await authFetch(`${API_BASE}/api/system/version`);
    if (r.ok) {
      const d = await r.json();
      if (d.version) appVersion.value = d.version;
    }
  } catch (e) {}
})();
</script>

<style scoped>
/* ═══ Seed控制台 · 设置页专属样式（对齐画布 settings.html · 豆包设计 token）═══ */

/* --- 视图外壳：覆盖 dedicated-view 的内边距，改为顶栏 + 滚动区 --- */
.settings-view {
  padding: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* --- 顶栏 --- */
.settings-topbar {
  flex: none;
  height: 52px;
  padding: 0 18px;
  display: flex;
  align-items: center;
  gap: 12px;
  border-bottom: 1px solid var(--border);
  background: var(--background);
}
.topbar-title-wrap {
  display: flex;
  flex-direction: column;
  gap: 0;
  line-height: 1.2;
}
.topbar-title {
  font-size: 0.94rem;
  font-weight: 650;
  color: var(--foreground);
}
.topbar-sub {
  font-size: 0.72rem;
  color: var(--muted-foreground);
  margin-top: 1px;
}
.topbar-spacer {
  flex: 1;
}

/* --- 滚动区 + 容器 --- */
.settings-scroll {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}
.settings-wrap {
  width: 100%;
  max-width: 760px;
  margin: 0 auto;
  padding: 28px 24px 48px;
}

/* --- 双栏布局：左侧导航 + 右侧内容 --- */
.settings-layout {
  display: grid;
  grid-template-columns: 168px minmax(0, 1fr);
  gap: 32px;
  align-items: start;
}

/* --- 左侧 sticky 导航 --- */
.settings-nav {
  position: sticky;
  top: 16px;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding-top: 6px;
}
.sn-item {
  width: 100%;
  border: 0;
  background: transparent;
  color: var(--muted-foreground);
  border-radius: calc(var(--radius) * 0.45);
  padding: 9px 14px;
  text-align: left;
  font-size: 0.88rem;
  font-weight: 450;
  line-height: 1.35;
  cursor: pointer;
  transition: background 150ms ease, color 150ms ease, font-weight 150ms ease;
  display: flex;
  align-items: center;
  gap: 9px;
}
.sn-item:hover {
  background: var(--muted);
  color: var(--foreground);
}
.sn-item.active {
  background: color-mix(in srgb, var(--primary) 12%, transparent);
  color: var(--primary);
  font-weight: 600;
}
.sn-icon {
  width: 17px;
  height: 17px;
  flex: none;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
  opacity: 0.7;
}
.sn-item.active .sn-icon {
  opacity: 1;
}

/* --- 右侧内容区 --- */
.settings-content {
  min-width: 0;
}
.settings-section {
  display: flex;
  flex-direction: column;
}
.settings-section h2 {
  font-size: 1.15rem;
  font-weight: 650;
  margin: 0 0 18px;
  color: var(--foreground);
  letter-spacing: -0.01em;
}

/* --- 设置行 --- */
.setting-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 0;
  border-bottom: 1px solid var(--border);
  gap: 16px;
}
.setting-row--first {
  padding-top: 0;
}
.setting-row--last {
  border-bottom: none;
  padding-bottom: 0;
}
.setting-row--col {
  flex-direction: column;
  align-items: stretch;
  gap: 12px;
}
.setting-left {
  min-width: 0;
  flex: 1;
}
.setting-right {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}
.setting-label {
  font-size: 0.88rem;
  font-weight: 500;
  color: var(--foreground);
  line-height: 1.3;
}
.setting-desc {
  font-size: 0.76rem;
  color: var(--muted-foreground);
  margin: 3px 0 0;
  line-height: 1.4;
}

/* --- 表单控件基础 --- */
.setting-right select,
.setting-right input[type="number"],
.setting-right input[type="text"] {
  background: var(--background);
  color: var(--foreground);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) * 0.45);
  padding: 7px 12px;
  font-size: 0.86rem;
  font-family: var(--font-sans);
  outline: none;
  transition: border-color 160ms ease, box-shadow 160ms ease;
}
.setting-right select:focus,
.setting-right input[type="number"]:focus,
.setting-right input[type="text"]:focus {
  border-color: var(--ring);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 16%, transparent);
}
.setting-right select {
  padding-right: 32px;
  appearance: none;
  -webkit-appearance: none;
  color: var(--muted-foreground);
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%237f8d9f' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 10px center;
  background-size: 15px;
  cursor: pointer;
}
.setting-right input[type="number"] {
  width: 90px;
  text-align: center;
  font-variant-numeric: tabular-nums;
}

/* --- Radio 组（界面密度）--- */
.radio-group {
  display: flex;
  gap: 4px;
  background: var(--muted);
  border-radius: calc(var(--radius) * 0.5);
  padding: 3px;
}
.radio-chip {
  position: relative;
  cursor: pointer;
}
.radio-chip input {
  position: absolute;
  opacity: 0;
  width: 0;
  height: 0;
}
.radio-chip .rc-label {
  display: block;
  padding: 5px 13px;
  border-radius: calc(var(--radius) * 0.4);
  font-size: 0.8rem;
  font-weight: 450;
  color: var(--muted-foreground);
  transition: background 150ms ease, color 150ms ease, font-weight 150ms ease;
  user-select: none;
  white-space: nowrap;
}
.radio-chip input:checked + .rc-label {
  background: var(--background);
  color: var(--foreground);
  font-weight: 600;
  box-shadow: 0 1px 2px color-mix(in srgb, var(--foreground) 8%, transparent);
}
.radio-chip:hover .rc-label {
  color: var(--foreground);
}

/* --- Range 滑块 --- */
.range-wrap {
  display: flex;
  align-items: center;
  gap: 10px;
}
.range-value {
  font-size: 0.82rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--primary);
  min-width: 42px;
  text-align: right;
}
.setting-right input[type="range"] {
  -webkit-appearance: none;
  appearance: none;
  width: 140px;
  height: 6px;
  background: var(--border);
  border-radius: 999px;
  border: 0;
  padding: 0;
  outline: none;
  cursor: pointer;
}
.setting-right input[type="range"]:focus {
  box-shadow: none;
}
.setting-right input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--primary);
  cursor: pointer;
  border: 2px solid var(--background);
  box-shadow: 0 1px 4px color-mix(in srgb, var(--foreground) 18%, transparent);
  transition: transform 120ms ease;
}
.setting-right input[type="range"]::-webkit-slider-thumb:hover {
  transform: scale(1.15);
}
.setting-right input[type="range"]::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--primary);
  cursor: pointer;
  border: 2px solid var(--background);
  box-shadow: 0 1px 4px color-mix(in srgb, var(--foreground) 18%, transparent);
}

/* --- Toggle 开关 --- */
.toggle {
  position: relative;
  display: inline-block;
  width: 42px;
  height: 24px;
  flex-shrink: 0;
  cursor: pointer;
}
.toggle input {
  opacity: 0;
  width: 0;
  height: 0;
  position: absolute;
  border: 0;
  padding: 0;
}
.toggle .track {
  position: absolute;
  inset: 0;
  background: var(--border);
  border-radius: 999px;
  cursor: pointer;
  transition: background 200ms ease;
}
.toggle .thumb {
  position: absolute;
  top: 3px;
  left: 3px;
  width: 18px;
  height: 18px;
  background: var(--background);
  border-radius: 50%;
  transition: transform 200ms cubic-bezier(0.34, 1.56, 0.64, 1);
  box-shadow: 0 1px 3px color-mix(in srgb, var(--foreground) 16%, transparent);
}
.toggle input:checked + .track {
  background: var(--primary);
}
.toggle input:checked + .track .thumb {
  transform: translateX(18px);
}
.toggle input:focus-visible + .track {
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 20%, transparent);
}

/* --- 主题预览网格 --- */
.theme-previews {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 10px;
}
.theme-preview-card {
  border: 2px solid var(--border);
  border-radius: 12px;
  padding: 8px 8px 10px;
  cursor: pointer;
  background: var(--background);
  text-align: center;
  transition: border-color 160ms ease, box-shadow 160ms ease, transform 140ms ease;
}
.theme-preview-card:hover {
  border-color: color-mix(in srgb, var(--primary) 35%, var(--border));
  transform: translateY(-1px);
}
.theme-preview-card.active {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--ring) 18%, transparent);
}
.theme-swatch {
  width: 100%;
  height: 36px;
  border-radius: 8px;
  display: block;
}
.theme-name {
  font-size: 0.72rem;
  margin-top: 7px;
  color: var(--muted-foreground);
  font-weight: 450;
  transition: color 150ms ease;
}
.theme-preview-card.active .theme-name {
  color: var(--foreground);
  font-weight: 600;
}

/* --- 小号按钮（导出 / 查看许可）--- */
.btn-sm {
  height: 32px;
  padding: 0 13px;
  font-size: 0.8rem;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid transparent;
  cursor: pointer;
  font-weight: 500;
  transition: background 150ms ease, border-color 150ms ease, transform 120ms ease;
}
.btn-sm:active {
  transform: translateY(1px);
}
.btn-sm:focus-visible {
  outline: 2px solid var(--ring);
  outline-offset: 2px;
}
.btn-sm.btn-outline {
  background: var(--background);
  color: var(--foreground);
  border-color: var(--border);
}
.btn-sm.btn-outline:hover {
  background: var(--muted);
}
.btn-sm.btn-ghost {
  background: var(--muted);
  color: var(--foreground);
}
.btn-sm.btn-ghost:hover {
  background: color-mix(in srgb, var(--muted) 80%, var(--foreground) 14%);
}

/* --- 危险区 --- */
.danger-zone {
  width: 100%;
  border: 1px solid var(--destructive);
  border-radius: 13px;
  padding: 20px;
  margin-top: 8px;
  background: color-mix(in srgb, var(--destructive) 4%, transparent);
}
.danger-zone h3 {
  color: var(--destructive);
  margin: 0 0 6px;
  font-size: 0.94rem;
  font-weight: 650;
  display: flex;
  align-items: center;
  gap: 7px;
}
.danger-zone .dz-icon {
  width: 18px;
  height: 18px;
  flex: none;
}
.danger-zone p {
  font-size: 0.78rem;
  color: var(--muted-foreground);
  margin: 0 0 14px;
  line-height: 1.5;
}
.btn-destructive {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  height: 36px;
  padding: 0 16px;
  border: 1px solid transparent;
  border-radius: 999px;
  background: var(--destructive);
  color: var(--destructive-foreground);
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  transition: background 150ms ease, transform 120ms ease;
}
.btn-destructive:hover {
  background: color-mix(in srgb, var(--destructive) 88%, var(--foreground) 12%);
}
.btn-destructive:active {
  transform: translateY(1px);
}
.btn-destructive:focus-visible {
  outline: 2px solid var(--ring);
  outline-offset: 2px;
}

/* --- 关于区块元信息 --- */
.about-meta {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 6px 0;
}
.about-meta .meta-line {
  font-size: 0.86rem;
  color: var(--foreground);
  display: flex;
  align-items: center;
  gap: 6px;
}
.about-meta .meta-line--muted {
  color: var(--muted-foreground);
  font-size: 0.8rem;
}
.about-meta .meta-tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 6px;
  font-size: 0.72rem;
  font-weight: 600;
  background: var(--muted);
  color: var(--muted-foreground);
  font-family: var(--font-mono);
}

/* --- 响应式 --- */
@media (max-width: 880px) {
  .settings-layout {
    grid-template-columns: 1fr;
    gap: 0;
  }
  .settings-nav {
    position: static;
    flex-direction: row;
    flex-wrap: wrap;
    gap: 4px;
    padding: 0 0 16px;
    margin-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }
  .sn-item {
    width: auto;
    padding: 7px 12px;
    font-size: 0.8rem;
  }
  .theme-previews {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (max-width: 560px) {
  .theme-previews {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 500px) {
  .settings-wrap {
    padding: 16px 12px 32px;
  }
  .setting-row:not(.setting-row--col) {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
  .setting-right {
    width: 100%;
    justify-content: flex-start;
  }
}
</style>
