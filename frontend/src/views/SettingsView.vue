<template>
  <section class="dedicated-view">
    <div class="view-header"><h2>{{ t('sys_settings') }}</h2></div>
    <div class="view-body">
      <!-- UI Theme -->
      <div class="panel-section">
        <div class="panel-header"><h3><Palette :size="18" /> {{ t('ui_settings') }}</h3></div>
        <div class="panel-content params-grid">
          <label class="param-item"><span>{{ t('theme') }}</span>
            <n-select
              v-model:value="appStore.currentTheme"
              :options="[
                { label: t('theme_light'), value: 'light' },
                { label: t('theme_dark'), value: 'dark' },
                { label: t('theme_auto'), value: 'auto' },
              ]"
              size="small"
              @update:value="appStore.setTheme"
            />
          </label>
        </div>
        <!-- 主题色选择 -->
        <div class="panel-content">
          <label class="param-item"><span>主题色</span></label>
          <div class="theme-accent-row">
            <button
              v-for="preset in appStore.accentPresets"
              :key="preset.color"
              class="accent-swatch"
              :class="{ active: appStore.currentAccent === preset.color }"
              :style="{ background: preset.color }"
              :title="preset.name"
              @click="appStore.setAccent(preset.color)"
            ></button>
            <button
              class="accent-swatch accent-reset"
              :class="{ active: !appStore.currentAccent }"
              title="恢复默认"
              @click="appStore.setAccent('')"
            >×</button>
            <input type="color" class="accent-custom" :value="appStore.currentAccent || '#5b7a8a'" @input="appStore.setAccent($event.target.value)" title="自定义颜色" />
          </div>
        </div>
        <!-- 背景图设置 -->
        <div class="panel-content">
          <label class="param-item"><span>背景图片</span></label>
          <div class="bg-upload-row">
            <label class="file-upload-area">
              <ImageIcon :size="14" /> 选择图片
              <input type="file" accept="image/*" @change="onBgImageSelect" />
            </label>
            <button v-if="appStore.currentBgImage" class="btn-secondary" @click="appStore.setBgImage('')">清除背景</button>
          </div>
          <p v-if="appStore.currentBgImage" class="text-muted" style="margin:4px 0 0;font-size:0.78rem;">✅ 背景图已生效，侧边栏和面板会自动适应</p>
        </div>
      </div>
      <!-- System Prompt -->
      <div class="panel-section">
        <div class="panel-header"><h3><MessageSquareText :size="18" /> {{ t('system_prompt_settings') }}</h3></div>
        <div class="panel-content">
          <n-input v-model:value="systemPrompt" type="textarea" :rows="4" placeholder="系统提示词..." />
          <div style="display:flex;gap:8px;margin-top:8px;">
            <n-button type="primary" @click="saveSettings">{{ t('save') }}</n-button>
            <n-button @click="systemPrompt=defaultPrompt;saveSettings()">{{ t('reset_default') }}</n-button>
          </div>
        </div>
      </div>
      <!-- Update -->
      <div class="panel-section">
        <div class="panel-header"><h3><RefreshCw :size="18" /> {{ t('software_update') }}</h3></div>
        <div class="panel-content">
          <p class="text-muted" style="margin:0 0 8px;">{{ t('current_version') }}: {{ appVersion }}</p>
          <div style="display:flex;gap:8px;">
            <button class="primary-btn" @click="checkUpdate" :disabled="updateChecking">{{ updateChecking ? t('checking') : t('check_update') }}</button>
            <button v-if="updateAvailable" class="train-btn primary" @click="applyUpdate">{{ t('update_now') }}</button>
          </div>
          <p v-if="updateMsg" class="text-muted" style="margin-top:8px;">{{ updateMsg }}</p>
        </div>
      </div>
    </div>
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