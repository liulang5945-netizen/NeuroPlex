import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { locales } from '@/locales/index.js'
import { API_BASE, authFetch } from '@/composables/apiClient.js'

// 防抖保存 UI 设置到后端
let _uiSaveTimer = null
function _debouncedSaveUI(data) {
  if (_uiSaveTimer) clearTimeout(_uiSaveTimer)
  _uiSaveTimer = setTimeout(async () => {
    try {
      await authFetch(`${API_BASE}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })
    } catch (e) { /* silent fail */ }
  }, 1000)
}

// 主题值规范化：旧值 'light' → 'classic'，保留 'auto'，其余原样
function _normalizeTheme(raw) {
  if (!raw) return 'classic'
  if (raw === 'light') return 'classic'
  return raw
}

export const useAppStore = defineStore('app', () => {
  // === State ===
  // 5 套主题：classic(经典蓝) / dark(深邃暗色) / teal(自然青绿) / violet(科技紫调) / warm(暖橙活力)
  // 向后兼容：旧值 'light' 映射为 'classic'，'auto' 根据系统偏好选择
  const currentTheme = ref(_normalizeTheme(localStorage.getItem('taiji_theme') || 'classic'))
  const currentAccent = ref(localStorage.getItem('taiji_accent') || '')
  const currentBgImage = ref(localStorage.getItem('taiji_bg_image') || '')
  const currentLang = ref('zh')
  const showWorkspace = ref(false)

  // === 主题元数据（供设置页使用） ===
  const themes = [
    { id: 'classic', name: '经典蓝', desc: '豆包原色，明亮专业', gradient: 'linear-gradient(135deg, #0065fd, #0057da)' },
    { id: 'dark', name: '深邃暗色', desc: '深色护眼，专注沉浸', gradient: 'linear-gradient(135deg, #0f1419, #4d8cff)' },
    { id: 'teal', name: '自然青绿', desc: '清新自然，舒缓视觉', gradient: 'linear-gradient(135deg, #0d9488, #5eead4)' },
    { id: 'violet', name: '科技紫调', desc: '神秘紫调，科技质感', gradient: 'linear-gradient(135deg, #7c3aed, #c4b5fd)' },
    { id: 'warm', name: '暖橙活力', desc: '温暖橙红，活力充沛', gradient: 'linear-gradient(135deg, #ea580c, #fdba74)' },
  ]

  // === Getters ===

  // resolvedTheme: 返回 'light' 或 'dark'，供 Naive UI 使用（只有亮/暗两种基础主题）
  const resolvedTheme = computed(() => {
    const t = currentTheme.value
    if (t === 'auto') {
      if (typeof window !== 'undefined' && window.matchMedia) {
        return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
      }
      return 'dark'
    }
    // dark 主题为暗色，其余 4 套为亮色
    return t === 'dark' ? 'dark' : 'light'
  })

  // resolvedDataTheme: 返回实际 data-theme 属性值，供 CSS 变量使用
  const resolvedDataTheme = computed(() => {
    const t = currentTheme.value
    if (t === 'auto') {
      if (typeof window !== 'undefined' && window.matchMedia) {
        return window.matchMedia('(prefers-color-scheme: light)').matches ? 'classic' : 'dark'
      }
      return 'dark'
    }
    return t
  })

  // === Helpers ===
  function t(key, params = {}) {
    let text = locales[currentLang.value][key] || locales['zh'][key] || key
    for (const [k, v] of Object.entries(params)) {
      text = text.replace(`{${k}}`, v)
    }
    return text
  }

  // === Actions ===
  function toggleWorkspace() {
    showWorkspace.value = !showWorkspace.value
  }

  // 预设主题色 (水墨风格)
  const accentPresets = [
    { name: '墨黑', color: '#1a1a1a' },
    { name: '深灰', color: '#4a4a4a' },
    { name: '中灰', color: '#8a8a8a' },
    { name: '浅灰', color: '#b0b0b0' },
    { name: '朱砂', color: '#8a3a2a' },
    { name: '靛蓝', color: '#2a4a6a' },
    { name: '青瓷', color: '#4a6a5a' },
    { name: '琥珀', color: '#6a5a3a' },
  ]

  function applyTheme() {
    const r = document.documentElement
    r.classList.remove('theme-dark', 'theme-light')
    // 设置 data-theme 属性，驱动 themes.css 的 5 套主题变量
    const dt = resolvedDataTheme.value
    r.setAttribute('data-theme', dt)
    // 保留 theme-dark/theme-light class，兼容旧样式
    if (resolvedTheme.value === 'dark') {
      r.classList.add('theme-dark')
    } else {
      r.classList.add('theme-light')
    }
    applyAccent()
    applyBgImage()
  }

  function applyAccent() {
    const hex = currentAccent.value
    if (!hex) return
    const r = document.documentElement
    const rgb = hexToRgb(hex)
    if (!rgb) return
    r.style.setProperty('--primary', hex)
    r.style.setProperty('--primary-hover', darken(hex, 15))
    r.style.setProperty('--primary-light', `rgba(${rgb.r},${rgb.g},${rgb.b},0.08)`)
    r.style.setProperty('--primary-subtle', `rgba(${rgb.r},${rgb.g},${rgb.b},0.04)`)
    r.style.setProperty('--primary-gradient', `linear-gradient(135deg, ${hex} 0%, ${lighten(hex, 20)} 100%)`)
  }

  function applyBgImage() {
    const wrapper = document.querySelector('.app-wrapper')
    if (!wrapper) return
    if (currentBgImage.value) {
      wrapper.style.backgroundImage = `url(${currentBgImage.value})`
      wrapper.style.backgroundSize = 'cover'
      wrapper.style.backgroundPosition = 'center'
      wrapper.style.backgroundAttachment = 'fixed'
    } else {
      wrapper.style.backgroundImage = ''
    }
  }

  function hexToRgb(hex) {
    const m = hex.replace('#', '').match(/^([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i)
    return m ? { r: parseInt(m[1], 16), g: parseInt(m[2], 16), b: parseInt(m[3], 16) } : null
  }

  function darken(hex, pct) {
    const rgb = hexToRgb(hex)
    if (!rgb) return hex
    const f = 1 - pct / 100
    return '#' + [rgb.r, rgb.g, rgb.b].map(c => Math.round(c * f).toString(16).padStart(2, '0')).join('')
  }

  function lighten(hex, pct) {
    const rgb = hexToRgb(hex)
    if (!rgb) return hex
    const f = pct / 100
    return '#' + [rgb.r, rgb.g, rgb.b].map(c => Math.round(c + (255 - c) * f).toString(16).padStart(2, '0')).join('')
  }

  function setTheme(theme) {
    currentTheme.value = theme
    localStorage.setItem('taiji_theme', theme)
    applyTheme()
    _debouncedSaveUI({ theme })
  }

  function setAccent(color) {
    currentAccent.value = color
    localStorage.setItem('taiji_accent', color)
    applyAccent()
    _debouncedSaveUI({ accent: color })
  }

  function setBgImage(dataUrl) {
    currentBgImage.value = dataUrl
    if (dataUrl) {
      localStorage.setItem('taiji_bg_image', dataUrl)
    } else {
      localStorage.removeItem('taiji_bg_image')
    }
    applyBgImage()
  }

  function restoreUISettings(serverSettings) {
    if (!serverSettings || typeof serverSettings !== 'object') return
    let needsApplyTheme = false

    if (serverSettings.accent !== undefined) {
      currentAccent.value = serverSettings.accent
      localStorage.setItem('taiji_accent', serverSettings.accent)
      needsApplyTheme = true
    }
    if (serverSettings.theme !== undefined) {
      currentTheme.value = _normalizeTheme(serverSettings.theme)
      localStorage.setItem('taiji_theme', currentTheme.value)
      needsApplyTheme = true
    }
    if (serverSettings.lang !== undefined) {
      currentLang.value = serverSettings.lang
      localStorage.setItem('taiji_lang', serverSettings.lang)
    }
    if (needsApplyTheme) {
      applyTheme()
    }
  }

  // 初始化主题
  applyTheme()
  if (typeof window !== 'undefined' && window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: light)').addEventListener?.('change', () => {
      if (currentTheme.value === 'auto') applyTheme()
    })
  }

  return {
    // State
    currentTheme,
    currentAccent,
    currentBgImage,
    currentLang,
    showWorkspace,
    themes,
    // Getters
    resolvedTheme,
    resolvedDataTheme,
    // Actions
    t,
    toggleWorkspace,
    applyTheme,
    setTheme,
    setAccent,
    setBgImage,
    accentPresets,
    restoreUISettings,
  }
})
