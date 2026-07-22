<template>
  <div class="life-status-view">
    <!-- ═══ 顶栏 ═══ -->
    <header class="topbar">
      <div class="topbar-left">
        <span class="topbar-title">生命状态</span>
        <span class="topbar-sub">实时监控态极神经元网络</span>
      </div>
      <span class="topbar-spacer"></span>
      <n-tag
        :type="runtimeStore.connectionClass === 'connected' ? 'success' : 'error'"
        size="small"
        round
      >
        {{ runtimeStore.connectionStatus }}
      </n-tag>
      <button class="btn btn-outline" @click="toast('生命状态报告导出中...', 'info')">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
        导出报告
      </button>
    </header>

    <!-- ═══ 滚动内容区 ═══ -->
    <div class="scroll-area">

      <!-- ═══ KPI 卡片行 ═══ -->
      <div class="kpi-grid">
        <!-- 卡1：神经元总数 -->
        <div class="kpi-card" style="--kpi-color: var(--chart-1);">
          <div class="kpi-label">
            <svg class="kpi-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><circle cx="9" cy="9" r="1.8"/><circle cx="15" cy="9" r="1.8"/><circle cx="12" cy="15" r="1.8"/><path d="M12 3v1.5M12 19.5V21M3 12h1.5M19.5 12H21"/></svg>
            神经元总数
          </div>
          <div class="kpi-value">{{ (1247 + (life.total_interactions || 0)).toLocaleString() }}</div>
          <div class="kpi-trend trend-up">
            <svg class="tr-icon" viewBox="0 0 24 24"><path d="m18 15-6-6-6 6"/></svg>
            ↑12
          </div>
        </div>

        <!-- 卡2：活跃神经元 -->
        <div class="kpi-card" style="--kpi-color: var(--chart-2);">
          <div class="kpi-label">
            <svg class="kpi-icon" viewBox="0 0 24 24"><path d="M3 12h3l1.5-6 3 12 1.5-6h8"/></svg>
            活跃神经元
          </div>
          <div class="kpi-value">{{ Math.round((1247 + (life.total_interactions || 0)) * 0.873).toLocaleString() }}</div>
          <div class="kpi-trend trend-up">
            <svg class="tr-icon" viewBox="0 0 24 24"><path d="m18 15-6-6-6 6"/></svg>
            ↑47
          </div>
        </div>

        <!-- 卡3：共振强度 -->
        <div class="kpi-card" style="--kpi-color: var(--chart-3);">
          <div class="kpi-label">
            <svg class="kpi-icon" viewBox="0 0 24 24"><path d="M3 12h2M7 8v8M11 5v14M15 8v8M19 12h2"/></svg>
            共振强度
          </div>
          <div class="kpi-value">{{ (life.needs && life.needs.curiosity != null ? Math.min(0.99, 0.5 + life.needs.curiosity / 200) : 0.72).toFixed(2) }}</div>
          <div class="kpi-trend trend-stable">
            <svg class="tr-icon" viewBox="0 0 24 24"><path d="M5 12h14"/></svg>
            稳定
          </div>
        </div>

        <!-- 卡4：系统能量 -->
        <div class="kpi-card" style="--kpi-color: var(--chart-4);">
          <div class="kpi-label">
            <svg class="kpi-icon" viewBox="0 0 24 24"><rect x="7" y="5" width="4" height="14" rx="1"/><rect x="13" y="7" width="4" height="10" rx="1"/></svg>
            系统能量
          </div>
          <div class="kpi-value">{{ runtimeStore.memoryAvailablePct != null ? Math.round(runtimeStore.memoryAvailablePct) + '%' : '87%' }}</div>
          <div class="kpi-trend trend-down">
            <svg class="tr-icon" viewBox="0 0 24 24"><path d="m6 9 6 6 6-6"/></svg>
            ↓3
          </div>
        </div>
      </div>

      <!-- ═══ 双图表行 ═══ -->
      <div class="chart-grid">
        <!-- 能力雷达图 -->
        <div class="panel">
          <div class="panel-head">
            <span class="panel-title">
              <svg class="pt-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 3v4M12 17v4M3 12h4M17 12h4"/><circle cx="12" cy="12" r="3" fill="currentColor" stroke="none" opacity="0.3"/></svg>
              能力雷达图
            </span>
            <span class="panel-sub">六维度评估</span>
          </div>
          <div class="chart-wrap">
            <svg viewBox="0 0 400 300">
              <!-- 网格六边形 -->
              <polygon points="200,50 329.9,125 329.9,225 200,300 70.1,225 70.1,125" fill="none" stroke="var(--border)" stroke-width="1" opacity="0.6"/>
              <polygon points="200,75 297.5,131.5 297.5,218.5 200,275 102.5,218.5 102.5,131.5" fill="none" stroke="var(--border)" stroke-width="0.6" opacity="0.4"/>
              <polygon points="200,100 265,137.5 265,212.5 200,250 135,212.5 135,137.5" fill="none" stroke="var(--border)" stroke-width="0.6" opacity="0.4"/>
              <!-- 轴线 -->
              <line x1="200" y1="150" x2="200" y2="45" stroke="var(--border)" stroke-width="0.8" opacity="0.5"/>
              <line x1="200" y1="150" x2="333.9" y2="125" stroke="var(--border)" stroke-width="0.8" opacity="0.5"/>
              <line x1="200" y1="150" x2="333.9" y2="225" stroke="var(--border)" stroke-width="0.8" opacity="0.5"/>
              <line x1="200" y1="150" x2="200" y2="305" stroke="var(--border)" stroke-width="0.8" opacity="0.5"/>
              <line x1="200" y1="150" x2="66.1" y2="225" stroke="var(--border)" stroke-width="0.8" opacity="0.5"/>
              <line x1="200" y1="150" x2="66.1" y2="125" stroke="var(--border)" stroke-width="0.8" opacity="0.5"/>
              <!-- 数据填充 - 语言85 推理92 代码70 知识80 记忆88 学习65 -->
              <polygon points="200,65 279.7,104 260.6,185 200,230 123.8,194 143.7,117.5" fill="var(--chart-1)" opacity="0.18" stroke="var(--chart-1)" stroke-width="2" stroke-linejoin="round"/>
              <!-- 数据点 -->
              <circle cx="200" cy="65" r="4" fill="var(--chart-1)" stroke="var(--background)" stroke-width="2"/>
              <circle cx="279.7" cy="104" r="4" fill="var(--chart-1)" stroke="var(--background)" stroke-width="2"/>
              <circle cx="260.6" cy="185" r="4" fill="var(--chart-1)" stroke="var(--background)" stroke-width="2"/>
              <circle cx="200" cy="230" r="4" fill="var(--chart-1)" stroke="var(--background)" stroke-width="2"/>
              <circle cx="123.8" cy="194" r="4" fill="var(--chart-1)" stroke="var(--background)" stroke-width="2"/>
              <circle cx="143.7" cy="117.5" r="4" fill="var(--chart-1)" stroke="var(--background)" stroke-width="2"/>
              <!-- 维度标签 -->
              <text x="200" y="34" text-anchor="middle" font-size="12" fill="var(--muted-foreground)" font-family="var(--font-sans)">语言</text>
              <text x="310" y="99" text-anchor="start" font-size="12" fill="var(--muted-foreground)" font-family="var(--font-sans)">推理</text>
              <text x="310" y="218" text-anchor="start" font-size="12" fill="var(--muted-foreground)" font-family="var(--font-sans)">代码</text>
              <text x="200" y="290" text-anchor="middle" font-size="12" fill="var(--muted-foreground)" font-family="var(--font-sans)">知识</text>
              <text x="95" y="218" text-anchor="end" font-size="12" fill="var(--muted-foreground)" font-family="var(--font-sans)">记忆</text>
              <text x="95" y="99" text-anchor="end" font-size="12" fill="var(--muted-foreground)" font-family="var(--font-sans)">学习</text>
            </svg>
          </div>
        </div>

        <!-- 共振波形图 -->
        <div class="panel">
          <div class="panel-head">
            <span class="panel-title">
              <svg class="pt-icon" viewBox="0 0 24 24"><path d="M3 12h2.5L7 6l3 12 2.5-6h2"/><path d="M17.5 8v8M21 6v12"/></svg>
              共振波形图
            </span>
            <span class="panel-sub">24h 趋势</span>
          </div>
          <div class="chart-wrap">
            <svg viewBox="0 0 400 300">
              <!-- 渐变定义 -->
              <defs>
                <linearGradient id="waveGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="var(--chart-2)" stop-opacity="0.25"/>
                  <stop offset="100%" stop-color="var(--chart-2)" stop-opacity="0.02"/>
                </linearGradient>
              </defs>
              <!-- Y轴网格 -->
              <line x1="40" y1="44" x2="380" y2="44" stroke="var(--border)" stroke-width="0.8" opacity="0.5"/>
              <line x1="40" y1="98" x2="380" y2="98" stroke="var(--border)" stroke-width="0.8" opacity="0.5"/>
              <line x1="40" y1="152" x2="380" y2="152" stroke="var(--border)" stroke-width="0.8" opacity="0.5"/>
              <line x1="40" y1="206" x2="380" y2="206" stroke="var(--border)" stroke-width="0.8" opacity="0.5"/>
              <line x1="40" y1="260" x2="380" y2="260" stroke="var(--border)" stroke-width="0.8" opacity="0.5"/>
              <!-- 填充多边形 -->
              <polygon points="40,260 40,144.8 68.3,159.2 96.7,140 125,116 153.3,87.2 181.7,58.4 210,72.8 238.3,44 266.7,77.6 295,82.4 323.3,87.2 351.7,116 380,140 380,260" fill="url(#waveGrad)"/>
              <!-- 折线 -->
              <polyline points="40,144.8 68.3,159.2 96.7,140 125,116 153.3,87.2 181.7,58.4 210,72.8 238.3,44 266.7,77.6 295,82.4 323.3,87.2 351.7,116 380,140" fill="none" stroke="var(--chart-2)" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>
              <!-- 数据点 -->
              <circle cx="40" cy="144.8" r="3.5" fill="var(--chart-2)"/>
              <circle cx="210" cy="72.8" r="3.5" fill="var(--chart-2)"/>
              <circle cx="238.3" cy="44" r="4" fill="var(--chart-2)" stroke="var(--background)" stroke-width="2"/>
              <circle cx="380" cy="140" r="3.5" fill="var(--chart-2)"/>
              <!-- X轴标签 -->
              <text x="40" y="278" text-anchor="middle" font-size="11" fill="var(--muted-foreground)" font-family="var(--font-sans)">00:00</text>
              <text x="125" y="278" text-anchor="middle" font-size="11" fill="var(--muted-foreground)" font-family="var(--font-sans)">06:00</text>
              <text x="210" y="278" text-anchor="middle" font-size="11" fill="var(--muted-foreground)" font-family="var(--font-sans)">12:00</text>
              <text x="295" y="278" text-anchor="middle" font-size="11" fill="var(--muted-foreground)" font-family="var(--font-sans)">18:00</text>
              <text x="380" y="278" text-anchor="middle" font-size="11" fill="var(--muted-foreground)" font-family="var(--font-sans)">24:00</text>
              <!-- Y轴标签 -->
              <text x="36" y="48" text-anchor="end" font-size="10" fill="var(--muted-foreground)" font-family="var(--font-sans)">100</text>
              <text x="36" y="156" text-anchor="end" font-size="10" fill="var(--muted-foreground)" font-family="var(--font-sans)">50</text>
              <text x="36" y="264" text-anchor="end" font-size="10" fill="var(--muted-foreground)" font-family="var(--font-sans)">0</text>
              <!-- 峰值标注 -->
              <text x="238.3" y="32" text-anchor="middle" font-size="11" fill="var(--chart-2)" font-family="var(--font-sans)" font-weight="600">90</text>
              <line x1="238.3" y1="38" x2="238.3" y2="44" stroke="var(--chart-2)" stroke-width="1" stroke-dasharray="3 3"/>
            </svg>
          </div>
        </div>
      </div>

      <!-- ═══ 底部双列 ═══ -->
      <div class="bottom-grid">
        <!-- 左：神经元健康列表 -->
        <div class="panel">
          <div class="panel-head">
            <span class="panel-title">
              <svg class="pt-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="2"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1"/></svg>
              神经元健康列表
            </span>
            <span class="panel-sub">共 6 个</span>
          </div>
          <table class="neuron-table">
            <thead>
              <tr>
                <th>神经元 ID</th>
                <th>域</th>
                <th>状态</th>
                <th>活跃度</th>
                <th>最后激活</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><span class="n-id">N-0842</span></td>
                <td><span class="n-domain">语言理解</span></td>
                <td><span class="chip chip-active">活跃</span></td>
                <td>
                  <div class="n-activity">
                    <div class="n-progress"><div class="n-progress-bar" style="width:94%"></div></div>
                    <span class="n-progress-text">94%</span>
                  </div>
                </td>
                <td><span class="n-time">2分钟前</span></td>
                <td><button class="n-action"><svg class="ac-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>详情</button></td>
              </tr>
              <tr>
                <td><span class="n-id">N-1205</span></td>
                <td><span class="n-domain">推理逻辑</span></td>
                <td><span class="chip chip-active">活跃</span></td>
                <td>
                  <div class="n-activity">
                    <div class="n-progress"><div class="n-progress-bar" style="width:87%"></div></div>
                    <span class="n-progress-text">87%</span>
                  </div>
                </td>
                <td><span class="n-time">5分钟前</span></td>
                <td><button class="n-action"><svg class="ac-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>详情</button></td>
              </tr>
              <tr>
                <td><span class="n-id">N-0317</span></td>
                <td><span class="n-domain">代码生成</span></td>
                <td><span class="chip chip-active">活跃</span></td>
                <td>
                  <div class="n-activity">
                    <div class="n-progress"><div class="n-progress-bar" style="width:91%"></div></div>
                    <span class="n-progress-text">91%</span>
                  </div>
                </td>
                <td><span class="n-time">12秒前</span></td>
                <td><button class="n-action"><svg class="ac-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>详情</button></td>
              </tr>
              <tr>
                <td><span class="n-id">N-0721</span></td>
                <td><span class="n-domain">知识检索</span></td>
                <td><span class="chip chip-learning">学习中</span></td>
                <td>
                  <div class="n-activity">
                    <div class="n-progress"><div class="n-progress-bar" style="width:76%"></div></div>
                    <span class="n-progress-text">76%</span>
                  </div>
                </td>
                <td><span class="n-time">28分钟前</span></td>
                <td><button class="n-action"><svg class="ac-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>详情</button></td>
              </tr>
              <tr>
                <td><span class="n-id">N-1053</span></td>
                <td><span class="n-domain">记忆回响</span></td>
                <td><span class="chip chip-dormant">休眠</span></td>
                <td>
                  <div class="n-activity">
                    <div class="n-progress"><div class="n-progress-bar" style="width:62%"></div></div>
                    <span class="n-progress-text">62%</span>
                  </div>
                </td>
                <td><span class="n-time">47分钟前</span></td>
                <td><button class="n-action"><svg class="ac-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>详情</button></td>
              </tr>
              <tr>
                <td><span class="n-id">N-0014</span></td>
                <td><span class="n-domain">学习适配</span></td>
                <td><span class="chip chip-active">活跃</span></td>
                <td>
                  <div class="n-activity">
                    <div class="n-progress"><div class="n-progress-bar" style="width:89%"></div></div>
                    <span class="n-progress-text">89%</span>
                  </div>
                </td>
                <td><span class="n-time">3分钟前</span></td>
                <td><button class="n-action"><svg class="ac-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>详情</button></td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- 右：事件流（绑定 activityLog） -->
        <div class="panel">
          <div class="panel-head">
            <span class="panel-title">
              <svg class="pt-icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
              系统事件流
            </span>
            <span class="panel-sub">实时</span>
          </div>
          <div class="event-list">
            <template v-if="activityLog.length">
              <div
                v-for="(log, i) in activityLog"
                :key="i"
                class="event-item"
                :class="'ev-' + log.type"
              >
                <div class="event-dot">
                  <span class="ev-emoji">{{ log.emoji }}</span>
                </div>
                <div class="event-body">
                  <div class="event-text">{{ log.message }}</div>
                  <div class="event-meta">{{ log.time }}</div>
                </div>
              </div>
            </template>
            <template v-else>
              <div class="event-item">
                <div class="event-dot">
                  <svg class="ev-icon" viewBox="0 0 24 24"><path d="M13 3 5 14h6l-1 7 8-11h-6l1-7Z"/></svg>
                </div>
                <div class="event-body">
                  <div class="event-text">神经元 N-0317 在代码生成域完成一次共振峰值检测</div>
                  <div class="event-meta">12秒前</div>
                </div>
              </div>
              <div class="event-item">
                <div class="event-dot" style="background:color-mix(in srgb, var(--chart-2) 14%, transparent); color:var(--chart-2);">
                  <svg class="ev-icon" viewBox="0 0 24 24"><path d="M12 2v4M12 18v4M4.9 4.9l2.8 2.8M16.3 16.3l2.8 2.8M2 12h4M18 12h4M4.9 19.1l2.8-2.8M16.3 7.7l2.8-2.8"/></svg>
                </div>
                <div class="event-body">
                  <div class="event-text">自动优化：调整语言理解域 3 个神经元权重参数</div>
                  <div class="event-meta">2分钟前</div>
                </div>
              </div>
              <div class="event-item">
                <div class="event-dot" style="background:color-mix(in srgb, var(--destructive) 12%, transparent); color:var(--destructive);">
                  <svg class="ev-icon" viewBox="0 0 24 24"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.7 21a2 2 0 0 1-3.4 0"/></svg>
                </div>
                <div class="event-body">
                  <div class="event-text">监测告警：推理逻辑域 N-1205 活跃度突破阈值</div>
                  <div class="event-meta">8分钟前</div>
                </div>
              </div>
              <div class="event-item">
                <div class="event-dot" style="background:color-mix(in srgb, var(--chart-3) 14%, transparent); color:var(--chart-3);">
                  <svg class="ev-icon" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="3"/><path d="M9 3v18"/></svg>
                </div>
                <div class="event-body">
                  <div class="event-text">系统更新：知识检索域新增 4 个训练数据批次</div>
                  <div class="event-meta">23分钟前</div>
                </div>
              </div>
              <div class="event-item">
                <div class="event-dot" style="background:color-mix(in srgb, var(--chart-1) 14%, transparent); color:var(--chart-1);">
                  <svg class="ev-icon" viewBox="0 0 24 24"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6h6v6"/></svg>
                </div>
                <div class="event-body">
                  <div class="event-text">神经元 N-0842 激活，加入态极网络主干</div>
                  <div class="event-meta">41分钟前</div>
                </div>
              </div>
            </template>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, inject } from 'vue'
import { useRuntimeStore } from '@/stores/runtimeStore.js'
import { API_BASE, authFetch } from '@/composables/apiClient.js'
import {
  Heart, Footprints, Zap, Eye,
  Apple, Moon, Gamepad2,
  Activity
} from 'lucide-vue-next'
import NeedsPentagram from '@/components/NeedsPentagram.vue'

const runtimeStore = useRuntimeStore()
const toast = inject('toast', () => {})

const actionResult = ref('')
const activityLog = ref([])
const currentActivity = ref('')
const actionLoading = ref(false)

// 从 runtimeStore 获取生命数据
const life = computed(() => runtimeStore.life || {})

// 生命状态文本
const lifeStateText = computed(() => {
  const stateMap = { idle: '清醒', sleeping: '睡眠', feeding: '吸收', playing: '探索', working: '执行' }
  return stateMap[life.value.life_state || 'idle'] || '清醒'
})

// 工具是否可用
const toolsAvailable = computed(() => {
  return runtimeStore.tools && runtimeStore.tools.length > 0
})

// 最强烈的需求文本
const dominantNeedText = computed(() => {
  const needMap = {
    hunger: '🍚 饥饿 - 需要喂养',
    fatigue: '😴 疲劳 - 需要休息',
    boredom: '🎮 无聊 - 需要玩耍',
    stress: '😰 压力 - 需要放松',
    curiosity: '🔍 好奇 - 需要探索',
  }
  return needMap[life.value.dominant_need] || life.value.dominant_need || ''
})

// 格式化运行时间
function formatUptime(seconds) {
  if (!seconds || seconds <= 0) return '-'
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (h > 0) return `${h}小时${m}分钟`
  return `${m}分钟`
}

// 添加日志
function addLog(type, emoji, message) {
  activityLog.value.unshift({
    time: new Date().toLocaleTimeString(),
    type,
    emoji,
    message,
  })
  if (activityLog.value.length > 30) {
    activityLog.value.pop()
  }
}

// 通用生命活动调用
async function callLifeAction(action) {
  actionLoading.value = true
  actionResult.value = ''
  try {
    const resp = await authFetch(`${API_BASE}/api/life/${action}`, { method: 'POST' })
    const data = await resp.json()
    if (data.success) {
      actionResult.value = data.message
      const emojiMap = { feed: '🍚', sleep: '💤', play: '🎮' }
      addLog(action, emojiMap[action] || '✅', data.message)
      toast(`✅ ${data.message}`, 'success')
    } else {
      actionResult.value = `失败: ${data.message}`
      addLog(action, '❌', `失败: ${data.message}`)
      toast(`❌ ${data.message}`, 'error')
    }
  } catch (e) {
    actionResult.value = `请求失败: ${e.message}`
    addLog(action, '❌', `请求失败: ${e.message}`)
    toast(`❌ 请求失败: ${e.message}`, 'error')
  } finally {
    actionLoading.value = false
    currentActivity.value = ''
    // 刷新状态
    runtimeStore.refreshAll()
  }
}

function feedTaiji() {
  currentActivity.value = '🍚 正在吃饭...'
  callLifeAction('feed')
}

function sleepTaiji() {
  currentActivity.value = '💤 正在睡觉...'
  callLifeAction('sleep')
}

function playTaiji() {
  currentActivity.value = '🎮 正在玩耍...'
  callLifeAction('play')
}

async function trainTaiji() {
  actionLoading.value = true
  currentActivity.value = '🧠 正在训练...'
  try {
    const resp = await authFetch(`${API_BASE}/api/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ epochs: 3, learning_rate: 5e-5 }),
    })
    const data = await resp.json()
    actionResult.value = data.message || '训练已启动'
    addLog('train', '🧠', data.message || '训练已启动')
  } catch (e) {
    actionResult.value = `训练请求失败: ${e.message}`
  } finally {
    actionLoading.value = false
    currentActivity.value = ''
  }
}

let refreshInterval = null
onMounted(() => {
  runtimeStore.refreshAll()
  // 每 15 秒自动刷新
  refreshInterval = setInterval(() => {
    runtimeStore.refreshAll().catch(() => {})
  }, 15000)
})

onUnmounted(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
    refreshInterval = null
  }
})
</script>

<style scoped>
/* ═══ 视图容器（豆包设计 token） ═══ */
.life-status-view {
  --chart-1: var(--primary);
  --chart-2: var(--success, #10b981);
  --chart-3: var(--warning, #f59e0b);
  --chart-4: var(--destructive, #ef4444);

  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--background);
  color: var(--foreground);
  font-family: var(--font-sans);
  overflow: hidden;
}

/* ═══ 顶栏 ═══ */
.topbar {
  height: 52px;
  flex: none;
  padding: 0 18px;
  display: flex;
  align-items: center;
  gap: 12px;
  border-bottom: 1px solid var(--border);
}
.topbar-left {
  display: flex;
  flex-direction: column;
  justify-content: center;
  line-height: 1.15;
}
.topbar-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--foreground);
}
.topbar-sub {
  margin-top: 2px;
  font-size: 0.72rem;
  color: var(--muted-foreground);
}
.topbar-spacer {
  flex: 1;
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
  transition: background 150ms ease, border-color 150ms ease, transform 120ms ease, color 150ms ease;
}
.btn:active { transform: translateY(1px); }
.btn:focus-visible {
  outline: 2px solid var(--ring);
  outline-offset: 2px;
}
.btn-outline {
  background: var(--background);
  color: var(--foreground);
  border-color: var(--border);
}
.btn-outline:hover {
  background: var(--muted);
}

/* ═══ 滚动内容区 ═══ */
.scroll-area {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  padding: 28px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* ═══ KPI 卡片行 ═══ */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
}
.kpi-card {
  position: relative;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px 18px 16px 22px;
  overflow: hidden;
  transition: border-color 160ms ease, transform 160ms ease;
}
.kpi-card:hover {
  border-color: color-mix(in srgb, var(--primary) 30%, var(--border));
  transform: translateY(-2px);
}
.kpi-card::before {
  content: "";
  position: absolute;
  left: 0;
  top: 10px;
  bottom: 10px;
  width: 4px;
  background: var(--kpi-color, var(--chart-1));
  border-radius: 0 4px 4px 0;
}
.kpi-label {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 0.8rem;
  color: var(--muted-foreground);
  margin-bottom: 10px;
}
.kpi-icon {
  width: 16px;
  height: 16px;
  flex: none;
  color: var(--kpi-color, var(--chart-1));
  stroke: currentColor;
  fill: none;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.kpi-value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--foreground);
  letter-spacing: -0.02em;
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
}
.kpi-trend {
  margin-top: 8px;
  font-size: 0.76rem;
  font-weight: 600;
  display: inline-flex;
  align-items: center;
  gap: 4px;
}
.tr-icon {
  width: 13px;
  height: 13px;
  flex: none;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.trend-up { color: var(--chart-2); }
.trend-down { color: var(--destructive); }
.trend-stable { color: var(--muted-foreground); }

/* ═══ 图表面板 ═══ */
.chart-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 16px;
}
.panel {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
}
.panel-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}
.panel-title {
  font-size: 0.92rem;
  font-weight: 600;
  color: var(--foreground);
  display: flex;
  align-items: center;
  gap: 8px;
}
.pt-icon {
  width: 17px;
  height: 17px;
  flex: none;
  color: var(--primary);
  stroke: currentColor;
  fill: none;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.panel-sub {
  font-size: 0.74rem;
  color: var(--muted-foreground);
}
.chart-wrap {
  flex: 1;
  display: grid;
  place-items: center;
  min-height: 260px;
}
.chart-wrap svg {
  width: 100%;
  height: auto;
  max-height: 300px;
}

/* ═══ 底部双列 ═══ */
.bottom-grid {
  display: grid;
  grid-template-columns: 1.7fr minmax(0, 1fr);
  gap: 16px;
  align-items: start;
}

/* ═══ 神经元健康表 ═══ */
.neuron-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.84rem;
}
.neuron-table thead th {
  text-align: left;
  font-weight: 600;
  color: var(--muted-foreground);
  font-size: 0.76rem;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
.neuron-table tbody td {
  padding: 11px 12px;
  border-bottom: 1px solid var(--border);
  color: var(--foreground);
  vertical-align: middle;
  white-space: nowrap;
}
.neuron-table tbody tr:last-child td { border-bottom: 0; }
.neuron-table tbody tr:hover {
  background: color-mix(in srgb, var(--accent) 35%, transparent);
}
.n-id {
  font-family: var(--font-mono);
  font-size: 0.8rem;
  color: var(--foreground);
}
.n-domain { color: var(--muted-foreground); }
.n-time {
  color: var(--muted-foreground);
  font-size: 0.78rem;
}
.n-activity {
  display: flex;
  align-items: center;
  gap: 9px;
  min-width: 130px;
}
.n-progress {
  flex: 1;
  height: 6px;
  border-radius: 999px;
  background: var(--muted);
  overflow: hidden;
}
.n-progress-bar {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--chart-1), var(--chart-2));
}
.n-progress-text {
  font-size: 0.76rem;
  color: var(--muted-foreground);
  font-variant-numeric: tabular-nums;
  width: 32px;
  text-align: right;
}
.n-action {
  border: 0;
  background: transparent;
  color: var(--primary);
  font-size: 0.78rem;
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 999px;
  transition: background 140ms ease;
  font-weight: 500;
  cursor: pointer;
}
.n-action:hover {
  background: color-mix(in srgb, var(--primary) 12%, transparent);
}
.ac-icon {
  width: 13px;
  height: 13px;
  flex: none;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}

/* ═══ 状态 Chip ═══ */
.chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  border-radius: 999px;
  font-size: 0.74rem;
  font-weight: 500;
  line-height: 1.5;
}
.chip::before {
  content: "";
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
  flex: none;
}
.chip-active {
  color: var(--chart-2);
  background: color-mix(in srgb, var(--chart-2) 14%, transparent);
}
.chip-dormant {
  color: var(--muted-foreground);
  background: color-mix(in srgb, var(--muted-foreground) 14%, transparent);
}
.chip-learning {
  color: var(--chart-1);
  background: color-mix(in srgb, var(--chart-1) 14%, transparent);
}

/* ═══ 事件流 ═══ */
.event-list {
  display: flex;
  flex-direction: column;
  gap: 0;
  margin-top: 2px;
}
.event-item {
  display: flex;
  gap: 12px;
  padding: 10px 6px;
  border-bottom: 1px dashed var(--border);
  transition: background 140ms ease;
}
.event-item:hover {
  background: color-mix(in srgb, var(--accent) 25%, transparent);
}
.event-item:last-child { border-bottom: 0; }
.event-dot {
  width: 30px;
  height: 30px;
  border-radius: 10px;
  flex: none;
  display: grid;
  place-items: center;
  background: color-mix(in srgb, var(--chart-1) 14%, transparent);
  color: var(--chart-1);
}
.ev-icon {
  width: 15px;
  height: 15px;
  flex: none;
  stroke: currentColor;
  fill: none;
  stroke-width: 1.7;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.ev-emoji {
  font-size: 1rem;
  line-height: 1;
}
.event-body {
  flex: 1;
  min-width: 0;
}
.event-text {
  font-size: 0.82rem;
  color: var(--foreground);
  line-height: 1.45;
}
.event-meta {
  font-size: 0.72rem;
  color: var(--muted-foreground);
  margin-top: 4px;
}

/* 事件类型色（绑定 activityLog type） */
.ev-feed .event-dot {
  background: color-mix(in srgb, var(--chart-2) 14%, transparent);
  color: var(--chart-2);
}
.ev-sleep .event-dot {
  background: color-mix(in srgb, var(--chart-1) 14%, transparent);
  color: var(--chart-1);
}
.ev-play .event-dot {
  background: color-mix(in srgb, var(--chart-3) 14%, transparent);
  color: var(--chart-3);
}
.ev-train .event-dot {
  background: color-mix(in srgb, var(--primary) 14%, transparent);
  color: var(--primary);
}

/* ═══ 响应式 ═══ */
@media (max-width: 1180px) {
  .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .chart-grid { grid-template-columns: 1fr; }
  .bottom-grid { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .kpi-grid { grid-template-columns: 1fr; }
  .scroll-area { padding: 18px; gap: 16px; }
  .topbar { padding: 0 14px; }
}
</style>
