<template>
  <svg
    :width="size"
    :height="size"
    viewBox="0 0 128 128"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    role="img"
    aria-label="态极"
    class="taiji-logo"
    :class="{ 'is-thinking': thinking, 'is-idle': !thinking }"
  >
    <defs>
      <filter id="ink" x="-5%" y="-5%" width="110%" height="110%">
        <feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="4" seed="5" result="noise"/>
        <feDisplacementMap in="SourceGraphic" in2="noise" scale="1.5"/>
      </filter>
    </defs>

    <!-- 外环 -->
    <circle cx="64" cy="64" r="60" stroke="#1a1a1a" stroke-width="2" fill="none" opacity="0.15" filter="url(#ink)"/>
    <circle cx="64" cy="64" r="56" stroke="#1a1a1a" stroke-width="0.5" fill="none" opacity="0.08"/>

    <!-- 太极主体 -->
    <g filter="url(#ink)" class="taiji-body">
      <!-- 阳 (白色) -->
      <path d="M64 4 A60 60 0 0 1 64 124 A30 30 0 0 1 64 64 A30 30 0 0 0 64 4 Z" 
        fill="#f5f5f0" opacity="0.9"/>
      
      <!-- 阴 (黑色) -->
      <path d="M64 124 A60 60 0 0 1 64 4 A30 30 0 0 1 64 64 A30 30 0 0 0 64 124 Z" 
        fill="#1a1a1a" opacity="0.85"/>
    </g>

    <!-- S曲线 -->
    <path d="M64 4 A60 60 0 0 1 64 124" stroke="#1a1a1a" stroke-width="2.5" fill="none" opacity="0.3" filter="url(#ink)"/>

    <!-- 阳眼 -->
    <circle cx="64" cy="34" r="8" fill="#1a1a1a" opacity="0.9"/>
    <circle cx="64" cy="34" r="4" fill="#f5f5f0" opacity="0.3" class="eye-pulse"/>
    
    <!-- 阴眼 -->
    <circle cx="64" cy="94" r="8" fill="#f5f5f0" opacity="0.9"/>
    <circle cx="64" cy="94" r="4" fill="#1a1a1a" opacity="0.3" class="eye-pulse"/>

    <!-- 中心点 -->
    <circle cx="64" cy="64" r="3" fill="#1a1a1a" opacity="0.5" class="center-pulse"/>

    <!-- 流动光环 - 主要动态元素 -->
    <circle cx="64" cy="64" r="52" fill="none" stroke="#1a1a1a" stroke-width="3" 
      stroke-dasharray="20 10 5 10" opacity="0.5" class="flow-ring"/>
    <circle cx="64" cy="64" r="48" fill="none" stroke="#4a4a4a" stroke-width="2" 
      stroke-dasharray="15 8 3 8" opacity="0.3" class="flow-ring-reverse"/>
  </svg>
</template>

<script setup>
defineProps({
  size: { type: [Number, String], default: 40 },
  thinking: { type: Boolean, default: false },
})
</script>

<style scoped>
.taiji-logo {
  flex-shrink: 0;
}

/* 流动光环 - 顺时针旋转 */
.flow-ring {
  animation: ring-spin 4s linear infinite;
  transform-origin: 64px 64px;
}

/* 流动光环 - 逆时针旋转 */
.flow-ring-reverse {
  animation: ring-spin-reverse 6s linear infinite;
  transform-origin: 64px 64px;
}

@keyframes ring-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

@keyframes ring-spin-reverse {
  from { transform: rotate(0deg); }
  to { transform: rotate(-360deg); }
}

/* 鱼眼脉动 */
.eye-pulse {
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 0.3; r: 4; }
  50% { opacity: 0.8; r: 5; }
}

/* 中心点脉动 */
.center-pulse {
  animation: center-glow 3s ease-in-out infinite;
}

@keyframes center-glow {
  0%, 100% { opacity: 0.5; r: 3; }
  50% { opacity: 1; r: 4; }
}

/* 整体呼吸 */
.taiji-body {
  animation: breathe 4s ease-in-out infinite;
}

@keyframes breathe {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.9; }
}

/* 思考时加速 */
.taiji-logo.is-thinking .flow-ring {
  animation-duration: 2s;
}

.taiji-logo.is-thinking .flow-ring-reverse {
  animation-duration: 3s;
}

/* 空闲时减速 */
.taiji-logo.is-idle .flow-ring {
  animation-duration: 6s;
}

.taiji-logo.is-idle .flow-ring-reverse {
  animation-duration: 8s;
}

@media (prefers-reduced-motion: reduce) {
  .flow-ring, .flow-ring-reverse, .eye-pulse, .center-pulse, .taiji-body {
    animation: none;
  }
}
</style>
