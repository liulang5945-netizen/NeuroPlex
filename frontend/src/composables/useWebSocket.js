/**
 * 态极 WebSocket 客户端
 * ====================
 *
 * 前端与态极核心的实时通信。
 */
import { ref, onMounted, onUnmounted } from 'vue'

// 动态 WebSocket URL：开发环境用 localhost，生产环境用当前 host
const WS_URL = import.meta.env.DEV
  ? 'ws://localhost:8765'
  : `ws://${window.location.hostname}:8765`

export function useWebSocket() {
  const connected = ref(false)
  const taijiStatus = ref(null)
  const lastMessage = ref(null)

  let ws = null
  let reconnectTimer = null
  const listeners = new Map()
  let manuallyClosed = false

  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return
    manuallyClosed = false

    ws = new WebSocket(WS_URL)

    ws.onopen = () => {
      connected.value = true
      reconnectAttempt = 0
      clearReconnect()
    }

    ws.onclose = () => {
      connected.value = false
      if (!manuallyClosed) scheduleReconnect()
    }

    ws.onerror = (error) => {
      connected.value = false
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        lastMessage.value = data
        handleMessage(data)
      } catch (e) {
        // 忽略格式异常，保持客户端可用
      }
    }
  }

  function disconnect() {
    manuallyClosed = true
    clearReconnect()
    if (ws) {
      ws.close()
      ws = null
    }
  }

  let reconnectAttempt = 0
  const MAX_RECONNECT_ATTEMPTS = 6

  function scheduleReconnect() {
    clearReconnect()
    if (reconnectAttempt >= MAX_RECONNECT_ATTEMPTS) {
      console.warn('WebSocket 重连次数超限，停止重连')
      return
    }
    const delay = Math.min(5000 * Math.pow(1.6, reconnectAttempt), 45000)
    reconnectAttempt++
    reconnectTimer = setTimeout(() => {
      connect()
    }, delay)
  }

  function clearReconnect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
  }

  function handleMessage(data) {
    const { type } = data

    // 触发监听器
    const typeListeners = listeners.get(type)
    if (typeListeners) {
      typeListeners.forEach(callback => callback(data))
    }

    // 更新状态
    if (type === 'status_response') {
      taijiStatus.value = data.status
    }
  }

  function send(data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data))
    } else {
      connected.value = false
    }
  }

  function on(type, callback) {
    if (!listeners.has(type)) {
      listeners.set(type, new Set())
    }
    listeners.get(type).add(callback)

    // 返回取消监听函数
    return () => {
      const typeListeners = listeners.get(type)
      if (typeListeners) {
        typeListeners.delete(callback)
      }
    }
  }

  // 聊天
  function chat(message) {
    send({ type: 'chat', message })
  }

  // 喂养
  function feed(content, contentType = 'text') {
    send({ type: 'feed', content, content_type: contentType })
  }

  // 训练
  function train(epochs = 3, learningRate = 5e-5) {
    send({ type: 'train', epochs, learning_rate: learningRate })
  }

  // 睡眠
  function sleep() {
    send({ type: 'sleep' })
  }

  // 玩耍
  function play() {
    send({ type: 'play' })
  }

  // 语音识别
  function listenVoice() {
    send({ type: 'voice', action: 'listen' })
  }

  // 语音合成
  function speakText(text) {
    send({ type: 'voice', action: 'speak', text })
  }

  // 获取状态
  function getStatus() {
    send({ type: 'status' })
  }

  onMounted(() => {
    connect()
  })

  onUnmounted(() => {
    disconnect()
  })

  return {
    connected,
    taijiStatus,
    lastMessage,
    connect,
    disconnect,
    send,
    on,
    chat,
    feed,
    train,
    sleep,
    play,
    listenVoice,
    speakText,
    getStatus,
  }
}
