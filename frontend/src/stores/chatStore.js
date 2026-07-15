import { defineStore } from 'pinia'
import { ref, computed, nextTick } from 'vue'
import { API_BASE, authFetch } from '@/composables/apiClient.js'
import { setChatReceiving } from '@/composables/useApi.js'
import { useRuntimeStore } from '@/stores/runtimeStore.js'

let _msgIdCounter = 0
function _nextMsgId() { return `msg_${Date.now()}_${++_msgIdCounter}` }

export const useChatStore = defineStore('chat', () => {
  // === State ===
  const sessions = ref([])
  const currentSessionId = ref(null)
  const messages = ref([])
  const chatInput = ref('')
  const isLoading = ref(false)
  const isReceiving = ref(false)
  const lastEngineType = ref('')  // 记录最近一次使用的引擎类型
  const sessionsLoaded = ref(false) // 标记是否已从后端加载过
  const lifeNeeds = ref(null)  // 态极内在需求（来自推理过程中的生命状态事件）
  let abortController = null

  // === Getters ===
  const currentSessionName = computed(() => {
    const s = sessions.value.find(s => s.id === currentSessionId.value)
    return s ? s.name : ''
  })

  // === Actions ===

  /**
   * 从后端加载所有历史会话（并行批量加载，避免 N+1）。
   * 每批最多 6 个并发请求，避免压垮后端。
   */
  async function loadSessions() {
    try {
      const res = await authFetch(`${API_BASE}/api/chat/sessions`)
      if (!res.ok) return
      const list = await res.json()
      if (!Array.isArray(list) || list.length === 0) {
        sessionsLoaded.value = true
        return
      }

      // 按 updated_at 降序排列
      list.sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0))

      async function _fetchSessionDetail(item) {
        try {
          const detailRes = await authFetch(`${API_BASE}/api/chat/history/${item.session_id}`)
          if (!detailRes.ok) return null
          const detail = await detailRes.json()
          return {
            id: Number(item.session_id) || item.session_id,
            name: detail.name || item.name || '',
            messages: detail.messages || [],
          }
        } catch (e) {
          console.warn('[chatStore] session detail load failed:', e.message)
          return null
        }
      }

      // 分批并行加载，每批最多 6 个并发
      const BATCH = 6
      const loaded = []
      for (let i = 0; i < list.length; i += BATCH) {
        const batch = list.slice(i, i + BATCH)
        const results = await Promise.all(batch.map(_fetchSessionDetail))
        for (const r of results) {
          if (r) loaded.push(r)
        }
      }

      if (loaded.length > 0) {
        sessions.value = loaded
        currentSessionId.value = loaded[0].id
        messages.value = [...(loaded[0].messages || [])]
      }
      sessionsLoaded.value = true
    } catch (e) {
      console.warn('[ChatStore] 加载历史会话失败:', e.message)
      sessionsLoaded.value = true
    }
  }

  function createNewSession() {
    const id = Date.now()
    sessions.value.unshift({
      id,
      name: `新对话 ${sessions.value.length + 1}`,
      messages: []
    })
    currentSessionId.value = id
    messages.value = []
  }

  function switchSession(id) {
    currentSessionId.value = id
    const s = sessions.value.find(s => s.id === id)
    messages.value = s ? (s.messages || []) : []
  }

  function deleteSession(id) {
    // 先从后端删除
    authFetch(`${API_BASE}/api/chat/history/${id}`, { method: 'DELETE' }).catch(() => {})
    sessions.value = sessions.value.filter(s => s.id !== id)
    if (currentSessionId.value === id) {
      if (sessions.value.length) switchSession(sessions.value[0].id)
      else createNewSession()
    }
  }

  function clearCurrentChat() {
    messages.value = []
    const s = sessions.value.find(s => s.id === currentSessionId.value)
    if (s) s.messages = []
  }

  function setChatInput(val) {
    chatInput.value = val
  }

  function appendRuntimeMessage(role, content, extra = {}) {
    messages.value.push({
      id: _nextMsgId(),
      role,
      content,
      timestamp: Date.now(),
      ...extra,
    })
    const s = sessions.value.find(s => s.id === currentSessionId.value)
    if (s) s.messages = [...messages.value]
  }

  function getSystemPrompt() {
    return localStorage.getItem('taiji_system_prompt') || '你是态极，一个独立的AI生命体。你用自己的大脑思考，用工具探索世界。'
  }

  async function sendMessage(engineType, attachments = []) {
    const readyAttachments = attachments.filter(att => att && !att.uploading)
    const input = chatInput.value.trim() || defaultPromptForAttachments(readyAttachments)
    if ((!input && readyAttachments.length === 0) || isLoading.value) return

    const s = sessions.value.find(s => s.id === currentSessionId.value)
    if (s && s.id !== sessions.value[0]?.id) {
      sessions.value = sessions.value.filter(x => x.id !== s.id)
      sessions.value.unshift(s)
    }

    const attachmentContext = attachments
      .filter(att => att && !att.uploading)
      .map(att => {
        const mediaPath = att.savedPath ? ` | path: ${att.savedPath}` : ''
        return `[${att.modality || 'file'}: ${att.name}${mediaPath}]\n${att.parsedText || ''}`.trim()
      })
      .filter(Boolean)
      .join('\n\n')
    const promptInput = attachmentContext ? `${input}\n\n${attachmentContext}` : input

    messages.value.push({
      id: _nextMsgId(),
      role: 'user',
      content: input,
      attachments: attachments.map(att => ({
        name: att.name,
        modality: att.modality,
        type: att.type,
        previewUrl: att.previewUrl,
        publicUrl: att.publicUrl,
        savedPath: att.savedPath,
        parsedText: att.parsedText,
      })),
    })
    if (s) s.messages = [...messages.value]
    chatInput.value = ''

    // 自动更新对话标题：如果是第一条用户消息且当前名称还是默认的"新对话"，则用用户输入前20个字符作为标题
    if (s && s.name && /^新对话\s*\d+$/.test(s.name) && messages.value.filter(m => m.role === 'user').length === 1) {
      const autoTitle = input.length > 20 ? input.slice(0, 20) + '…' : input
      s.name = autoTitle
      // 同步更新到后端（异步，不阻塞）
      authFetch(`${API_BASE}/api/chat/history/${s.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: autoTitle, messages: s.messages }),
      }).catch(() => {})
    }
    isLoading.value = true
    isReceiving.value = false

    // 记录引擎类型供重新生成使用
    lastEngineType.value = 'agent'  // 统一使用 ReAct 引擎
    // 更新全局接收状态
    setChatReceiving(true)

    abortController = new AbortController()

    // 接收超时保护：15 秒内没有收到第一个 token 则自动中止
    let receiveTimeout = setTimeout(() => {
      if (!isReceiving.value && isLoading.value && abortController) {
        console.warn('[chatStore] 接收超时（15秒无响应），自动中止')
        abortController.abort()
      }
    }, 15000)

    try {
      const history = []
      let pendingUser = ''
      for (const msg of messages.value.slice(0, -1)) {
        if (msg.role === 'user') {
          pendingUser = msg.content || ''
        } else if (msg.role === 'assistant' && pendingUser) {
          history.push([pendingUser, msg.content || ''])
          pendingUser = ''
        }
      }

      const sysPrompt = getSystemPrompt()
      const runtimeStore = useRuntimeStore()

      const res = await authFetch(`${API_BASE}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: promptInput,
          system_prompt: sysPrompt,
          history,
          engine: 'agent',  // 统一使用 ReAct 引擎
          agent_max_iterations: Number(runtimeStore.agentPrefs.maxIterations || 10),
          agent_temperature: Number(runtimeStore.agentPrefs.temperature || 0.7)
        }),
        signal: abortController.signal,
      })

      if (!res.ok) {
        const e = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(e.detail || `HTTP ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      const aiMsg = { id: _nextMsgId(), role: 'assistant', content: '', toolCalls: [] }
      messages.value.push(aiMsg)
      let buffer = ''

      // 处理 SSE data 行的辅助函数
      function processData(data) {
        if (data.trim() === '[DONE]') {
          isReceiving.value = false
          reader.cancel()
          return false
        }
        if (data === '[START]') {
          aiMsg.content = ''
          return true
        }

        // 尝试解析结构化事件（ReAct 引擎）
        let parsed = null
        try {
          parsed = JSON.parse(data)
        } catch (e) { console.debug('[chatStore] JSON parse failed:', e.message) }

        if (parsed && parsed.type) {
          // 结构化 ReAct 事件
          if (!isReceiving.value) isReceiving.value = true
          const evt = parsed
          switch (evt.type) {
            case 'life':
              // 生命状态事件 — 更新全局生命状态
              if (evt.data?.needs) {
                lifeNeeds.value = evt.data.needs
              }
              break
            case 'thought':
              aiMsg.content += evt.data?.content || ''
              break
            case 'action':
              aiMsg.toolCalls.push({
                tool: evt.data?.tool,
                args: evt.data?.args,
                status: 'running',
              })
              break
            case 'observation':
              const lastTool = aiMsg.toolCalls.findLast(t => t.tool === evt.data?.tool && t.status === 'running')
              if (lastTool) {
                lastTool.result = evt.data?.result
                lastTool.status = 'done'
              }
              break
            case 'final':
              aiMsg.content = evt.data?.answer || aiMsg.content
              break
            case 'error':
              aiMsg.content += `\n\n❌ ${evt.data?.error || '未知错误'}`
              break
            case 'start':
            case 'done':
            case 'cancelled':
              break
          }
        } else if (typeof parsed === 'string') {
          // JSON 编码的纯文本 chunk
          aiMsg.content += parsed
          if (!isReceiving.value) isReceiving.value = true
        } else {
          // 未编码的纯文本 chunk（向后兼容）
          aiMsg.content += data
          if (!isReceiving.value) isReceiving.value = true
        }
        messages.value[messages.value.length - 1] = { ...aiMsg }
        return true
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        // 按 SSE 规范收集 data 行（多行 data 用 \n 拼接）
        let dataLines = []
        for (const line of lines) {
          if (line.startsWith('data:')) {
            dataLines.push(line.startsWith('data: ') ? line.slice(6) : line.slice(5))
          } else if (line.startsWith(':')) {
            // SSE 注释，忽略
            continue
          } else {
            // 非 data 行 = 事件边界，处理已收集的 data
            if (dataLines.length > 0) {
              if (!processData(dataLines.join('\n'))) break
              dataLines = []
            }
          }
        }
        // 处理本批次最后收集的 data（无后续空行分隔时）
        if (dataLines.length > 0) {
          if (!processData(dataLines.join('\n'))) break
        }
      }

      if (s) s.messages = [...messages.value]
    } catch (err) {
      if (err.name !== 'AbortError') {
        const m = messages.value[messages.value.length - 1]
        if (m && m.role === 'assistant') {
          m.content += `\n\n❌ 错误: ${err.message}`
        } else {
          messages.value.push({ id: _nextMsgId(), role: 'assistant', content: `❌ 错误: ${err.message}` })
        }
      }
    } finally {
      clearTimeout(receiveTimeout)
      isLoading.value = false
      isReceiving.value = false
      abortController = null
      setChatReceiving(false)

      // 对话完成后自动保存到后端（异步，不阻塞）
      if (s && s.messages && s.messages.length > 0) {
        authFetch(`${API_BASE}/api/chat/history/${s.id}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: s.name, messages: s.messages }),
        }).catch(() => {})
      }
    }
  }

  function defaultPromptForAttachments(attachments) {
    if (!attachments.length) return ''
    const types = new Set(attachments.map(att => att.modality))
    if (types.has('audio')) return '请转写并理解这段语音。'
    if (types.has('video')) return '请理解这个视频的内容。'
    if (types.has('image')) return '请描述并分析这张图片。'
    return '请阅读并分析这些附件。'
  }

  function stopGeneration() {
    if (abortController) abortController.abort()
    isLoading.value = false
    isReceiving.value = false
    setChatReceiving(false)
  }

  function regenerateMessage(msgId) {
    const idx = messages.value.findIndex(m => m.id === msgId)
    if (idx > 0 && messages.value[idx - 1]?.role === 'user') {
      const userMsg = messages.value[idx - 1]
      const m = userMsg.content
      const attachments = userMsg.attachments || []
      messages.value.splice(idx - 1, 2)
      chatInput.value = m
      nextTick(() => sendMessage(lastEngineType.value || 'taiji', attachments))
    }
  }

  return {
    // State
    sessions,
    currentSessionId,
    messages,
    chatInput,
    isLoading,
    isReceiving,
    sessionsLoaded,
    lifeNeeds,
    // Getters
    currentSessionName,
    // Actions
    loadSessions,
    createNewSession,
    switchSession,
    deleteSession,
    clearCurrentChat,
    setChatInput,
    appendRuntimeMessage,
    sendMessage,
    stopGeneration,
    regenerateMessage,
  }
})
