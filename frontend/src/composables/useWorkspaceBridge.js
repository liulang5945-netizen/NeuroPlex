/**
 * 工作台桥接 composable
 * ====================
 *
 * 打通对话、工具、代码、终端的工作流：
 *
 * 1. 对话中生成文件 → 自动打开到 IDE
 * 2. 运行命令 → 终端结果回流到聊天
 * 3. 错误自动解释 → 推送到聊天
 * 4. 工具调用结果 → 可在 IDE 中查看
 *
 * 使用方式：
 *   const bridge = useWorkspaceBridge()
 *   bridge.openFileInIDE('path/to/file.py')
 *   bridge.runCommandInTerminal('python test.py')
 *   bridge.sendTerminalOutputToChat('output text')
 */
import { ref } from 'vue'
import { API_BASE, authFetch } from './apiClient.js'

// 全局状态（跨组件共享）
const pendingFileOpen = ref(null)
const pendingCommand = ref(null)
const terminalOutputBuffer = ref([])
const chatCallback = ref(null)

export function useWorkspaceBridge() {

  /**
   * 注册聊天回调（ChatView 挂载时调用）
   * 当终端有输出或错误时，通过这个回调推送到聊天
   */
  function registerChatCallback(callback) {
    chatCallback.value = callback
  }

  /**
   * 在 IDE 中打开文件
   * 对话中生成的文件可以通过这个方法自动打开
   */
  function openFileInIDE(filePath) {
    pendingFileOpen.value = filePath
    // 触发 WorkspaceView 监听
    window.dispatchEvent(new CustomEvent('taiji-open-file', { detail: { path: filePath } }))
  }

  /**
   * 在终端中运行命令
   * Agent 可以通过这个方法在终端执行命令
   */
  function runCommandInTerminal(command) {
    pendingCommand.value = command
    window.dispatchEvent(new CustomEvent('taiji-run-command', { detail: { command } }))
  }

  /**
   * 将终端输出发送到聊天
   * 终端执行结果可以通过这个方法回流到对话
   */
  function sendTerminalOutputToChat(output, type = 'terminal') {
    if (chatCallback.value) {
      chatCallback.value({
        type,
        content: output,
        timestamp: Date.now(),
      })
    }
    terminalOutputBuffer.value.push({ output, type, timestamp: Date.now() })
    // 保持最近 100 条
    if (terminalOutputBuffer.value.length > 100) {
      terminalOutputBuffer.value = terminalOutputBuffer.value.slice(-100)
    }
  }

  /**
   * 发送错误到聊天并自动解释
   */
  async function sendErrorToChat(error, context = '') {
    // 发送原始错误
    sendTerminalOutputToChat(`❌ 错误: ${error}`, 'error')

    // 尝试自动解释错误
    try {
      const r = await authFetch(`${API_BASE}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: `请简短解释以下错误并给出修复建议：\n${error}\n${context ? '上下文：' + context : ''}`,
          engine: 'taiji',
          system_prompt: '你是态极。请用简洁的中文解释错误原因并给出具体的修复建议。不超过 3 句话。',
        }),
      })

      if (r.ok) {
        const reader = r.body.getReader()
        const decoder = new TextDecoder()
        let explanation = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          const chunk = decoder.decode(value, { stream: true })
          for (const line of chunk.split('\n')) {
            if (line.startsWith('data: ')) {
              const p = line.slice(6)
              if (p !== '[DONE]') {
                try {
                  const parsed = JSON.parse(p)
                  if (parsed.type === 'final') {
                    explanation = parsed.data?.answer || ''
                  } else if (parsed.type === 'thought') {
                    explanation += parsed.data?.content || ''
                  }
                } catch (e) { console.debug('[WSBridge] parse error:', e.message) }
                  explanation += p
                }
              }
            }
          }
        }

        if (explanation) {
          sendTerminalOutputToChat(`💡 ${explanation}`, 'explanation')
        }
      }
    } catch (e) { console.debug('[WSBridge] execution error:', e.message) }
      // 自动解释失败，静默处理
    }
  }

  /**
   * Agent 在对话中生成文件后，自动保存并打开
   */
  async function agentCreateFile(fileName, content) {
    try {
      const r = await authFetch(`${API_BASE}/api/workspace/file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: fileName, content }),
      })

      if (r.ok) {
        openFileInIDE(fileName)
        sendTerminalOutputToChat(`📄 已创建文件: ${fileName}`, 'info')
        return true
      }
    } catch (e) {
      sendTerminalOutputToChat(`❌ 创建文件失败: ${e.message}`, 'error')
    }
    return false
  }

  return {
    pendingFileOpen,
    pendingCommand,
    terminalOutputBuffer,
    registerChatCallback,
    openFileInIDE,
    runCommandInTerminal,
    sendTerminalOutputToChat,
    sendErrorToChat,
    agentCreateFile,
  }
}
