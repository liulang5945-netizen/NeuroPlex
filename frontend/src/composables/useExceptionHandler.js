/**
 * 运行时异常处理 composable
 * ==========================
 *
 * 统一处理所有运行时异常，将技术错误转为用户可理解的信息：
 * - 告诉用户发生什么
 * - 告诉用户影响什么
 * - 告诉用户怎么恢复
 *
 * 使用方式：
 *   const { handleError, handleApiError, handleWebSocketError } = useExceptionHandler()
 *   try { ... } catch (e) { handleError(e, '模型加载') }
 */
import { useRuntimeStore } from '@/stores/runtimeStore.js'

// 错误类型 → 用户友好描述映射
const ERROR_MAP = {
  // 网络错误
  'Failed to fetch': {
    title: '连接中断',
    message: '无法连接到后端服务',
    impact: '对话、工具调用、生命系统暂时不可用',
    recovery: '请检查后端是否运行，或点击"重新连接"',
  },
  'NetworkError': {
    title: '网络异常',
    message: '网络连接出现问题',
    impact: '部分功能可能不可用',
    recovery: '请检查网络连接后重试',
  },
  'ECONNREFUSED': {
    title: '服务未启动',
    message: '后端服务未运行或端口被占用',
    impact: '所有功能不可用',
    recovery: '请启动后端服务：运行 dev.bat 或 python -m uvicorn api.app:app',
  },

  // 认证错误
  'JWT token': {
    title: '认证过期',
    message: '登录凭证已过期',
    impact: '终端和部分 API 不可用',
    recovery: '请重新登录，或在设置中关闭认证',
  },
  '401': {
    title: '认证失败',
    message: '需要登录才能访问',
    impact: '受保护的功能不可用',
    recovery: '请登录后重试',
  },
  '403': {
    title: '权限不足',
    message: '没有访问权限',
    impact: '该功能不可用',
    recovery: '请联系管理员获取权限',
  },

  // 模型错误
  '模型未加载': {
    title: '模型未就绪',
    message: '态极的大脑还未装载',
    impact: '无法对话和调用工具',
    recovery: '等待自动装载，或在设置中手动加载模型',
  },
  'CUDA out of memory': {
    title: '显存不足',
    message: 'GPU 显存已满',
    impact: '模型推理可能失败',
    recovery: '请关闭其他占用显存的程序，或切换到 CPU 模式',
  },
  'OutOfMemoryError': {
    title: '内存不足',
    message: '系统内存已满',
    impact: '模型装载被延后',
    recovery: '请关闭其他程序释放内存，模型会在内存充足后自动装载',
  },

  // 工具错误
  'tool_error': {
    title: '工具执行失败',
    message: '某个工具调用出错',
    impact: '当前任务可能未完成',
    recovery: '态极会自动重试或换一种方法',
  },
  'search_failed': {
    title: '搜索失败',
    message: '无法完成网络搜索',
    impact: '无法获取最新信息',
    recovery: '请检查网络连接，或稍后重试',
  },

  // 终端错误
  'terminal_auth': {
    title: '终端认证失败',
    message: '终端连接需要认证',
    impact: '终端不可用',
    recovery: '请重新登录，或在设置中关闭终端认证',
  },
  'terminal_timeout': {
    title: '终端超时',
    message: '终端长时间无响应',
    impact: '终端已断开',
    recovery: '请点击"重连"按钮',
  },

  // 默认
  'unknown': {
    title: '未知错误',
    message: '遇到了一个意外问题',
    impact: '部分功能可能受影响',
    recovery: '请重试，如果问题持续请重启应用',
  },
}

export function useExceptionHandler() {

  /**
   * 处理通用错误
   */
  function handleError(error, context = '') {
    const runtimeStore = useRuntimeStore()
    const errorType = matchErrorType(error)
    const info = ERROR_MAP[errorType] || ERROR_MAP['unknown']

    runtimeStore.addException(
      'error',
      info.title,
      {
        message: info.message,
        context,
        technical: String(error),
      },
      {
        impact: info.impact,
        recovery: info.recovery,
      }
    )

    return info
  }

  /**
   * 处理 API 错误
   */
  function handleApiError(response, context = '') {
    const runtimeStore = useRuntimeStore()

    if (response.status === 401) {
      runtimeStore.reportAuthExpired()
      return handleError('401', context)
    }
    if (response.status === 403) {
      return handleError('403', context)
    }
    if (response.status >= 500) {
      return handleError('server_error', context)
    }

    return handleError(`HTTP ${response.status}`, context)
  }

  /**
   * 处理 WebSocket 错误
   */
  function handleWebSocketError(event, context = '') {
    const runtimeStore = useRuntimeStore()

    if (event.code === 4001) {
      runtimeStore.reportAuthExpired('终端认证失败')
      return handleError('terminal_auth', context)
    }
    if (event.code === 1006) {
      return handleError('ECONNREFUSED', context)
    }

    return handleError(event.reason || 'WebSocket error', context)
  }

  /**
   * 处理模型加载错误
   */
  function handleModelError(error, context = '') {
    const runtimeStore = useRuntimeStore()

    const info = handleError(error, context)

    // 更新模型状态
    runtimeStore.syncHealth('error', info.message, false)

    return info
  }

  /**
   * 匹配错误类型
   */
  function matchErrorType(error) {
    const errorStr = String(error).toLowerCase()

    for (const key of Object.keys(ERROR_MAP)) {
      if (key === 'unknown') continue
      if (errorStr.includes(key.toLowerCase())) {
        return key
      }
    }

    return 'unknown'
  }

  /**
   * 获取用户友好的错误信息
   */
  function getFriendlyError(error) {
    const errorType = matchErrorType(error)
    return ERROR_MAP[errorType] || ERROR_MAP['unknown']
  }

  return {
    handleError,
    handleApiError,
    handleWebSocketError,
    handleModelError,
    getFriendlyError,
    ERROR_MAP,
  }
}
