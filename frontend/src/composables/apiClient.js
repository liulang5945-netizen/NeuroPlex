function resolveApiBase() {
  if (typeof window === 'undefined') return ''

  const params = new URLSearchParams(window.location.search)
  const hashQuery = window.location.hash.includes('?')
    ? new URLSearchParams(window.location.hash.slice(window.location.hash.indexOf('?') + 1))
    : new URLSearchParams()

  if (params.get('taiji_client') === 'desktop' || hashQuery.get('taiji_client') === 'desktop') {
    return `${window.location.protocol}//${window.location.hostname}:8000`
  }

  if (import.meta.env.DEV) return ''

  const { protocol, hostname, port } = window.location
  if (port === '8000') return ''
  return `${protocol}//${hostname}:8000`
}

export const API_BASE = resolveApiBase()

export async function authFetch(url, options = {}) {
  const maxRetries = options.retries ?? 2
  const token = localStorage.getItem('jwt_token') || ''
  const headers = new Headers(options.headers || {})
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  let lastError = null
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, { ...options, headers })
      if (response.status === 401) {
        localStorage.removeItem('jwt_token')
        if (typeof window !== 'undefined') {
          window.dispatchEvent(new CustomEvent('taiji-auth-expired', {
            detail: { message: 'JWT token 缺失或已过期，请重新登录' },
          }))
        }
      }
      // 服务端错误 → 重试；客户端错误（4xx 除 401 外）→ 不重试
      if (response.status >= 500 && attempt < maxRetries) {
        lastError = new Error(`Server error HTTP ${response.status}`)
        await new Promise(r => setTimeout(r, (attempt + 1) * 500))
        continue
      }
      return response
    } catch (e) {
      lastError = e
      if (attempt < maxRetries) {
        await new Promise(r => setTimeout(r, (attempt + 1) * 500))
      }
    }
  }
  throw lastError || new Error('authFetch failed after retries')
}

/**
 * 带 Content-Type 校验的 JSON 请求封装。
 * 若响应非 JSON，抛出错误避免静默解析失败。
 */
export async function authFetchJSON(url, options = {}) {
  const response = await authFetch(url, options)
  const ctype = response.headers.get('content-type') || ''
  if (!ctype.includes('application/json')) {
    const text = await response.text().catch(() => '')
    throw new Error(`Expected JSON but got ${ctype || 'unknown content-type'}: ${text.slice(0, 200)}`)
  }
  return response.json()
}
