import { createApp } from 'vue'
import { createPinia } from 'pinia'
import router from './router'
import App from './App.vue'
import { useRuntimeStore } from './stores/runtimeStore.js'

const app = createApp(App)
const pinia = createPinia()
app.use(pinia)
app.use(router)

app.config.errorHandler = (err, instance, info) => {
  console.error('[Vue Error]', err, info)
  try {
    useRuntimeStore().addException('error', '界面运行异常', {
      message: err?.message || String(err),
      context: info,
    }, {
      impact: '当前页面可能无法响应点击或加载内容',
      recovery: '请刷新页面，或回到对话后重试',
    })
  } catch (e) { console.debug('[main] error handler:', e.message) }
}

window.addEventListener('unhandledrejection', (e) => {
  console.error('[Unhandled Rejection]', e.reason)
  try {
    const reason = e.reason
    useRuntimeStore().addException('error', '后台任务异常', {
      message: reason?.message || String(reason),
    }, {
      impact: '部分功能可能没有完成',
      recovery: '请稍后重试或检查运行时状态',
    })
  } catch (e) { console.debug('[main] error handler:', e.message) }
})

app.mount('#app')
