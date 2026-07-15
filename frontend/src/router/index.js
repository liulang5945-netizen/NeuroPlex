import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'chat',
    component: () => import('@/components/ChatView.vue'),
    meta: { title: '聊天' }
  },
  {
    path: '/kb',
    name: 'kb',
    component: () => import('@/views/KBView.vue'),
    meta: { title: '知识库管理' }
  },
  {
    path: '/train',
    name: 'train',
    component: () => import('@/views/TrainingView.vue'),
    meta: { title: '微调训练' }
  },
  {
    path: '/agent',
    name: 'agent',
    component: () => import('@/views/AgentConfigView.vue'),
    meta: { title: '智能体配置' }
  },
  {
    path: '/workspace',
    name: 'workspace',
    component: () => import('@/views/WorkspaceView.vue'),
    meta: { title: 'IDE 工作区' }
  },
  {
    path: '/life',
    name: 'life',
    component: () => import('@/views/LifeStatusView.vue'),
    meta: { title: '生命状态' }
  },
  {
    path: '/settings',
    name: 'settings',
    component: () => import('@/views/SettingsView.vue'),
    meta: { title: '系统设置' }
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'not-found',
    redirect: '/'
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

router.afterEach((to) => {
  const title = to.meta?.title
  if (title) {
    document.title = `态极 · ${title}`
  }
})

export default router
