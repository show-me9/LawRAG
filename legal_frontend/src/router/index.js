import { createRouter, createWebHistory } from 'vue-router'
import ChatView      from '@/views/ChatView.vue'
import DocumentsView from '@/views/DocumentsView.vue'
import SettingsView  from '@/views/SettingsView.vue'

const routes = [
  { path: '/',          redirect: '/chat' },
  { path: '/chat',      component: ChatView,      meta: { title: '智能问答' } },
  { path: '/documents', component: DocumentsView, meta: { title: '文档管理' } },
  { path: '/settings',  component: SettingsView,  meta: { title: '系统设置' } },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.afterEach(to => {
  document.title = `${to.meta.title || ''} — 法律智能问答系统`
})

export default router
