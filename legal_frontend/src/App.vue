<template>
  <div class="app-layout">
    <!-- 侧边栏 -->
    <aside class="sidebar">
      <div class="sidebar-logo">
        <el-icon class="logo-icon"><Scales /></el-icon>
        <span class="logo-text">法律问答</span>
      </div>

      <nav class="sidebar-nav">
        <router-link
          v-for="item in navItems"
          :key="item.path"
          :to="item.path"
          class="nav-item"
          :class="{ active: $route.path === item.path }"
        >
          <el-icon><component :is="item.icon" /></el-icon>
          <span>{{ item.label }}</span>
        </router-link>
      </nav>

      <!-- 服务状态指示 -->
      <div class="sidebar-status">
        <span class="status-dot" :class="statusClass"></span>
        <span class="status-text">{{ statusText }}</span>
      </div>
    </aside>

    <!-- 主内容区 -->
    <main class="main-content">
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { getHealth } from '@/api'

const navItems = [
  { path: '/chat',      label: '智能问答', icon: 'ChatLineRound' },
  { path: '/documents', label: '文档管理', icon: 'FolderOpened'  },
  { path: '/settings',  label: '系统设置', icon: 'Setting'       },
]

const health = ref(null)

const statusClass = computed(() => {
  if (!health.value) return 'unknown'
  return health.value.status === 'ok' ? 'ok' : 'degraded'
})

const statusText = computed(() => {
  if (!health.value) return '检查中...'
  return health.value.status === 'ok' ? '服务正常' : '配置未完成'
})

onMounted(async () => {
  try { health.value = await getHealth() } catch {}
})
</script>

<style scoped>
.app-layout {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

/* ── 侧边栏 ── */
.sidebar {
  width: var(--sidebar-width);
  background: var(--color-primary);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  padding: 0 0 16px;
}

.sidebar-logo {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 20px 20px 24px;
  border-bottom: 1px solid rgba(255,255,255,.1);
}
.logo-icon {
  font-size: 22px;
  color: var(--color-accent);
}
.logo-text {
  font-size: 16px;
  font-weight: 600;
  color: #fff;
  letter-spacing: .5px;
}

.sidebar-nav {
  flex: 1;
  padding: 16px 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: var(--radius);
  color: rgba(255,255,255,.65);
  text-decoration: none;
  font-size: 14px;
  transition: all .2s;
}
.nav-item:hover  { background: rgba(255,255,255,.1); color: #fff; }
.nav-item.active { background: rgba(255,255,255,.15); color: #fff; font-weight: 500; }
.nav-item .el-icon { font-size: 16px; flex-shrink: 0; }

/* 状态指示 */
.sidebar-status {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 20px;
  font-size: 12px;
  color: rgba(255,255,255,.45);
}
.status-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}
.status-dot.ok       { background: #52c41a; box-shadow: 0 0 6px #52c41a; }
.status-dot.degraded { background: #faad14; box-shadow: 0 0 6px #faad14; }
.status-dot.unknown  { background: #909399; }

/* ── 主内容 ── */
.main-content {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
</style>
