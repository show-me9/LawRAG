<template>
  <div class="chat-view">
    <!-- 顶部栏 -->
    <div class="chat-header">
      <span class="page-title">智能问答</span>
      <el-button
        size="small" plain
        :icon="Delete"
        @click="handleClear"
      >清空对话</el-button>
    </div>

    <!-- 消息列表 -->
    <div class="message-list" ref="listRef">
      <!-- 空状态 -->
      <div v-if="messages.length === 0" class="empty-state">
        <el-icon class="empty-icon"><ChatLineRound /></el-icon>
        <p class="empty-title">您好，请输入法律问题</p>
        <p class="empty-sub">系统将基于已摄入的法律文档为您解答，并标注引用条文</p>
      </div>

      <!-- 消息气泡 -->
      <template v-for="(msg, idx) in messages" :key="idx">
        <!-- 用户消息 -->
        <div v-if="msg.role === 'user'" class="msg-row msg-user">
          <div class="bubble bubble-user">{{ msg.content }}</div>
        </div>

        <!-- 助手消息 -->
        <div v-else class="msg-row msg-assistant">
          <div class="avatar-icon">
            <el-icon><Memo /></el-icon>
          </div>
          <div class="bubble-wrap">
            <div class="bubble bubble-assistant" v-html="renderAnswer(msg.content)"></div>

            <!-- 引用条文 -->
            <div v-if="msg.sources?.length" class="sources">
              <div class="sources-title">
                <el-icon><Document /></el-icon>
                引用条文
              </div>
              <div
                v-for="(src, si) in msg.sources"
                :key="si"
                class="source-item"
              >
                <span class="source-tag">[{{ si + 1 }}]</span>
                <span class="source-citation">{{ src.citation }}</span>
                <el-tooltip :content="src.snippet" placement="top" :show-after="300">
                  <el-icon class="source-eye"><View /></el-icon>
                </el-tooltip>
              </div>
            </div>

            <!-- 改写查询（可折叠） -->
            <el-collapse-transition>
              <div v-if="msg.rewritten_query && msg.rewritten_query !== messages[idx-1]?.content" class="rewritten">
                <el-icon><Search /></el-icon>
                检索查询：{{ msg.rewritten_query }}
              </div>
            </el-collapse-transition>
          </div>
        </div>
      </template>

      <!-- 加载中 -->
      <div v-if="loading" class="msg-row msg-assistant">
        <div class="avatar-icon">
          <el-icon><Memo /></el-icon>
        </div>
        <div class="bubble bubble-assistant bubble-loading">
          <span class="dot"></span>
          <span class="dot"></span>
          <span class="dot"></span>
        </div>
      </div>
    </div>

    <!-- 输入区 -->
    <div class="input-area">
      <el-input
        v-model="inputText"
        type="textarea"
        :rows="3"
        placeholder="请输入您的法律问题，按 Ctrl+Enter 发送..."
        resize="none"
        :disabled="loading"
        @keydown.ctrl.enter="handleSend"
      />
      <el-button
        type="primary"
        :icon="Promotion"
        :loading="loading"
        :disabled="!inputText.trim()"
        @click="handleSend"
        class="send-btn"
      >发送</el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Delete, Promotion } from '@element-plus/icons-vue'
import { sendChat, clearChat } from '@/api'

const messages  = ref([])
const inputText = ref('')
const loading   = ref(false)
const listRef   = ref(null)

// 简单将换行转为 <br>，保留基本格式
function renderAnswer(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/【(.+?)】/g, '<span class="cite-tag">【$1】</span>')
    .replace(/\n/g, '<br>')
}

async function scrollToBottom() {
  await nextTick()
  if (listRef.value) {
    listRef.value.scrollTop = listRef.value.scrollHeight
  }
}

async function handleSend() {
  const q = inputText.value.trim()
  if (!q || loading.value) return

  messages.value.push({ role: 'user', content: q })
  inputText.value = ''
  loading.value = true
  await scrollToBottom()

  try {
    const res = await sendChat(q)
    messages.value.push({
      role:            'assistant',
      content:         res.answer,
      sources:         res.sources,
      rewritten_query: res.rewritten_query,
    })
  } catch {
    messages.value.push({
      role:    'assistant',
      content: '抱歉，请求失败，请检查服务状态后重试。',
      sources: [],
    })
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}

async function handleClear() {
  if (messages.value.length === 0) return
  await ElMessageBox.confirm('确认清空当前对话记录？', '提示', {
    confirmButtonText: '确认',
    cancelButtonText:  '取消',
    type: 'warning',
  })
  await clearChat()
  messages.value = []
  ElMessage.success('对话已清空')
}
</script>

<style scoped>
.chat-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--color-bg);
}

/* 顶部栏 */
.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 24px;
  background: var(--color-bg-card);
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}
.page-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--color-text-primary);
}

/* 消息列表 */
.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* 空状态 */
.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--color-text-hint);
  padding: 60px 0;
}
.empty-icon { font-size: 48px; color: #c0c4cc; }
.empty-title { font-size: 15px; color: var(--color-text-second); }
.empty-sub   { font-size: 13px; }

/* 消息行 */
.msg-row {
  display: flex;
  gap: 12px;
  max-width: 820px;
}
.msg-user {
  align-self: flex-end;
  flex-direction: row-reverse;
}
.msg-assistant { align-self: flex-start; }

/* 头像 */
.avatar-icon {
  width: 34px; height: 34px;
  border-radius: 8px;
  background: var(--color-primary);
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  flex-shrink: 0;
  margin-top: 2px;
}

/* 气泡 */
.bubble {
  padding: 12px 16px;
  border-radius: 12px;
  line-height: 1.7;
  font-size: 14px;
  word-break: break-word;
}
.bubble-user {
  background: var(--color-primary);
  color: #fff;
  border-bottom-right-radius: 4px;
}
.bubble-assistant {
  background: var(--color-bg-card);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  border-bottom-left-radius: 4px;
  box-shadow: var(--shadow-sm);
}

/* 引用高亮 */
:deep(.cite-tag) {
  color: var(--color-primary-light);
  font-weight: 500;
}

/* 加载动画 */
.bubble-loading {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 14px 18px;
}
.dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: #c0c4cc;
  animation: bounce .9s infinite;
}
.dot:nth-child(2) { animation-delay: .2s; }
.dot:nth-child(3) { animation-delay: .4s; }
@keyframes bounce {
  0%,60%,100% { transform: translateY(0); }
  30%          { transform: translateY(-6px); }
}

/* 引用条文 */
.bubble-wrap { display: flex; flex-direction: column; gap: 8px; }
.sources {
  background: #f8f9fb;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.sources-title {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-second);
  margin-bottom: 2px;
}
.source-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--color-text-second);
}
.source-tag {
  color: var(--color-primary-light);
  font-weight: 600;
  flex-shrink: 0;
}
.source-citation { flex: 1; }
.source-eye { cursor: pointer; color: #c0c4cc; }
.source-eye:hover { color: var(--color-primary-light); }

/* 改写查询 */
.rewritten {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  color: var(--color-text-hint);
  padding: 4px 0;
}

/* 输入区 */
.input-area {
  padding: 16px 24px;
  background: var(--color-bg-card);
  border-top: 1px solid var(--color-border);
  display: flex;
  gap: 12px;
  align-items: flex-end;
  flex-shrink: 0;
}
.input-area :deep(.el-textarea__inner) {
  border-radius: var(--radius);
  font-size: 14px;
  line-height: 1.6;
  resize: none;
}
.send-btn {
  height: 72px;
  padding: 0 22px;
  flex-shrink: 0;
  font-size: 14px;
}
</style>
