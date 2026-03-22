<template>
  <div class="doc-view">
    <!-- 顶部栏 -->
    <div class="page-header">
      <span class="page-title">文档管理</span>
      <el-button type="primary" :icon="Upload" @click="triggerUpload">
        上传文档
      </el-button>
      <input
        ref="fileInputRef"
        type="file"
        accept=".docx"
        multiple
        style="display:none"
        @change="handleFileChange"
      />
    </div>

    <!-- 上传进度区 -->
    <div v-if="uploadingFiles.length" class="upload-progress-area">
      <div
        v-for="item in uploadingFiles"
        :key="item.name"
        class="upload-progress-item"
      >
        <el-icon class="file-icon"><Document /></el-icon>
        <div class="progress-info">
          <div class="progress-name">{{ item.name }}</div>
          <el-progress
            :percentage="item.percent"
            :status="item.status === 'done' ? 'success' : item.status === 'failed' ? 'exception' : undefined"
            :striped="item.status === 'processing'"
            :striped-flow="item.status === 'processing'"
          />
          <div class="progress-msg">{{ item.message }}</div>
        </div>
      </div>
    </div>

    <!-- 文档列表 -->
    <div class="doc-content">
      <el-table
        :data="documents"
        v-loading="tableLoading"
        empty-text="暂无文档，请先上传"
        stripe
        style="width:100%"
      >
        <el-table-column label="文件名" prop="filename" min-width="200">
          <template #default="{ row }">
            <div class="filename-cell">
              <el-icon><Document /></el-icon>
              <span>{{ row.filename }}</span>
            </div>
          </template>
        </el-table-column>

        <el-table-column label="条文数量" prop="article_count" width="120" align="center">
          <template #default="{ row }">
            <el-tag type="info" size="small">{{ row.article_count }} 条</el-tag>
          </template>
        </el-table-column>

        <el-table-column label="状态" width="110" align="center">
          <template #default>
            <el-tag type="success" size="small">已摄入</el-tag>
          </template>
        </el-table-column>

        <el-table-column label="Hash" prop="file_hash" min-width="150">
          <template #default="{ row }">
            <span class="hash-text">{{ row.file_hash.slice(0, 12) }}...</span>
          </template>
        </el-table-column>

        <el-table-column label="操作" width="100" align="center">
          <template #default="{ row }">
            <el-popconfirm
              title="确认删除该文档及其所有向量数据？"
              confirm-button-text="删除"
              cancel-button-text="取消"
              @confirm="handleDelete(row.filename)"
            >
              <template #reference>
                <el-button type="danger" link size="small" :icon="Delete">
                  删除
                </el-button>
              </template>
            </el-popconfirm>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Upload, Delete } from '@element-plus/icons-vue'
import { getDocuments, uploadDocument, deleteDocument, getIngestStatus } from '@/api'

const documents     = ref([])
const tableLoading  = ref(false)
const fileInputRef  = ref(null)
const uploadingFiles = ref([])  // [{name, percent, status, message}]

async function fetchDocuments() {
  tableLoading.value = true
  try { documents.value = await getDocuments() }
  finally { tableLoading.value = false }
}

function triggerUpload() {
  fileInputRef.value?.click()
}

async function handleFileChange(e) {
  const files = Array.from(e.target.files)
  e.target.value = ''   // 重置，允许重复上传同名文件
  if (!files.length) return

  for (const file of files) {
    await uploadSingle(file)
  }
}

async function uploadSingle(file) {
  // 加入进度列表
  const item = { name: file.name, percent: 10, status: 'processing', message: '正在上传...' }
  uploadingFiles.value.push(item)

  try {
    const fd = new FormData()
    fd.append('file', file)
    const res = await uploadDocument(fd)

    if (res.skipped) {
      item.percent = 100
      item.status  = 'done'
      item.message = '文件未变化，已跳过'
      await fetchDocuments()
      return
    }

    // 轮询摄入进度
    item.message = '正在摄入...'
    item.percent = 30
    await pollIngestStatus(file.name, item)

  } catch (err) {
    item.status  = 'failed'
    item.percent = 100
    item.message = '上传失败'
  }
}

async function pollIngestStatus(filename, item) {
  const MAX_RETRY = 120   // 最多轮询 2 分钟
  let   retry     = 0

  return new Promise(resolve => {
    const timer = setInterval(async () => {
      retry++
      if (retry > MAX_RETRY) {
        clearInterval(timer)
        item.status  = 'failed'
        item.message = '摄入超时，请检查服务日志'
        resolve()
        return
      }

      try {
        const res = await getIngestStatus(filename)
        item.message = res.message

        if (res.status === 'done') {
          item.percent = 100
          item.status  = 'done'
          clearInterval(timer)
          ElMessage.success(`${filename} 摄入完成`)
          await fetchDocuments()
          resolve()
        } else if (res.status === 'failed') {
          item.percent = 100
          item.status  = 'failed'
          clearInterval(timer)
          resolve()
        } else {
          // 还在处理中，平滑推进进度条
          item.percent = Math.min(item.percent + 5, 90)
        }
      } catch { /* 忽略轮询中的网络抖动 */ }
    }, 1000)
  })
}

async function handleDelete(filename) {
  try {
    await deleteDocument(filename)
    ElMessage.success(`已删除：${filename}`)
    await fetchDocuments()
  } catch {}
}

onMounted(fetchDocuments)
</script>

<style scoped>
.doc-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--color-bg);
}

.page-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 24px;
  background: var(--color-bg-card);
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}
.page-title {
  font-size: 15px;
  font-weight: 600;
  flex: 1;
}

/* 上传进度区 */
.upload-progress-area {
  margin: 16px 24px 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.upload-progress-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  background: var(--color-bg-card);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  padding: 12px 16px;
}
.file-icon { font-size: 22px; color: var(--color-primary-light); margin-top: 2px; }
.progress-info { flex: 1; display: flex; flex-direction: column; gap: 4px; }
.progress-name { font-size: 13px; font-weight: 500; }
.progress-msg  { font-size: 12px; color: var(--color-text-hint); }

/* 文档列表 */
.doc-content {
  flex: 1;
  overflow: auto;
  padding: 20px 24px;
}
.filename-cell {
  display: flex;
  align-items: center;
  gap: 7px;
}
.filename-cell .el-icon { color: var(--color-primary-light); }
.hash-text {
  font-family: monospace;
  font-size: 12px;
  color: var(--color-text-hint);
}
</style>
