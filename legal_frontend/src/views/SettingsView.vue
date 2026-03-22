<template>
  <div class="settings-view">
    <div class="page-header">
      <span class="page-title">系统设置</span>
    </div>

    <div class="settings-body" v-loading="pageLoading">
      <!-- API 配置 -->
      <div class="settings-card">
        <div class="card-title">
          <el-icon><Key /></el-icon>
          API 配置
        </div>
        <el-form label-width="130px" label-position="left">
          <el-form-item label="SiliconFlow API Key">
            <el-input
              v-model="form.api_key"
              type="password"
              show-password
              placeholder="sk-xxxxxxxxxxxx（留空则不修改）"
              style="max-width:400px"
            />
            <el-button
              class="test-btn"
              :loading="testing"
              @click="handleTest"
            >测试连接</el-button>
            <el-tag
              v-if="testResult"
              :type="testResult.success ? 'success' : 'danger'"
              style="margin-left:8px"
            >{{ testResult.message }}</el-tag>
          </el-form-item>

          <el-form-item label="LLM 模型">
            <el-input v-model="form.llm_model" style="max-width:340px" />
          </el-form-item>

          <el-form-item label="Embedding 模型">
            <el-input v-model="form.embedding_model" style="max-width:340px" />
          </el-form-item>

          <el-form-item label="Reranker 模型">
            <el-input v-model="form.reranker_model" style="max-width:340px" />
          </el-form-item>
        </el-form>
      </div>

      <!-- 检索参数 -->
      <div class="settings-card">
        <div class="card-title">
          <el-icon><Setting /></el-icon>
          检索参数
        </div>
        <el-form label-width="130px" label-position="left">
          <el-form-item label="向量召回数">
            <el-input-number v-model="form.vector_search_k" :min="5" :max="50" />
            <span class="hint">向量检索初步召回的文档数量</span>
          </el-form-item>

          <el-form-item label="BM25 召回数">
            <el-input-number v-model="form.bm25_search_k" :min="5" :max="50" />
            <span class="hint">BM25 关键词检索召回的文档数量</span>
          </el-form-item>

          <el-form-item label="精排保留数">
            <el-input-number v-model="form.rerank_top_k" :min="1" :max="10" />
            <span class="hint">Reranker 最终保留的文档数量</span>
          </el-form-item>

          <el-form-item label="BM25 权重">
            <el-slider
              v-model="form.bm25_weight"
              :min="0" :max="1" :step="0.1"
              show-input
              style="max-width:360px"
            />
            <span class="hint">向量权重自动补足为 {{ (1 - form.bm25_weight).toFixed(1) }}</span>
          </el-form-item>
        </el-form>
      </div>

      <!-- 保存按钮 -->
      <div class="save-area">
        <el-button type="primary" :loading="saving" @click="handleSave">
          保存配置
        </el-button>
        <el-button @click="handleReset">重置</el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { getSettings, updateSettings, testConnection } from '@/api'

const pageLoading = ref(false)
const saving      = ref(false)
const testing     = ref(false)
const testResult  = ref(null)

const form = ref({
  api_key:         '',
  llm_model:       '',
  embedding_model: '',
  reranker_model:  '',
  vector_search_k: 20,
  bm25_search_k:   20,
  rerank_top_k:    5,
  bm25_weight:     0.4,
})

let originalForm = {}

async function fetchSettings() {
  pageLoading.value = true
  try {
    const data = await getSettings()
    form.value = {
      api_key:         '',   // 不回填明文
      llm_model:       data.llm_model,
      embedding_model: data.embedding_model,
      reranker_model:  data.reranker_model,
      vector_search_k: data.vector_search_k,
      bm25_search_k:   data.bm25_search_k,
      rerank_top_k:    data.rerank_top_k,
      bm25_weight:     data.bm25_weight,
    }
    originalForm = { ...form.value }
  } finally {
    pageLoading.value = false
  }
}

async function handleSave() {
  saving.value = true
  try {
    const payload = { ...form.value }
    payload.vector_weight = parseFloat((1 - payload.bm25_weight).toFixed(1))
    if (!payload.api_key) delete payload.api_key   // 留空则不修改
    await updateSettings(payload)
    ElMessage.success('配置已保存')
    originalForm = { ...form.value }
  } finally {
    saving.value = false
  }
}

function handleReset() {
  form.value = { ...originalForm }
  testResult.value = null
}

async function handleTest() {
  testing.value = true
  testResult.value = null
  try {
    testResult.value = await testConnection()
  } finally {
    testing.value = false
  }
}

onMounted(fetchSettings)
</script>

<style scoped>
.settings-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--color-bg);
}

.page-header {
  padding: 14px 24px;
  background: var(--color-bg-card);
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}
.page-title { font-size: 15px; font-weight: 600; }

.settings-body {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  max-width: 760px;
}

.settings-card {
  background: var(--color-bg-card);
  border: 1px solid var(--color-border);
  border-radius: var(--radius);
  padding: 20px 24px;
  box-shadow: var(--shadow-sm);
}
.card-title {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin-bottom: 18px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--color-border);
}
.card-title .el-icon { color: var(--color-primary-light); }

.test-btn { margin-left: 10px; }

.hint {
  margin-left: 10px;
  font-size: 12px;
  color: var(--color-text-hint);
}

.save-area {
  display: flex;
  gap: 10px;
  padding: 4px 0 16px;
}
</style>
