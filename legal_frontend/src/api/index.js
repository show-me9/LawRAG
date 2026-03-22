import axios from 'axios'
import { ElMessage } from 'element-plus'

const http = axios.create({
  baseURL: '/api',
  timeout: 60000,   // 问答请求可能较慢，设 60s
})

// 响应拦截：统一错误提示
http.interceptors.response.use(
  res => res.data,
  err => {
    const msg = err.response?.data?.detail || err.message || '请求失败'
    ElMessage.error(msg)
    return Promise.reject(err)
  }
)

// ── 健康检查 ──────────────────────────────────────────────
export const getHealth = () => http.get('/health')

// ── 文档管理 ──────────────────────────────────────────────
export const getDocuments  = ()         => http.get('/documents')
export const uploadDocument = (formData) => http.post('/documents/upload', formData, {
  headers: { 'Content-Type': 'multipart/form-data' },
  timeout: 300000,  // 上传+摄入最长 5 分钟
})
export const getIngestStatus = (filename) =>
  http.get(`/documents/upload/status/${encodeURIComponent(filename)}`)
export const deleteDocument = (filename) =>
  http.delete(`/documents/${encodeURIComponent(filename)}`)

// ── 智能问答 ──────────────────────────────────────────────
export const sendChat   = (question) => http.post('/chat', { question })
export const clearChat  = ()         => http.post('/chat/clear')

// ── 设置 ─────────────────────────────────────────────────
export const getSettings    = ()       => http.get('/settings')
export const updateSettings = (data)   => http.post('/settings', data)
export const testConnection = ()       => http.get('/settings/test-connection')
