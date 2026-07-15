/**
 * 聊天文件上传 composable
 * 从 ChatView.vue 中提取的文件上传/粘贴/附件管理逻辑
 */
import { ref, reactive } from 'vue'
import { API_BASE, authFetch } from './apiClient.js'

const IMAGE_EXTS = ['png','jpg','jpeg','bmp','gif','webp','tiff','tif','svg']
const AUDIO_EXTS = ['wav','mp3','ogg','flac','m4a','aac','webm']
const VIDEO_EXTS = ['mp4','avi','mov','mkv','webm']

function inferModality(file) {
  const ext = (file.name || '').split('.').pop()?.toLowerCase() || ''
  if ((file.type || '').startsWith('image/') || IMAGE_EXTS.includes(ext)) return 'image'
  if ((file.type || '').startsWith('audio/') || AUDIO_EXTS.includes(ext)) return 'audio'
  if ((file.type || '').startsWith('video/') || VIDEO_EXTS.includes(ext)) return 'video'
  return 'file'
}

export function useChatUpload() {
  const chatFileInput = ref(null)
  const chatAttachments = reactive([])

  const onChatFileSelect = async (e) => {
    const files = Array.from(e.target.files || [])
    if (e.target) e.target.value = ''
    await uploadChatFiles(files)
  }

  const onPaste = async (e) => {
    const items = Array.from(e.clipboardData?.items || [])
    const imageFiles = []
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        const blob = item.getAsFile()
        if (blob) {
          const ext = item.type.split('/')[1] || 'png'
          const file = new File([blob], `paste_${Date.now()}.${ext}`, { type: item.type })
          imageFiles.push(file)
        }
      }
    }
    if (imageFiles.length) {
      e.preventDefault()
      await uploadChatFiles(imageFiles)
    }
  }

  const uploadChatFiles = async (files) => {
    for (const file of files) {
      const modality = inferModality(file)
      const att = reactive({
        name: file.name,
        type: file.type || '',
        modality,
        isImage: modality === 'image',
        isAudio: modality === 'audio',
        isVideo: modality === 'video',
        previewUrl: ['image', 'audio', 'video'].includes(modality) ? URL.createObjectURL(file) : '',
        publicUrl: '',
        savedPath: '',
        parsedText: '',
        uploading: true,
      })
      chatAttachments.push(att)
      try {
        const formData = new FormData()
        formData.append('file', file)
        const res = await authFetch(`${API_BASE}/api/chat/upload`, { method: 'POST', body: formData })
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || '上传失败')
        const data = await res.json()
        att.parsedText = data.parsed_text || ''
        att.type = data.type || att.type
        if (['image', 'audio', 'video'].includes(modality)) {
          try {
            const mediaForm = new FormData()
            mediaForm.append('file', file)
            const mediaRes = await authFetch(`${API_BASE}/api/taiji/upload`, { method: 'POST', body: mediaForm })
            if (mediaRes.ok) {
              const mediaData = await mediaRes.json()
              att.savedPath = mediaData.saved_path || ''
              att.publicUrl = mediaData.public_url || ''
              att.modality = mediaData.modality || modality
            }
          } catch (e) { console.warn('[Upload] 媒体上传失败:', e.message) }
        }
        att.uploading = false
      } catch (err) {
        att.parsedText = `[上传失败: ${err.message}]`
        att.uploading = false
      }
    }
  }

  const removeChatAttachment = (idx) => {
    const att = chatAttachments[idx]
    if (att?.previewUrl) URL.revokeObjectURL(att.previewUrl)
    chatAttachments.splice(idx, 1)
  }

  const clearAttachments = () => {
    chatAttachments.forEach(att => {
      if (att.previewUrl) URL.revokeObjectURL(att.previewUrl)
    })
    chatAttachments.splice(0, chatAttachments.length)
  }

  const detachAttachments = () => {
    chatAttachments.forEach(att => {
      if (att.previewUrl) URL.revokeObjectURL(att.previewUrl)
    })
    chatAttachments.splice(0, chatAttachments.length)
  }

  return {
    chatFileInput,
    chatAttachments,
    onChatFileSelect,
    onPaste,
    uploadChatFiles,
    removeChatAttachment,
    clearAttachments,
    detachAttachments,
  }
}
