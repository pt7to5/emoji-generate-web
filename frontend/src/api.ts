export type MaskCandidate = {
  id: string
  maskUrl: string
  previewUrl: string
  bbox: [number, number, number, number]
  score?: number
  label?: string
}

export type UploadResult = {
  imageId: string
  imageUrl: string
  width: number
  height: number
  objects: MaskCandidate[]
  detectionWarning?: string
}

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
const apiUrl = (path: string) => `${API_BASE}${path}`

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: '请求失败' }))
    throw new Error(body.detail || '请求失败')
  }
  return response.json() as Promise<T>
}

export async function uploadImage(file: File, signal?: AbortSignal): Promise<UploadResult> {
  const data = new FormData()
  data.append('file', file)
  return json(await fetch(apiUrl('/api/images'), { method: 'POST', body: data, signal }))
}

export async function detectImage(imageId: string) {
  return json<UploadResult>(await fetch(apiUrl(`/api/images/${imageId}/detect`), { method: 'POST' }))
}

export async function refineMask(imageId: string, box: [number, number, number, number]) {
  return json<MaskCandidate>(await fetch(apiUrl(`/api/images/${imageId}/refine`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ box }),
  }))
}

export async function refinePaint(imageId: string, points: [number, number][], radius: number) {
  return json<MaskCandidate>(await fetch(apiUrl(`/api/images/${imageId}/refine-paint`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ points, radius }),
  }))
}

export async function editMask(imageId: string, objectId: string, points: [number, number, number][], radius: number, mode: 'add' | 'remove') {
  return json<MaskCandidate>(await fetch(apiUrl(`/api/images/${imageId}/masks/${objectId}/edit`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ points, radius, mode }),
  }))
}

export async function preciseEditMask(imageId: string, objectId: string, points: [number, number, number][], radius: number, mode: 'add' | 'remove') {
  return json<MaskCandidate>(await fetch(apiUrl(`/api/images/${imageId}/masks/${objectId}/precise-edit`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ points, radius, mode }),
  }))
}

export async function generateEmoji(imageId: string, objectId: string) {
  return json<{ emojiUrl: string; elapsedMs: number }>(await fetch(apiUrl('/api/emojis'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageId, objectId }),
  }))
}

export async function composeImage(imageId: string, objectId: string, emojiUrl: string) {
  return json<{ resultUrl: string; repairedUrl: string; elapsedMs: number }>(await fetch(apiUrl('/api/compositions'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageId, objectId, emojiUrl }),
  }))
}

export async function composeBatch(imageId: string, items: { objectId: string; emojiUrl: string }[]) {
  return json<{ resultUrl: string; elapsedMs: number }>(await fetch(apiUrl('/api/compositions/batch'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageId, items }),
  }))
}
