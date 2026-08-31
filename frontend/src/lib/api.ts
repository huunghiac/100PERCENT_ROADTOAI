import type { ChatApiResponse, HealthResponse } from '@/types/chat';

export const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export class ApiError extends Error {
  constructor(message: string, public readonly status?: number) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init: RequestInit = {}, signal?: AbortSignal): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 15_000);
  const abort = () => controller.abort();
  signal?.addEventListener('abort', abort, { once: true });
  try {
    const response = await fetch(`${API_URL}${path}`, { ...init, signal: controller.signal });
    if (!response.ok) {
      let detail = `Yêu cầu thất bại (${response.status})`;
      try {
        const payload = await response.json() as { detail?: string };
        if (payload.detail) detail = payload.detail;
      } catch { /* Keep the HTTP fallback. */ }
      throw new ApiError(detail, response.status);
    }
    return response.json() as Promise<T>;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if ((error as Error).name === 'AbortError') throw new ApiError('Hệ thống phản hồi quá thời gian.');
    throw new ApiError('Không thể kết nối hệ thống phân tích.');
  } finally {
    window.clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> { return request<HealthResponse>('/api/health', {}, signal); }
export function getSuggestions(signal?: AbortSignal): Promise<string[]> { return request<string[]>('/api/suggestions', {}, signal); }
export function askQuestion(question: string, signal?: AbortSignal): Promise<ChatApiResponse> {
  return request<ChatApiResponse>('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, mode: 'auto' }) }, signal);
}
