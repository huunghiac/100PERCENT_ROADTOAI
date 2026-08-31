import type { HistoryRecord } from '@/types/chat';

const HISTORY_KEY = 'vifinqa.history.v1';
const SIDEBAR_KEY = 'vifinqa.sidebar.collapsed';
const canUseStorage = () => typeof window !== 'undefined' && Boolean(window.localStorage);

export function readHistory(): HistoryRecord[] {
  if (!canUseStorage()) return [];
  try {
    const value = JSON.parse(window.localStorage.getItem(HISTORY_KEY) || '[]') as unknown;
    return Array.isArray(value) ? value.filter((item): item is HistoryRecord => Boolean(item && typeof item === 'object' && 'id' in item && 'question' in item)) : [];
  } catch { return []; }
}

export function saveHistory(records: HistoryRecord[]): void { if (canUseStorage()) window.localStorage.setItem(HISTORY_KEY, JSON.stringify(records.slice(0, 50))); }
export function addHistory(record: HistoryRecord): HistoryRecord[] {
  const next = [record, ...readHistory().filter(item => item.id !== record.id)].slice(0, 50);
  saveHistory(next); return next;
}
export function removeHistory(id: string): HistoryRecord[] { const next = readHistory().filter(item => item.id !== id); saveHistory(next); return next; }
export function clearHistory(): void { if (canUseStorage()) window.localStorage.removeItem(HISTORY_KEY); }
export function readSidebarCollapsed(): boolean { return canUseStorage() && window.localStorage.getItem(SIDEBAR_KEY) === 'true'; }
export function saveSidebarCollapsed(value: boolean): void { if (canUseStorage()) window.localStorage.setItem(SIDEBAR_KEY, String(value)); }
