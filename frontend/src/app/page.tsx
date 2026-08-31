'use client';

import { useEffect, useState } from 'react';
import AppShell from '@/components/layout/AppShell';
import AssistantView from '@/components/assistant/AssistantView';
import CompareView from '@/components/compare/CompareView';
import DashboardView from '@/components/dashboard/DashboardView';
import DataExplorer from '@/components/data/DataExplorer';
import HistoryView from '@/components/history/HistoryView';
import { getHealth, getSuggestions } from '@/lib/api';
import { addHistory, clearHistory, readHistory, removeHistory } from '@/lib/storage';
import type { HealthResponse, HistoryRecord, WorkspaceView } from '@/types/chat';

export default function HomePage() {
  const [active, setActive] = useState<WorkspaceView>('assistant');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [online, setOnline] = useState<boolean | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [draft, setDraft] = useState('');
  const [history, setHistory] = useState<HistoryRecord[]>([]);

  useEffect(() => {
    setHistory(readHistory());
    const controller = new AbortController();
    Promise.all([getHealth(controller.signal), getSuggestions(controller.signal)])
      .then(([healthData, suggestionData]) => { setHealth(healthData); setSuggestions(suggestionData); setOnline(true); })
      .catch(() => setOnline(false));
    return () => controller.abort();
  }, []);

  function askAI(prompt: string) { setDraft(prompt); setActive('assistant'); }
  function recordHistory(record: HistoryRecord) { setHistory(addHistory(record)); }

  return <AppShell active={active} onNavigate={setActive} health={health} online={online}>
    <div className={active === 'assistant' ? 'view-panel active' : 'view-panel'} aria-hidden={active !== 'assistant'}>
      <AssistantView online={online} health={health} suggestions={suggestions} draft={draft} onDraftChange={setDraft} onOnlineChange={setOnline} onHistoryAdd={recordHistory} />
    </div>
    <div className={active === 'compare' ? 'view-panel active' : 'view-panel'} aria-hidden={active !== 'compare'}><CompareView onAskAI={askAI} /></div>
    <div className={active === 'dashboard' ? 'view-panel active' : 'view-panel'} aria-hidden={active !== 'dashboard'}><DashboardView onAskAI={askAI} /></div>
    <div className={active === 'data' ? 'view-panel active' : 'view-panel'} aria-hidden={active !== 'data'}><DataExplorer onAskAI={askAI} /></div>
    <div className={active === 'history' ? 'view-panel active' : 'view-panel'} aria-hidden={active !== 'history'}><HistoryView records={history} onOpen={askAI} onDelete={id => setHistory(removeHistory(id))} onClear={() => { clearHistory(); setHistory([]); }} /></div>
  </AppShell>;
}
