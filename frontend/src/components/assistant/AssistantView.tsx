'use client';

import { useEffect, useRef, useState } from 'react';
import { Bot, Clipboard, CloudOff, MessageSquarePlus, RotateCcw, Search, Sparkles } from 'lucide-react';
import BotMessageCard from '@/components/BotMessageCard';
import AssistantLoading from '@/components/assistant/AssistantLoading';
import ChatComposer from '@/components/assistant/ChatComposer';
import { askQuestion, ApiError } from '@/lib/api';
import { createDemoAnswer, FALLBACK_SUGGESTIONS } from '@/data/demo';
import type { ChatMessage, HealthResponse, HistoryRecord } from '@/types/chat';

interface Props {
  online: boolean | null;
  health: HealthResponse | null;
  suggestions: string[];
  draft: string;
  onDraftChange: (value: string) => void;
  onOnlineChange: (value: boolean) => void;
  onHistoryAdd: (record: HistoryRecord) => void;
}

const wait = (milliseconds: number) => new Promise(resolve => window.setTimeout(resolve, milliseconds));

export default function AssistantView({ online, health, suggestions, draft, onDraftChange, onOnlineChange, onHistoryAdd }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const visibleSuggestions = suggestions.length ? suggestions : FALLBACK_SUGGESTIONS;

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }, [messages, loading]);
  useEffect(() => { if (online) setDemoMode(false); }, [online]);

  function newConversation() { setMessages([]); onDraftChange(''); }

  async function send(questionOverride?: string) {
    const question = (questionOverride ?? draft).trim();
    if (!question || loading) return;
    const userMessage: ChatMessage = { id: `${Date.now()}-user`, sender: 'user', text: question, createdAt: new Date().toISOString() };
    setMessages(previous => [...previous, userMessage]); onDraftChange(''); setLoading(true);
    try {
      const payload = demoMode
        ? (await wait(850), { status: 'success' as const, data: createDemoAnswer(question) })
        : await askQuestion(question);
      const createdAt = new Date().toISOString();
      setMessages(previous => [...previous, { id: `${Date.now()}-bot`, sender: 'bot', data: payload.data, status: payload.status, isDemo: demoMode, createdAt }]);
      if (!demoMode) onOnlineChange(true);
      const citation = payload.data.citations[0];
      onHistoryAdd({ id: `${Date.now()}-${Math.random().toString(16).slice(2)}`, question, createdAt, formattedAnswer: payload.data.formatted_answer, ticker: citation?.company, year: citation?.year, status: payload.status, isDemo: demoMode });
    } catch (error) {
      const message = error instanceof ApiError ? error.message : 'Không thể kết nối hệ thống phân tích.';
      const createdAt = new Date().toISOString();
      onOnlineChange(false);
      setMessages(previous => [...previous, { id: `${Date.now()}-error`, sender: 'bot', status: 'error', error: message, retryQuestion: question, createdAt }]);
      onHistoryAdd({ id: `${Date.now()}-error-history`, question, createdAt, status: 'error' });
    } finally { setLoading(false); }
  }

  return <section className="screen assistant-screen">
    <header className="screen-header assistant-header"><div><span className="screen-eyebrow">AI Financial Research Assistant</span><h1>Trợ lý AI</h1><p>Hỏi và phân tích Báo cáo tài chính bằng ngôn ngữ tự nhiên.</p></div>
      <div className="header-actions"><span className={`connection-pill ${online === false ? 'offline' : ''}`}><i />{online === null ? 'Đang kiểm tra' : online ? `${health?.available_tickers ?? 0} doanh nghiệp` : demoMode ? 'Chế độ demo' : 'Backend offline'}</span><button className="secondary-button" onClick={newConversation}><MessageSquarePlus size={16} />Cuộc trò chuyện mới</button></div>
    </header>

    {online === false && <div className="offline-banner"><CloudOff size={19} /><div><strong>Backend chưa kết nối</strong><p>Bạn vẫn có thể xem đầy đủ luồng giao diện bằng dữ liệu minh hoạ.</p></div><button onClick={() => setDemoMode(true)}>{demoMode ? 'Đang ở chế độ demo' : 'Xem giao diện demo'}</button></div>}

    <div className="assistant-workspace">
      <div className="message-list" aria-live="polite">
        {messages.length === 0 && <div className="welcome-state">
          <div className="welcome-icon"><Sparkles size={23} /></div><h2>Bạn muốn phân tích gì hôm nay?</h2><p>Tra cứu số liệu, tính chỉ số hoặc so sánh doanh nghiệp từ Báo cáo tài chính.</p>
          <div className="suggestion-grid">{visibleSuggestions.slice(0, 5).map((suggestion, index) => <button key={`${suggestion}-${index}`} onClick={() => void send(suggestion)}><span>{index + 1 < 10 ? `0${index + 1}` : index + 1}</span><p>{suggestion}</p><Search size={16} /></button>)}</div>
          <div className="trust-note"><Bot size={15} />Mỗi câu trả lời đều kèm bảng nguồn và truy vấn Pandas để kiểm chứng.</div>
        </div>}
        {messages.map(message => <div key={message.id} className={`chat-row ${message.sender}`}>
          {message.sender === 'bot' && <div className="assistant-avatar"><Bot size={16} /></div>}
          {message.sender === 'user' && <div className="user-message"><p>{message.text}</p><button onClick={() => void navigator.clipboard.writeText(message.text || '')} aria-label="Sao chép câu hỏi"><Clipboard size={13} /></button></div>}
          {message.data && <BotMessageCard data={message.data} status={message.status} isDemo={message.isDemo} />}
          {message.status === 'error' && <div className="api-error-card"><CloudOff size={20} /><div><strong>Không thể kết nối hệ thống phân tích</strong><p>{message.error}</p><button onClick={() => void send(message.retryQuestion)}><RotateCcw size={14} />Thử lại</button></div></div>}
        </div>)}
        {loading && <div className="chat-row bot"><div className="assistant-avatar"><Bot size={16} /></div><AssistantLoading /></div>}
        <div ref={endRef} />
      </div>
      <ChatComposer value={draft} onChange={onDraftChange} onSend={() => void send()} disabled={loading} />
    </div>
  </section>;
}
