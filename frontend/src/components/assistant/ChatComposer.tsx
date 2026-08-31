'use client';

import { useEffect, useRef } from 'react';
import { ArrowUp, CornerDownLeft, Sparkles } from 'lucide-react';
import { CONTEXT_CHIPS } from '@/data/demo';

interface Props { value: string; onChange: (value: string) => void; onSend: () => void; disabled: boolean }

export default function ChatComposer({ value, onChange, onSend, disabled }: Props) {
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const element = ref.current; if (!element) return;
    element.style.height = '0px'; element.style.height = `${Math.min(element.scrollHeight, 152)}px`;
  }, [value]);
  return <div className="composer-region">
    <div className="context-chips" aria-label="Loại câu hỏi">
      {CONTEXT_CHIPS.map(item => <button key={item.label} onClick={() => { onChange(item.prompt); ref.current?.focus(); }} disabled={disabled}><Sparkles size={12} />{item.label}</button>)}
    </div>
    <div className="chat-composer">
      <textarea ref={ref} rows={1} value={value} onChange={event => onChange(event.target.value)} onKeyDown={event => {
        if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); if (value.trim() && !disabled) onSend(); }
      }} placeholder="Hỏi về doanh thu, lợi nhuận, tài sản, ROE, ROA..." aria-label="Câu hỏi tài chính" disabled={disabled} />
      <button className="send-button" onClick={onSend} disabled={disabled || !value.trim()} aria-label="Gửi câu hỏi"><ArrowUp size={18} /></button>
    </div>
    <p className="composer-help"><CornerDownLeft size={12} /> Enter để gửi · Shift + Enter để xuống dòng · Luôn đối chiếu báo cáo gốc</p>
  </div>;
}
