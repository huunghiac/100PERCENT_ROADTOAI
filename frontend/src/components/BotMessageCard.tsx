'use client';

import { useState, type ReactNode } from 'react';
import { AlertTriangle, Check, CheckCircle2, Clipboard, Database, FileSearch, Info, ShieldCheck } from 'lucide-react';
import type { ChatResponseData, ChatStatus } from '@/types/chat';
import CitationCard from '@/components/assistant/CitationCard';
import EvidenceTable from '@/components/assistant/EvidenceTable';
import QueryInspector from '@/components/assistant/QueryInspector';

interface Props { data: ChatResponseData; status?: ChatStatus; isDemo?: boolean }
type DetailTab = 'overview' | 'sources' | 'query';

function RichText({ value }: { value: string }) {
  const parts = value.split(/\*\*(.*?)\*\*/g);
  return <>{parts.map((part, index) => index % 2 ? <strong key={index}>{part}</strong> : <span key={index}>{part}</span>)}</>;
}

function confidenceMeta(confidence: string): { label: string; tone: string; icon: ReactNode } {
  const normalized = confidence.toLowerCase();
  if (normalized === 'high') return { label: 'Độ tin cậy cao', tone: 'high', icon: <CheckCircle2 size={14} /> };
  if (normalized === 'medium') return { label: 'Cần đối chiếu', tone: 'medium', icon: <AlertTriangle size={14} /> };
  return { label: 'Cần kiểm tra', tone: 'low', icon: <AlertTriangle size={14} /> };
}

export default function BotMessageCard({ data, status = 'success', isDemo = false }: Props) {
  const [tab, setTab] = useState<DetailTab>('overview');
  const [copied, setCopied] = useState(false);
  const citation = data.citations[0];
  const confidence = confidenceMeta(data.safety.confidence || 'none');
  const unresolved = status === 'zero_result' || status === 'not_found';

  async function copyAnswer() {
    try { await navigator.clipboard.writeText(data.formatted_answer || String(data.answer)); setCopied(true); window.setTimeout(() => setCopied(false), 1800); } catch { setCopied(false); }
  }

  return <article className={`answer-card status-${status}`}>
    <header className="answer-header">
      <div className="answer-source-line">
        <span className="ticker-badge">{citation?.company || 'ViFinQA'}</span>
        {citation?.year && <><i /> <span>{citation.year}</span></>}
        {citation?.report_type && <><i /> <span>{citation.report_type}</span></>}
      </div>
      <div className="answer-badges">{isDemo && <span className="demo-badge">Dữ liệu minh hoạ</span>}<span className={`confidence-badge ${confidence.tone}`}>{confidence.icon}{confidence.label}</span></div>
    </header>

    <div className="answer-summary">
      <div className="answer-label">Kết quả</div>
      {status === 'not_found' ? <div className="unresolved-answer"><FileSearch size={22} /><div><strong>Không tìm thấy nguồn phù hợp</strong><p>Hãy kiểm tra mã cổ phiếu, năm báo cáo hoặc thử cách diễn đạt khác.</p></div></div>
        : status === 'zero_result' ? <div className="unresolved-answer"><AlertTriangle size={22} /><div><strong>Chưa xác định được kết quả đáng tin cậy</strong><p>Giá trị kỹ thuật trả về là {data.formatted_answer || data.answer}. Vui lòng kiểm tra phần nguồn.</p></div></div>
        : <div className="answer-value-row"><strong className="answer-value">{data.formatted_answer || data.answer.toLocaleString('vi-VN')}</strong><button className="copy-answer" onClick={copyAnswer}>{copied ? <Check size={14} /> : <Clipboard size={14} />}{copied ? 'Đã sao chép' : 'Sao chép'}</button></div>}
      {!unresolved && data.unit && <span className="unit-note">Đơn vị API: {data.unit}</span>}
    </div>

    <div className="answer-tabs" role="tablist" aria-label="Chi tiết câu trả lời">
      <button role="tab" aria-selected={tab === 'overview'} className={tab === 'overview' ? 'active' : ''} onClick={() => setTab('overview')}><Info size={15} />Tổng quan</button>
      <button role="tab" aria-selected={tab === 'sources'} className={tab === 'sources' ? 'active' : ''} onClick={() => setTab('sources')}><Database size={15} />Nguồn dữ liệu <span>{data.evidence_tables.length}</span></button>
      <button role="tab" aria-selected={tab === 'query'} className={tab === 'query' ? 'active' : ''} onClick={() => setTab('query')}><ShieldCheck size={15} />Cách tính</button>
    </div>

    <div className="answer-detail">
      {tab === 'overview' && <div className="overview-panel">
        <section className="explanation-block"><div className="section-kicker"><Info size={15} />Diễn giải</div><p><RichText value={data.explanation || 'Chưa có diễn giải cho kết quả này.'} /></p></section>
        <div className="pipeline-strip" aria-label="Quy trình tạo câu trả lời">
          {['Truy hồi bảng', 'Sinh Pandas', 'Tính kết quả', 'Gắn nguồn'].map((step, index) => <div key={step}><span><Check size={12} /></span><small>{step}</small>{index < 3 && <i />}</div>)}
        </div>
        {data.safety.warning && <section className="safety-warning"><AlertTriangle size={18} /><div><strong>Cần kiểm tra thêm</strong><p>{data.safety.warning}</p></div></section>}
      </div>}
      {tab === 'sources' && <div className="sources-panel">
        <div className="detail-heading"><div><strong>Nguồn kiểm chứng</strong><p>Các tài liệu và bảng đã được dùng làm căn cứ.</p></div>{data.citations.length > 0 && <span>{data.citations.length} tài liệu</span>}</div>
        {data.citations.length === 0 && data.evidence_tables.length === 0 && <div className="detail-empty"><FileSearch size={20} /><p>Không có nguồn dữ liệu tương ứng.</p></div>}
        <div className="citation-list">{data.citations.map((item, index) => <CitationCard key={`${item.doc_id}-${index}`} citation={item} evidence={data.evidence_tables[index]} />)}</div>
        <div className="evidence-list">{data.evidence_tables.map((item, index) => <EvidenceTable key={`${item.csv_path}-${index}`} evidence={item} defaultOpen={index === 0} />)}</div>
      </div>}
      {tab === 'query' && <div className="query-panel">
        <div className="detail-heading"><div><strong>Cách hệ thống tính toán</strong><p>Text-to-Pandas giúp kết quả có thể thực thi và kiểm chứng lại.</p></div></div>
        {data.pandas_query ? <QueryInspector query={data.pandas_query} /> : <div className="detail-empty"><FileSearch size={20} /><p>Không có truy vấn Pandas cho câu trả lời này.</p></div>}
      </div>}
    </div>
  </article>;
}
