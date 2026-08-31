'use client';

import { useMemo, useState } from 'react';
import { ArrowRight, CheckCircle2, Clock3, History, Search, Trash2, XCircle } from 'lucide-react';
import type { HistoryRecord } from '@/types/chat';
import { formatDateTime } from '@/lib/format';

interface Props { records: HistoryRecord[]; onOpen: (question: string) => void; onDelete: (id: string) => void; onClear: () => void }

export default function HistoryView({ records, onOpen, onDelete, onClear }: Props) {
  const [search, setSearch] = useState(''); const [confirmClear, setConfirmClear] = useState(false);
  const filtered = useMemo(() => records.filter(record => record.question.toLocaleLowerCase('vi-VN').includes(search.toLocaleLowerCase('vi-VN'))), [records, search]);
  return <section className="screen"><header className="screen-header"><div><span className="screen-eyebrow">Local workspace</span><h1>Lịch sử phân tích</h1><p>Các câu hỏi gần đây được lưu cục bộ trên trình duyệt này.</p></div>{records.length > 0 && <div className="clear-history-wrap">{confirmClear ? <div className="clear-confirm"><span>Xóa toàn bộ?</span><button onClick={() => { onClear(); setConfirmClear(false); }}>Xác nhận</button><button onClick={() => setConfirmClear(false)}>Hủy</button></div> : <button className="danger-ghost" onClick={() => setConfirmClear(true)}><Trash2 size={15} />Xóa toàn bộ</button>}</div>}</header>
    {records.length > 0 && <label className="history-search"><Search size={17} /><input value={search} onChange={event => setSearch(event.target.value)} placeholder="Tìm trong lịch sử câu hỏi..." aria-label="Tìm lịch sử" /></label>}
    {records.length === 0 ? <div className="empty-panel history-empty"><History size={28} /><h2>Chưa có lịch sử phân tích</h2><p>Các câu hỏi bạn đã thực hiện sẽ xuất hiện tại đây.</p></div>
      : filtered.length === 0 ? <div className="empty-panel"><Search size={25} /><h2>Không có kết quả phù hợp</h2><p>Thử tìm bằng mã doanh nghiệp hoặc nội dung câu hỏi khác.</p></div>
      : <div className="history-list">{filtered.map(record => <article key={record.id} className="history-item"><div className={`history-status ${record.status}`}>{record.status === 'success' ? <CheckCircle2 size={17} /> : <XCircle size={17} />}</div><div className="history-copy"><div><span><Clock3 size={13} />{formatDateTime(record.createdAt)}</span>{record.isDemo && <span className="demo-badge">Minh hoạ</span>}</div><h2>{record.question}</h2><p>{record.formattedAnswer || (record.status === 'error' ? 'Không thể kết nối hệ thống phân tích' : 'Chưa xác định được kết quả')}</p><div className="history-tags">{record.ticker && <span>{record.ticker}</span>}{record.year && <span>{record.year}</span>}<span>{record.status}</span></div></div><div className="history-actions"><button onClick={() => onOpen(record.question)}>Mở lại<ArrowRight size={14} /></button><button onClick={() => onDelete(record.id)} aria-label="Xóa mục lịch sử"><Trash2 size={15} /></button></div></article>)}</div>}
  </section>;
}
