import { Database, FileText } from 'lucide-react';
import type { Citation, TableEvidence } from '@/types/chat';

export default function CitationCard({ citation, evidence }: { citation: Citation; evidence?: TableEvidence }) {
  return <article className="citation-card">
    <div className="citation-icon"><FileText size={18} /></div>
    <div className="citation-content">
      <div className="citation-heading"><strong>{citation.company || 'Không rõ doanh nghiệp'}</strong><span>{citation.report_type || 'Báo cáo tài chính'} · {citation.year || 'Không rõ năm'}</span></div>
      <dl className="metadata-list">
        <div><dt>Document ID</dt><dd>{citation.doc_id || '—'}</dd></div>
        <div><dt>Table ID</dt><dd>{citation.table_id || '—'}</dd></div>
        {evidence && <><div><dt>Bảng</dt><dd>{evidence.table_name || '—'}</dd></div><div><dt>CSV</dt><dd>{evidence.csv_path || '—'}</dd></div></>}
        {citation.page !== undefined && <div><dt>Trang</dt><dd>{citation.page}</dd></div>}
        {citation.line !== undefined && <div><dt>Dòng</dt><dd>{citation.line}</dd></div>}
        {citation.source_location && <div><dt>Vị trí</dt><dd>{citation.source_location}</dd></div>}
      </dl>
    </div>
    <Database size={15} className="citation-database" />
  </article>;
}
