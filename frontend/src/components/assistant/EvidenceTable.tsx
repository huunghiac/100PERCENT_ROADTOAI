'use client';

import { useState } from 'react';
import { ChevronDown, Table2 } from 'lucide-react';
import type { TableEvidence } from '@/types/chat';
import { formatCell } from '@/lib/format';

export default function EvidenceTable({ evidence, defaultOpen = false }: { evidence: TableEvidence; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return <section className={`evidence-panel ${open ? 'open' : ''}`}>
    <button className="evidence-trigger" onClick={() => setOpen(value => !value)} aria-expanded={open}>
      <span className="evidence-title"><i><Table2 size={16} /></i><span><strong>{evidence.table_name || 'Bảng dữ liệu nguồn'}</strong><small>{evidence.variable || 'dataframe'} · {evidence.columns.length} cột · {evidence.rows.length} dòng xem trước</small></span></span>
      <ChevronDown size={17} />
    </button>
    {open && <div className="evidence-body">
      <div className="evidence-path"><span>CSV</span><code>{evidence.csv_path || 'Không có đường dẫn'}</code></div>
      <div className="table-scroll"><table className="evidence-table"><thead><tr>{evidence.columns.map((column, index) => <th key={`${column}-${index}`}>{column}</th>)}</tr></thead>
        <tbody>{evidence.rows.map((row, rowIndex) => <tr key={rowIndex} className={rowIndex === evidence.highlight_row_index ? 'highlight' : ''}>
          {evidence.columns.map((column, columnIndex) => <td key={`${column}-${columnIndex}`}>{formatCell(row[column])}</td>)}
        </tr>)}</tbody>
      </table></div>
      <p className="highlight-legend"><span /> Dòng được hệ thống sử dụng để tạo kết quả</p>
    </div>}
  </section>;
}
