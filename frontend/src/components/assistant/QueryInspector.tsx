'use client';

import { useState } from 'react';
import { Check, Clipboard, Code2 } from 'lucide-react';

export default function QueryInspector({ query }: { query: string }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try { await navigator.clipboard.writeText(query); setCopied(true); window.setTimeout(() => setCopied(false), 1800); } catch { setCopied(false); }
  }
  return <section className="query-inspector">
    <div className="query-heading"><div><span><Code2 size={16} /> Pandas Query</span><p>Truy vấn có thể chạy lại trên bảng nguồn để kiểm chứng kết quả.</p></div><button onClick={copy} className="copy-button"><>{copied ? <Check size={14} /> : <Clipboard size={14} />}{copied ? 'Đã sao chép' : 'Sao chép'}</></button></div>
    <pre><code>{query}</code></pre>
  </section>;
}
