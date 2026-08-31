'use client';

import { useMemo, useState } from 'react';
import { ArrowRight, Award, GitCompareArrows, Sparkles, TrendingUp, Waves } from 'lucide-react';
import { COMPANY_OPTIONS, COMPARE_SERIES, DEMO_SOURCES, YEARS } from '@/data/demo';
import { formatFinancialNumber, formatPercent } from '@/lib/format';
import { LineChart } from '@/components/ui/Charts';

export default function CompareView({ onAskAI }: { onAskAI: (prompt: string) => void }) {
  const [selected, setSelected] = useState(['VNM', 'HPG']);
  const [applied, setApplied] = useState(['VNM', 'HPG']);
  const [metric, setMetric] = useState('Doanh thu thuần');
  const [message, setMessage] = useState('');
  const series = useMemo(() => COMPARE_SERIES.filter(item => applied.includes(item.ticker)), [applied]);
  const latest = series.reduce((best, item) => !best || item.values[2023] > best.values[2023] ? item : best, series[0]);
  const growth = (item: typeof series[number]) => ((item.values[2023] / item.values[2020]) ** (1 / 3) - 1) * 100;
  const fastest = series.reduce((best, item) => !best || growth(item) > growth(best) ? item : best, series[0]);

  function toggleTicker(ticker: string) {
    setMessage('');
    setSelected(current => current.includes(ticker) ? current.filter(item => item !== ticker) : current.length < 4 ? [...current, ticker] : current);
  }
  function compare() { if (selected.length < 2) { setMessage('Chọn ít nhất 2 doanh nghiệp để bắt đầu so sánh.'); return; } setApplied(selected); setMessage(''); }

  return <section className="screen"><header className="screen-header"><div><span className="screen-eyebrow">Dữ liệu minh hoạ</span><h1>So sánh doanh nghiệp</h1><p>Đối chiếu chỉ tiêu tài chính giữa nhiều doanh nghiệp và giai đoạn.</p></div><span className="demo-badge prominent">Dữ liệu minh hoạ cho UI</span></header>
    <div className="compare-builder panel">
      <div className="builder-field company-field"><label>Doanh nghiệp <span>{selected.length}/4</span></label><div className="ticker-selector">{COMPANY_OPTIONS.map(ticker => <button key={ticker} className={selected.includes(ticker) ? 'selected' : ''} onClick={() => toggleTicker(ticker)}>{ticker}</button>)}</div></div>
      <div className="builder-field"><label htmlFor="metric">Chỉ tiêu</label><select id="metric" value={metric} onChange={event => setMetric(event.target.value)}><option>Doanh thu thuần</option><option>Lợi nhuận sau thuế</option><option>Tổng tài sản</option><option>Vốn chủ sở hữu</option><option>ROE</option><option>ROA</option></select></div>
      <div className="builder-field"><label>Giai đoạn</label><div className="range-control"><select aria-label="Năm bắt đầu" defaultValue="2020"><option>2019</option><option>2020</option><option>2021</option></select><span>—</span><select aria-label="Năm kết thúc" defaultValue="2023"><option>2022</option><option>2023</option></select></div></div>
      <div className="builder-field"><label htmlFor="report">Loại báo cáo</label><select id="report"><option>Hợp nhất</option><option>Công ty mẹ</option></select></div>
      <div className="builder-field"><label htmlFor="unit">Đơn vị</label><select id="unit"><option>Tự động</option><option>Tỷ đồng</option><option>Triệu đồng</option><option>%</option></select></div>
      <button className="primary-button compare-button" onClick={compare}><GitCompareArrows size={17} />So sánh</button>
      {message && <p className="form-message">{message}</p>}
    </div>

    {series.length >= 2 && <div className="compare-results">
      <div className="summary-grid">
        <article className="summary-card"><i className="blue"><Award size={18} /></i><span>Cao nhất năm 2023</span><strong>{latest?.ticker}</strong><p>{formatFinancialNumber(latest?.values[2023] ?? 0)} tỷ đồng</p></article>
        <article className="summary-card"><i className="green"><TrendingUp size={18} /></i><span>Tăng trưởng nhanh nhất</span><strong>{fastest?.ticker}</strong><p>CAGR {formatPercent(growth(fastest ?? series[0]))}</p></article>
        <article className="summary-card"><i className="amber"><Waves size={18} /></i><span>Biến động lớn nhất</span><strong>HPG</strong><p>Biên độ 58.300 tỷ đồng</p></article>
      </div>
      <article className="panel chart-panel"><div className="panel-heading"><div><span>Xu hướng 2020–2023</span><h2>{metric}</h2></div><span className="demo-badge">Minh hoạ</span></div><LineChart series={series} years={YEARS} /></article>
      <article className="panel"><div className="panel-heading"><div><span>Bảng đối chiếu</span><h2>Số liệu theo doanh nghiệp</h2></div></div><div className="table-scroll"><table className="comparison-table"><thead><tr><th>Doanh nghiệp</th>{YEARS.map(year => <th key={year}>{year}</th>)}<th>CAGR</th></tr></thead><tbody>{series.map(item => <tr key={item.ticker}><th><span style={{ background: item.color }} />{item.ticker}<small>{item.name}</small></th>{YEARS.map(year => <td key={year}>{formatFinancialNumber(item.values[year])}</td>)}<td className={growth(item) >= 0 ? 'positive' : 'negative'}>{formatPercent(growth(item), true)}</td></tr>)}</tbody></table></div></article>
      <article className="insight-card"><div className="insight-icon"><Sparkles size={20} /></div><div><span>Nhận xét nhanh · Dữ liệu minh hoạ</span><h3>Khoảng cách tăng trưởng đang mở rộng</h3><p>Trong dữ liệu minh hoạ, VNM duy trì doanh thu ổn định trong khi HPG biến động mạnh hơn theo chu kỳ. HPG vẫn có quy mô doanh thu lớn hơn ở năm 2023.</p><div className="source-chips">{DEMO_SOURCES.filter(source => applied.includes(source.ticker)).slice(0, 3).map(source => <span key={source.id}>{source.ticker} · {source.year}</span>)}</div></div><button onClick={() => onAskAI(`So sánh ${metric.toLowerCase()} của ${applied.join(' và ')} giai đoạn 2020–2023`)}>Hỏi AI về so sánh này<ArrowRight size={15} /></button></article>
    </div>}
  </section>;
}
