'use client';

import { ArrowRight, Building2, Sparkles, TrendingDown, TrendingUp } from 'lucide-react';
import { COMPARE_SERIES, DASHBOARD_KPIS, YEARS } from '@/data/demo';
import { formatFinancialNumber, formatPercent } from '@/lib/format';
import { LineChart, Sparkline } from '@/components/ui/Charts';
import type { FinancialSeries } from '@/types/demo';

const performance = [
  { label: 'ROE', value: '26,1%', change: '+1,3 điểm %' },
  { label: 'ROA', value: '17,7%', change: '+0,6 điểm %' },
  { label: 'Biên lợi nhuận', value: '14,9%', change: '+0,5 điểm %' },
  { label: 'Tăng trưởng doanh thu', value: '1,6%', change: 'Phục hồi nhẹ' },
];

export default function DashboardView({ onAskAI }: { onAskAI: (prompt: string) => void }) {
  const revenue = COMPARE_SERIES.find(item => item.ticker === 'VNM')!;
  const profit: FinancialSeries = { ticker: 'LNST', name: 'Lợi nhuận sau thuế', color: '#14a274', values: { 2020: 11240, 2021: 10630, 2022: 8578, 2023: 9019 } };
  return <section className="screen"><header className="screen-header"><div><span className="screen-eyebrow">Financial Dashboard</span><h1>Phân tích doanh nghiệp</h1><p>Tổng quan hiệu quả và xu hướng tài chính theo thời gian.</p></div><span className="demo-badge prominent">Dữ liệu minh hoạ</span></header>
    <div className="dashboard-filters panel"><label><span>Doanh nghiệp</span><select defaultValue="VNM"><option>VNM · Vinamilk</option><option>HPG · Hòa Phát</option><option>ACB · ACB</option></select></label><label><span>Giai đoạn</span><select defaultValue="2019–2023"><option>2019–2023</option><option>2020–2023</option></select></label><label><span>Báo cáo</span><select defaultValue="Hợp nhất"><option>Hợp nhất</option><option>Công ty mẹ</option></select></label><button className="primary-button" onClick={() => onAskAI('Phân tích tình hình tài chính của VNM giai đoạn 2019–2023')}><Sparkles size={16} />Phân tích bằng AI</button></div>
    <div className="kpi-grid">{DASHBOARD_KPIS.map(item => <article className="kpi-card" key={item.label}><div><span>{item.label}</span><i className={item.change >= 0 ? 'positive' : 'negative'}>{item.change >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}{formatPercent(item.change, true)} YoY</i></div><strong>{formatFinancialNumber(item.value)} <small>{item.unit}</small></strong><Sparkline values={item.sparkline} positive={item.change >= 0} /></article>)}</div>
    <div className="dashboard-grid"><article className="panel dashboard-chart"><div className="panel-heading"><div><span>Kết quả kinh doanh</span><h2>Doanh thu & lợi nhuận</h2></div></div><LineChart series={[revenue, profit]} years={YEARS} /></article>
      <article className="panel structure-panel"><div className="panel-heading"><div><span>Cơ cấu tài chính</span><h2>Tài sản & nguồn vốn</h2></div></div><div className="structure-total"><Building2 size={18} /><span>Tổng tài sản 2023</span><strong>52.714 tỷ</strong></div>{[
        ['Tài sản ngắn hạn', 65, '34.265'], ['Tài sản dài hạn', 35, '18.449'], ['Nợ phải trả', 31, '16.237'], ['Vốn chủ sở hữu', 69, '36.477'],
      ].map(([label, percent, value], index) => <div className="structure-row" key={String(label)}><div><span>{label}</span><strong>{value} tỷ</strong></div><div className="progress-track"><i style={{ width: `${percent}%` }} className={index > 1 ? 'green' : ''} /></div><small>{percent}%</small></div>)}</article></div>
    <div className="performance-grid">{performance.map(item => <article key={item.label}><span>{item.label}</span><strong>{item.value}</strong><small>{item.change}</small></article>)}</div>
    <article className="panel"><div className="panel-heading"><div><span>Financial trend</span><h2>Số liệu theo năm</h2></div></div><div className="table-scroll"><table className="trend-table"><thead><tr><th>Chỉ tiêu</th>{YEARS.map(year => <th key={year}>{year}</th>)}</tr></thead><tbody><tr><th>Doanh thu thuần</th>{YEARS.map(year => <td key={year}>{formatFinancialNumber(revenue.values[year])}</td>)}</tr><tr><th>Lợi nhuận sau thuế</th>{YEARS.map(year => <td key={year}>{formatFinancialNumber(profit.values[year])}</td>)}</tr><tr><th>Biên lợi nhuận</th>{[19.1,17.5,14.4,14.9].map((value, index) => <td key={YEARS[index]}>{formatPercent(value)}</td>)}</tr></tbody></table></div></article>
    <button className="ask-ai-banner" onClick={() => onAskAI('Phân tích doanh thu, lợi nhuận và hiệu quả hoạt động của VNM giai đoạn 2020–2023')}><span><Sparkles size={19} /></span><div><strong>Đi sâu hơn với ViFinQA</strong><p>Yêu cầu AI giải thích các biến động và chỉ ra bảng dữ liệu liên quan.</p></div><ArrowRight size={18} /></button>
  </section>;
}
