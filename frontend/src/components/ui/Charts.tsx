import type { FinancialSeries } from '@/types/demo';
import { formatFinancialNumber } from '@/lib/format';

export function Sparkline({ values, positive = true }: { values: number[]; positive?: boolean }) {
  const width = 120; const height = 36; const pad = 3;
  const min = Math.min(...values); const max = Math.max(...values); const range = max - min || 1;
  const points = values.map((value, index) => {
    const x = pad + index * ((width - pad * 2) / Math.max(1, values.length - 1));
    const y = height - pad - ((value - min) / range) * (height - pad * 2);
    return `${x},${y}`;
  }).join(' ');
  return <svg className="sparkline" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Biểu đồ xu hướng nhỏ">
    <polyline points={points} fill="none" stroke={positive ? 'var(--positive)' : 'var(--danger)'} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>;
}

export function LineChart({ series, years, unit = 'tỷ đồng' }: { series: FinancialSeries[]; years: number[]; unit?: string }) {
  const width = 760; const height = 280; const left = 60; const right = 22; const top = 22; const bottom = 42;
  const allValues = series.flatMap(item => years.map(year => item.values[year] ?? 0));
  const max = Math.max(...allValues, 1) * 1.08; const min = Math.min(...allValues, 0);
  const x = (index: number) => left + index * ((width - left - right) / Math.max(1, years.length - 1));
  const y = (value: number) => top + (max - value) / (max - min || 1) * (height - top - bottom);
  const ticks = [0, .25, .5, .75, 1];

  return <div className="chart-shell">
    <div className="chart-legend" aria-label="Chú giải biểu đồ">
      {series.map(item => <span key={item.ticker}><i style={{ background: item.color }} />{item.ticker}</span>)}
      <small>Đơn vị: {unit}</small>
    </div>
    <svg className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Biểu đồ xu hướng ${series.map(item => item.ticker).join(', ')}`}>
      {ticks.map(tick => {
        const value = min + (max - min) * tick; const rowY = y(value);
        return <g key={tick}><line x1={left} x2={width - right} y1={rowY} y2={rowY} className="chart-grid" /><text x={left - 10} y={rowY + 4} textAnchor="end" className="chart-axis">{Math.round(value / 1000)}k</text></g>;
      })}
      {years.map((year, index) => <text key={year} x={x(index)} y={height - 12} textAnchor="middle" className="chart-axis">{year}</text>)}
      {series.map(item => {
        const points = years.map((year, index) => `${x(index)},${y(item.values[year] ?? 0)}`).join(' ');
        return <g key={item.ticker}>
          <polyline points={points} fill="none" stroke={item.color} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
          {years.map((year, index) => <circle key={year} cx={x(index)} cy={y(item.values[year] ?? 0)} r="4" fill="white" stroke={item.color} strokeWidth="3">
            <title>{`${item.ticker} ${year}: ${formatFinancialNumber(item.values[year] ?? 0)} ${unit}`}</title>
          </circle>)}
        </g>;
      })}
    </svg>
  </div>;
}
