export function formatFinancialNumber(value: number, maximumFractionDigits = 1): string {
  return new Intl.NumberFormat('vi-VN', { maximumFractionDigits }).format(value);
}

export function formatCompactFinancialNumber(value: number): string {
  return new Intl.NumberFormat('vi-VN', { notation: 'compact', maximumFractionDigits: 1 }).format(value);
}

export function formatPercent(value: number, showSign = false): string {
  const sign = showSign && value > 0 ? '+' : '';
  return `${sign}${new Intl.NumberFormat('vi-VN', { maximumFractionDigits: 1 }).format(value)}%`;
}

export function formatDateTime(value: string): string {
  return new Intl.DateTimeFormat('vi-VN', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit', year: 'numeric' }).format(new Date(value));
}

export function formatCell(value: unknown): string {
  if (typeof value === 'number') return formatFinancialNumber(value, 2);
  return String(value ?? '—');
}
