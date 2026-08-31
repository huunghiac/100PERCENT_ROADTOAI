export interface FinancialSeries {
  ticker: string;
  name: string;
  color: string;
  values: Record<number, number>;
}

export interface KpiItem {
  label: string;
  value: number;
  unit: string;
  change: number;
  sparkline: number[];
}

export interface SourcePreviewRow {
  label: string;
  value: string | number;
  unit?: string;
}

export interface DemoSourceRecord {
  id: string;
  ticker: string;
  company: string;
  year: number;
  reportType: string;
  tableName: string;
  tableSlug: string;
  unit: string;
  rowCount: number;
  csvPath: string;
  docId: string;
  category: string;
  preview: SourcePreviewRow[];
}
