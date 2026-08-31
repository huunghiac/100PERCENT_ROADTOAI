import type { ChatResponseData } from '@/types/chat';
import type { DemoSourceRecord, FinancialSeries, KpiItem } from '@/types/demo';

export const FALLBACK_SUGGESTIONS = [
  'Doanh thu thuần của VNM năm 2023 là bao nhiêu tỷ đồng?',
  'So sánh lợi nhuận sau thuế của VNM giai đoạn 2020–2023',
  'ROE của ACB năm 2022 là bao nhiêu?',
  'So sánh tổng tài sản của ACB và BID năm 2023',
  'Chi phí quản lý doanh nghiệp của HPG năm 2022',
];

export const CONTEXT_CHIPS = [
  { label: 'Tra cứu số liệu', prompt: 'Doanh thu thuần của VNM năm 2023 là bao nhiêu tỷ đồng?' },
  { label: 'So sánh doanh nghiệp', prompt: 'So sánh doanh thu thuần của VNM và HPG giai đoạn 2020–2023' },
  { label: 'Phân tích xu hướng', prompt: 'Phân tích xu hướng lợi nhuận sau thuế của VNM giai đoạn 2020–2023' },
  { label: 'Tính chỉ số', prompt: 'ROE của ACB năm 2022 là bao nhiêu?' },
];

export const YEARS = [2020, 2021, 2022, 2023];
export const COMPANY_OPTIONS = ['VNM', 'ACB', 'BID', 'HPG', 'VJC', 'MBB', 'FPT'];

export const COMPARE_SERIES: FinancialSeries[] = [
  { ticker: 'VNM', name: 'Vinamilk', color: '#246bfe', values: { 2020: 59000, 2021: 60800, 2022: 59400, 2023: 60370 } },
  { ticker: 'HPG', name: 'Hòa Phát', color: '#14a274', values: { 2020: 91400, 2021: 149700, 2022: 142800, 2023: 120400 } },
  { ticker: 'ACB', name: 'ACB', color: '#d68b1f', values: { 2020: 15000, 2021: 19400, 2022: 23600, 2023: 27300 } },
  { ticker: 'BID', name: 'BIDV', color: '#8b5cf6', values: { 2020: 47700, 2021: 53000, 2022: 61500, 2023: 71000 } },
  { ticker: 'FPT', name: 'FPT', color: '#e05858', values: { 2020: 29600, 2021: 35600, 2022: 44000, 2023: 52600 } },
];

export const DASHBOARD_KPIS: KpiItem[] = [
  { label: 'Doanh thu', value: 60370, unit: 'tỷ đồng', change: 1.6, sparkline: [59000, 60800, 59400, 60370] },
  { label: 'Lợi nhuận sau thuế', value: 9019, unit: 'tỷ đồng', change: 5.2, sparkline: [11240, 10630, 8578, 9019] },
  { label: 'Tổng tài sản', value: 52714, unit: 'tỷ đồng', change: 7.1, sparkline: [48200, 49700, 49200, 52714] },
  { label: 'Vốn chủ sở hữu', value: 36477, unit: 'tỷ đồng', change: 3.8, sparkline: [33500, 35200, 35140, 36477] },
];

export const DEMO_SOURCES: DemoSourceRecord[] = [
  {
    id: 'src-vnm-2023-income', ticker: 'VNM', company: 'CTCP Sữa Việt Nam', year: 2023,
    reportType: 'Báo cáo tài chính hợp nhất', tableName: 'Báo cáo kết quả hoạt động kinh doanh',
    tableSlug: 'BaoCaoKetQuaKinhDoanh_consolidated', unit: 'Triệu VND', rowCount: 24,
    csvPath: 'data/processed_csv/VNM/VNM_2023_BaoCaoKetQuaKinhDoanh_consolidated.csv',
    docId: 'VNM_financial_statements_2023_consolidated', category: 'Kết quả kinh doanh',
    preview: [
      { label: 'Doanh thu bán hàng và cung cấp dịch vụ', value: 61608421, unit: 'Triệu VND' },
      { label: 'Các khoản giảm trừ doanh thu', value: 1239315, unit: 'Triệu VND' },
      { label: 'Doanh thu thuần', value: 60369106, unit: 'Triệu VND' },
      { label: 'Lợi nhuận sau thuế', value: 9019342, unit: 'Triệu VND' },
    ],
  },
  {
    id: 'src-acb-2023-balance', ticker: 'ACB', company: 'Ngân hàng TMCP Á Châu', year: 2023,
    reportType: 'Báo cáo tài chính hợp nhất', tableName: 'Bảng cân đối kế toán',
    tableSlug: 'BangCanDoiKeToan_consolidated', unit: 'Triệu VND', rowCount: 68,
    csvPath: 'data/processed_csv/ACB/ACB_2023_BangCanDoiKeToan_consolidated.csv',
    docId: 'ACB_financial_statements_2023_consolidated', category: 'Cân đối kế toán',
    preview: [
      { label: 'Tổng tài sản', value: 718794000, unit: 'Triệu VND' },
      { label: 'Tiền mặt, vàng bạc, đá quý', value: 6582000, unit: 'Triệu VND' },
      { label: 'Vốn chủ sở hữu', value: 70421000, unit: 'Triệu VND' },
    ],
  },
  {
    id: 'src-hpg-2022-expense', ticker: 'HPG', company: 'CTCP Tập đoàn Hòa Phát', year: 2022,
    reportType: 'Báo cáo tài chính hợp nhất', tableName: 'Chi phí bán hàng và quản lý doanh nghiệp',
    tableSlug: 'ChiPhiBanHangVaQuanLy_consolidated', unit: 'VND', rowCount: 17,
    csvPath: 'data/processed_csv/HPG/HPG_2022_ChiPhiBanHangVaQuanLy_consolidated.csv',
    docId: 'HPG_financial_statements_2022_consolidated', category: 'Thuyết minh chi phí',
    preview: [
      { label: 'Chi phí nhân viên', value: 1331000000000, unit: 'VND' },
      { label: 'Chi phí khấu hao', value: 207000000000, unit: 'VND' },
      { label: 'Chi phí dịch vụ mua ngoài', value: 779000000000, unit: 'VND' },
    ],
  },
  {
    id: 'src-bid-2023-balance', ticker: 'BID', company: 'Ngân hàng TMCP Đầu tư và Phát triển Việt Nam', year: 2023,
    reportType: 'Báo cáo tài chính hợp nhất', tableName: 'Bảng cân đối kế toán',
    tableSlug: 'BangCanDoiKeToan_consolidated', unit: 'Triệu VND', rowCount: 72,
    csvPath: 'data/processed_csv/BID/BID_2023_BangCanDoiKeToan_consolidated.csv',
    docId: 'BID_financial_statements_2023_consolidated', category: 'Cân đối kế toán',
    preview: [
      { label: 'Tổng tài sản', value: 2260000000, unit: 'Triệu VND' },
      { label: 'Cho vay khách hàng', value: 1780000000, unit: 'Triệu VND' },
      { label: 'Vốn chủ sở hữu', value: 118000000, unit: 'Triệu VND' },
    ],
  },
];

export function createDemoAnswer(question: string): ChatResponseData {
  return {
    question,
    answer: 60369.1,
    unit: 'tỷ đồng',
    formatted_answer: '60.369,1 tỷ đồng',
    explanation: 'Theo dữ liệu minh hoạ từ Báo cáo tài chính hợp nhất năm 2023, doanh thu thuần của VNM được xác định từ dòng “Doanh thu thuần về bán hàng và cung cấp dịch vụ”.',
    citations: [{
      doc_id: 'VNM_financial_statements_2023_consolidated',
      table_id: 'VNM_2023_BaoCaoKetQuaKinhDoanh_consolidated',
      report_type: 'Báo cáo tài chính hợp nhất', year: '2023', company: 'VNM',
    }],
    evidence_tables: [{
      variable: 'df1', csv_path: 'data/processed_csv/VNM/VNM_2023_BaoCaoKetQuaKinhDoanh_consolidated.csv',
      table_name: 'Báo cáo kết quả hoạt động kinh doanh',
      columns: ['Chi_tieu', 'Gia_tri', 'Don_vi'],
      rows: [
        { Chi_tieu: 'Doanh thu bán hàng và cung cấp dịch vụ', Gia_tri: 61608.4, Don_vi: 'tỷ đồng' },
        { Chi_tieu: 'Các khoản giảm trừ doanh thu', Gia_tri: 1239.3, Don_vi: 'tỷ đồng' },
        { Chi_tieu: 'Doanh thu thuần', Gia_tri: 60369.1, Don_vi: 'tỷ đồng' },
        { Chi_tieu: 'Giá vốn hàng bán', Gia_tri: 35538.8, Don_vi: 'tỷ đồng' },
      ],
      highlight_row_index: 2,
    }],
    pandas_query: "float(df1.loc[df1['Chi_tieu'].str.contains('Doanh thu thuần'), 'Gia_tri'].iloc[0])",
    safety: { confidence: 'high', warning: null },
  };
}
