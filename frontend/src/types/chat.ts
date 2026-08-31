export type ChatStatus = 'success' | 'zero_result' | 'not_found' | 'error';
export type MessageSender = 'user' | 'bot';

export interface Citation {
  doc_id: string;
  table_id: string;
  report_type: string;
  year: string;
  company: string;
  page?: number;
  line?: number;
  source_location?: string;
}

export interface TableEvidence {
  variable: string;
  csv_path: string;
  table_name: string;
  columns: string[];
  rows: Record<string, unknown>[];
  highlight_row_index: number;
}

export interface ChatResponseData {
  question: string;
  answer: number;
  unit: string;
  formatted_answer: string;
  explanation: string;
  citations: Citation[];
  evidence_tables: TableEvidence[];
  pandas_query: string;
  safety: { confidence: string; warning: string | null };
}

export interface ChatMessage {
  id: string;
  sender: MessageSender;
  text?: string;
  data?: ChatResponseData;
  status?: ChatStatus;
  error?: string;
  retryQuestion?: string;
  isDemo?: boolean;
  createdAt: string;
}

export interface ChatApiResponse {
  status: Exclude<ChatStatus, 'error'>;
  data: ChatResponseData;
}

export interface HealthResponse {
  status: string;
  manifest_count: number;
  available_tickers: number;
}

export interface HistoryRecord {
  id: string;
  question: string;
  createdAt: string;
  formattedAnswer?: string;
  ticker?: string;
  year?: string;
  status: ChatStatus;
  isDemo?: boolean;
}

export type WorkspaceView = 'assistant' | 'compare' | 'dashboard' | 'data' | 'history';
