# HƯỚNG DẪN PHÁT TRIỂN FULLSTACK CHATBOT TÀI CHÍNH (ViFinQA WEB)

Tài liệu này dành cho Developer phát triển giao diện Web Chatbot hoàn chỉnh cho hệ thống ViFinQA theo mô hình **Fullstack Decoupled (FastAPI Backend + Next.js Frontend)**.

---

## 1. Kiến Trúc Hệ Thống

```
┌────────────────────────────────────────────────────────────────────────┐
│                        NGƯỜI DÙNG / BROWSER                           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP (Port 3000)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  FRONTEND: Next.js 14+ (App Router) + TailwindCSS + Lucide Icons       │
│  - Giao diện Chat trực quan, gợi ý câu hỏi                              │
│  - Thẻ hiển thị số liệu nổi bật + Đơn vị                               │
│  - Khối diễn giải số liệu tự nhiên (Explanation)                        │
│  - Bảng số liệu đối soát (Highlight dòng chứa đáp án)                  │
│  - Xem và copy mã truy vấn Pandas (Code Inspector)                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ REST API JSON (Port 8000)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  BACKEND: FastAPI (`server.py`)                                        │
│  - Endpoint `/api/chat`: Nhận câu hỏi, điều phối Engine                │
│  - Endpoint `/api/suggestions`: Trả về câu hỏi mẫu                    │
│  - Sinh diễn giải tài chính dễ hiểu cho người không chuyên             │
│  - Chuẩn bị dữ liệu bảng preview và highlight dòng kết quả             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Gọi module nội bộ
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  VIFINQA CORE ENGINE (Python)                                          │
│  - `TableRetriever`: Tra cứu bảng CSV từ kho BCTC                      │
│  - `try_rule_based_answer`: Khớp tất định nhanh (Fallback)             │
│  - `PandasAgent`: LLM Text-to-Pandas Agent                             │
│  - Kho dữ liệu CSV: `data/processed_csv/`                              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phần 1: Cài Đặt & Chạy Backend (FastAPI)

### 2.1 Cài thư viện Backend
Chạy lệnh tại thư mục gốc của project:
```bash
pip install fastapi uvicorn pydantic
```

### 2.2 Mã nguồn Backend (`server.py`)
Mã nguồn backend đã được tạo sẵn tại file `server.py` ở thư mục gốc của repository, kết nối trực tiếp với các module xử lý `retriever`, `agent`, `fallback`.

## 3. Phần 2: Cài Đặt Frontend (Next.js)

### 3.1 Khởi tạo dự án Next.js
Mở terminal mới (song song với Backend):
```bash
npx create-next-app@latest frontend --typescript --tailwind --eslint --app --src-dir --import-alias "@/*"
cd frontend
npm install lucide-react clsx tailwind-merge
```

### 3.2 Kiểu Dữ Liệu TypeScript (`src/types/chat.ts`)
Tạo file `src/types/chat.ts`:

```typescript
export interface Citation {
  doc_id: string;
  table_id: string;
  report_type: string;
  year: string;
  company: string;
}

export interface TableEvidence {
  variable: string;
  csv_path: string;
  table_name: string;
  columns: string[];
  rows: Record<string, any>[];
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
  safety: {
    confidence: string;
    warning: string | null;
  };
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'bot';
  text?: string;
  data?: ChatResponseData;
  loading?: boolean;
### 3.3 Component Hiển Thị Kết Quả Bot (`src/components/BotMessageCard.tsx`)
Tạo file `src/components/BotMessageCard.tsx`:

```tsx
'use client';

import React, { useState } from 'react';
import { ChatResponseData } from '@/types/chat';
import { CheckCircle2, ChevronDown, ChevronUp, Code2, Database, Info, Copy, Check } from 'lucide-react';

interface Props {
  data: ChatResponseData;
}

export default function BotMessageCard({ data }: Props) {
  const [showCode, setShowCode] = useState(false);
  const [showTable, setShowTable] = useState(true);
  const [copied, setCopied] = useState(false);

  const handleCopyCode = () => {
    navigator.clipboard.writeText(data.pandas_query);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-white border border-slate-200 rounded-2xl shadow-sm overflow-hidden text-slate-800">
      {/* 1. Header: Thẻ trích dẫn Công ty & Năm */}
      {data.citations.length > 0 && (
        <div className="bg-slate-50 border-b border-slate-100 px-5 py-3 flex items-center justify-between text-xs text-slate-600">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full border border-blue-100">
              {data.citations[0].company}
            </span>
            <span>• Năm {data.citations[0].year}</span>
            <span>• {data.citations[0].report_type}</span>
          </div>
          {data.safety.confidence === 'high' ? (
            <span className="flex items-center gap-1 text-emerald-600 font-medium">
              <CheckCircle2 className="w-3.5 h-3.5" /> Khớp chính xác
            </span>
          ) : (
            <span className="text-amber-600 font-medium">⚠️ Độ tin cậy thấp</span>
          )}
        </div>
      )}

      <div className="p-5 space-y-4">
        {/* 2. Con số đáp án chính */}
        <div>
          <div className="text-xs uppercase font-medium tracking-wider text-slate-400 mb-1">
            Kết quả tính toán
          </div>
          <div className="text-3xl font-extrabold text-blue-900 tracking-tight">
            {data.formatted_answer}
          </div>
        </div>
        {/* 4. Bảng số liệu trích xuất */}
        {data.evidence_tables.length > 0 && (
          <div className="border border-slate-200 rounded-xl overflow-hidden text-xs">
            <button
              onClick={() => setShowTable(!showTable)}
              className="w-full bg-slate-50 px-4 py-2.5 flex items-center justify-between font-medium text-slate-700 hover:bg-slate-100 transition"
            >
              <span className="flex items-center gap-2">
                <Database className="w-3.5 h-3.5 text-blue-600" />
                Bảng nguồn: {data.evidence_tables[0].table_name}
              </span>
              {showTable ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>

            {showTable && (
              <div className="overflow-x-auto max-h-56">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="bg-slate-100 text-slate-600 border-b border-slate-200">
                      {data.evidence_tables[0].columns.map((col, idx) => (
                        <th key={idx} className="p-2.5 font-semibold">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.evidence_tables[0].rows.map((row, rIdx) => {
                      const isHighlighted = rIdx === data.evidence_tables[0].highlight_row_index;
                      return (
                        <tr
                          key={rIdx}
                          className={`border-b border-slate-100 transition ${
                            isHighlighted ? 'bg-amber-50 font-medium text-amber-900' : 'hover:bg-slate-50'
                          }`}
                        >
                          {data.evidence_tables[0].columns.map((col, cIdx) => (
                            <td key={cIdx} className="p-2.5">
                              {typeof row[col] === 'number' ? row[col].toLocaleString('vi-VN') : row[col]}
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* 5. Mã truy vấn Pandas */}
        {data.pandas_query && (
          <div className="border border-slate-200 rounded-xl overflow-hidden text-xs">
            <button
              onClick={() => setShowCode(!showCode)}
              className="w-full bg-slate-50 px-4 py-2.5 flex items-center justify-between font-medium text-slate-700 hover:bg-slate-100 transition"
            >
              <span className="flex items-center gap-2">
                <Code2 className="w-3.5 h-3.5 text-indigo-600" />
                Truy vấn Pandas
              </span>
              {showCode ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>

            {showCode && (
              <div className="relative bg-slate-900 text-slate-100 p-4 font-mono text-xs overflow-x-auto">
                <button
                  onClick={handleCopyCode}
                  className="absolute top-3 right-3 bg-slate-800 hover:bg-slate-700 text-slate-300 px-2 py-1 rounded flex items-center gap-1 transition"
                >
                  {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  {copied ? 'Đã chép' : 'Sao chép'}
                </button>
                <pre>{data.pandas_query}</pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
```


        {/* 3. Diễn giải số liệu */}
### 3.4 Trang Chat Chính (`src/app/page.tsx`)
Tạo file `src/app/page.tsx`:

```tsx
'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Sparkles, RefreshCw } from 'lucide-react';
import BotMessageCard from '@/components/BotMessageCard';
import { ChatMessage, ChatResponseData } from '@/types/chat';

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      sender: 'bot',
      text: 'Xin chào! Tôi là Trợ lý Báo cáo Tài chính ViFinQA. Bạn có thể đặt câu hỏi về doanh thu, lợi nhuận, chi phí, tài sản của các doanh nghiệp niêm yết.'
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch('http://localhost:8000/api/suggestions')
      .then(res => res.json())
      .then(data => setSuggestions(data))
      .catch(() => {});
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async (questionText?: string) => {
    const q = (questionText || input).trim();
    if (!q || loading) return;

    const userMsg: ChatMessage = { id: Date.now().toString(), sender: 'user', text: q };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, mode: 'auto' })
      });
      const json = await res.json();
      
      const botMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        data: json.data as ChatResponseData
      };
      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      const errMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        sender: 'bot',
        text: 'Lỗi kết nối tới Backend (Port 8000). Vui lòng kiểm tra server.py.'
      };
      setMessages(prev => [...prev, errMsg]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-100 font-sans">
      <header className="bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-600 flex items-center justify-center text-white shadow-md shadow-blue-500/20">
            <Bot className="w-6 h-6" />
          </div>
          <div>
            <h1 className="font-bold text-slate-800 text-lg leading-tight">ViFinQA Financial Chatbot</h1>
            <p className="text-xs text-slate-500">Tra cứu &amp; Diễn giải Báo cáo Tài chính tự động bằng AI</p>
          </div>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6 max-w-4xl w-full mx-auto">
        {messages.map(msg => (
          <div key={msg.id} className={`flex gap-3 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.sender === 'bot' && (
              <div className="w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center shrink-0 mt-1 shadow">
                <Bot className="w-4 h-4" />
              </div>
            )}
            <div className={`max-w-[85%] ${msg.sender === 'user' ? 'bg-blue-600 text-white rounded-2xl px-4 py-3 shadow-sm' : 'w-full'}`}>
              {msg.text && <p className="text-sm leading-relaxed">{msg.text}</p>}
              {msg.data && <BotMessageCard data={msg.data} />}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex gap-3 items-center text-slate-500 text-sm">
            <RefreshCw className="w-4 h-4 animate-spin text-blue-600" />
            <span>Đang tra cứu dữ liệu BCTC và thực thi Pandas...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      <footer className="bg-white border-t border-slate-200 p-4 max-w-4xl w-full mx-auto rounded-t-2xl shadow-lg">
        {suggestions.length > 0 && messages.length <= 2 && (
          <div className="mb-3 flex flex-wrap gap-2">
            {suggestions.map((item, idx) => (
              <button
                key={idx}
                onClick={() => handleSend(item)}
                className="text-xs bg-slate-50 hover:bg-blue-50 hover:text-blue-600 border border-slate-200 text-slate-600 px-3 py-1.5 rounded-full transition"
              >
                {item}
              </button>
            ))}
          </div>
        )}
        <form onSubmit={e => { e.preventDefault(); handleSend(); }} className="flex items-center gap-2">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Nhập câu hỏi tài chính..."
            className="flex-1 border border-slate-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-100 rounded-xl px-4 py-3 text-sm outline-none"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white px-5 py-3 rounded-xl font-medium text-sm flex items-center gap-1.5"
          >
            <Send className="w-4 h-4" />
            <span>Gửi</span>
          </button>
        </form>
      </footer>
    </div>
  );
}
```

---

## 4. Hướng Dẫn Chạy Toàn Bộ Hệ Thống

1. **Khởi động Backend (FastAPI):**
   ```bash
   python server.py
   ```
   *(Server chạy tại `http://localhost:8000`)*

2. **Khởi động Frontend (Next.js):**
   ```bash
   cd frontend
   npm run dev
   ```
   *(Giao diện mở tại `http://localhost:3000`)*

        <div className="bg-blue-50/60 border border-blue-100 rounded-xl p-3.5 text-sm text-slate-700 flex items-start gap-2.5 leading-relaxed">
          <Info className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
          <div>{data.explanation}</div>
        </div>

}
```

## 3. Phần 2: Cài Đặt & Phát Triển Frontend (Next.js)

### 3.1 Khởi tạo dự án Next.js
Mở terminal mới (song song với Backend):
```bash
npx create-next-app@latest frontend --typescript --tailwind --eslint --app --src-dir --import-alias "@/*"
cd frontend
npm install lucide-react clsx tailwind-merge
```

### 3.2 Kiểu Dữ Liệu TypeScript (`src/types/chat.ts`)
Tạo file `src/types/chat.ts`:

```typescript
export interface Citation {
  doc_id: string;
  table_id: string;
  report_type: string;
  year: string;
  company: string;
}

export interface TableEvidence {
  variable: string;
  csv_path: string;
  table_name: string;
  columns: string[];
  rows: Record<string, any>[];
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
  safety: {
    confidence: string;
    warning: string | null;
  };
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'bot';
  text?: string;
  data?: ChatResponseData;
  loading?: boolean;
}
```

---

### 2.3 Khởi động Backend
```bash
python server.py
# hoặc
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
Kiểm tra API tại: `http://localhost:8000/docs`

---
