import os
import re
import sys
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

sys.path.insert(0, "src")
from retriever import TableRetriever
from agent import PandasAgent
from fallback import try_rule_based_answer, detect_target_unit

app = FastAPI(title="ViFinQA Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = TableRetriever(csv_dir="data/processed_csv")
agent = None

class ChatRequest(BaseModel):
    question: str
    mode: Optional[str] = "auto"

def generate_explanation(question: str, answer: float, unit: str, matched_text: str, ticker: str, year: str) -> str:
    """Tạo lời diễn giải tài chính dễ hiểu cho người dùng không chuyên."""
    if answer == 0.0 or answer is None:
        return f"Hệ thống không tìm thấy số liệu phù hợp cho câu hỏi về mã {ticker or 'doanh nghiệp'} trong kỳ báo cáo {year or ''}."
    unit_str = unit if unit else "đơn vị chuẩn"
    formatted_num = f"{answer:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    clean_q = re.sub(r"^(cho biết|hãy cho biết|tính|xác định|hỏi)\s*", "", question, flags=re.IGNORECASE).strip()
    target_name = matched_text if matched_text else clean_q
    return f"Theo số liệu từ Báo cáo tài chính năm {year} của {ticker}, khoản mục **{target_name}** ghi nhận giá trị là **{formatted_num} {unit_str}**."

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "manifest_count": len(retriever.manifest),
        "available_tickers": len(os.listdir("data/processed_csv")) if os.path.exists("data/processed_csv") else 0
    }

@app.get("/api/suggestions")
def suggestions():
    return [
        "Lãi tiền gửi năm 2018 của công ty mẹ VJC là bao nhiêu triệu đồng?",
        "Doanh thu thuần của VNM năm 2023 là bao nhiêu tỷ đồng?",
        "Lợi nhuận sau thuế của ACB năm 2022 là bao nhiêu triệu đồng?",
        "Chi phí quản lý doanh nghiệp của HPG năm 2022 là bao nhiêu tỷ đồng?",
        "Tổng tài sản của BID năm 2021 là bao nhiêu tỷ đồng?"
    ]

@app.post("/api/chat")
def chat(req: ChatRequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Câu hỏi không được rỗng.")

    ticker, year = retriever.extract_entities(q)
    csv_paths = retriever.retrieve(q)
    target_unit = detect_target_unit(q) or ""

    if not csv_paths:
        return {
            "status": "not_found",
            "data": {
                "question": q,
                "answer": 0.0,
                "unit": target_unit,
                "formatted_answer": "0.0",
                "explanation": f"Không tìm thấy tài liệu báo cáo tài chính của công ty {ticker or 'này'} năm {year or 'này'}.",
                "citations": [],
                "evidence_tables": [],
                "pandas_query": "",
                "safety": {"confidence": "none", "warning": "Không có bảng dữ liệu tương ứng."}
            }
        }

    fallback_res = try_rule_based_answer(q, csv_paths)
    ans = None
    pandas_code = ""
    evidence_csv = csv_paths[0]
    matched_row_idx = 0
    matched_chi_tieu = ""

    if fallback_res and fallback_res.score >= 12.0:
        ans = fallback_res.answer
        pandas_code = fallback_res.pandas_query
        evidence_csv = fallback_res.csv_path
        matched_row_idx = fallback_res.row_index
    elif req.mode == "auto":
        # Skip PandasAgent - only use fallback for immediate response
        if fallback_res:
            ans = fallback_res.answer
            pandas_code = fallback_res.pandas_query

    if ans is None:
        ans = 0.0

    evidence_tables = []
    real_csv = evidence_csv if os.path.exists(evidence_csv) else os.path.join("data", "processed_csv", ticker or "", os.path.basename(evidence_csv))
    if os.path.exists(real_csv):
        try:
            df = pd.read_csv(real_csv)
            if 0 <= matched_row_idx < len(df):
                matched_chi_tieu = str(df.iloc[matched_row_idx].get("Chi_tieu", ""))
            
            start_i = max(0, matched_row_idx - 2)
            end_i = min(len(df), matched_row_idx + 4)
            preview_df = df.iloc[start_i:end_i].fillna("")

            evidence_tables.append({
                "variable": "df1",
                "csv_path": f"data/{os.path.basename(real_csv)}",
                "table_name": os.path.basename(real_csv).replace(".csv", ""),
                "columns": list(df.columns),
                "rows": preview_df.to_dict(orient="records"),
                "highlight_row_index": matched_row_idx - start_i
            })
        except Exception:
            pass

    explanation = generate_explanation(q, ans, target_unit, matched_chi_tieu, ticker or "", year or "")
    formatted_answer = f"{ans:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if target_unit:
        formatted_answer += f" {target_unit}"

    return {
        "status": "success" if ans != 0.0 else "zero_result",
        "data": {
            "question": q,
            "answer": ans,
            "unit": target_unit,
            "formatted_answer": formatted_answer,
            "explanation": explanation,
            "citations": [
                {
                    "doc_id": f"{ticker}_financial_statements_{year}",
                    "table_id": f"{ticker}_{year}_{os.path.basename(evidence_csv)}",
                    "report_type": "Báo cáo tài chính",
                    "year": year,
                    "company": ticker
                }
            ],
            "evidence_tables": evidence_tables,
            "pandas_query": pandas_code,
            "safety": {
                "confidence": "high" if ans != 0.0 else "low",
                "warning": None if ans != 0.0 else "Số liệu = 0.0 hoặc không khớp chỉ tiêu trong bảng."
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
