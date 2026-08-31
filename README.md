# ViFinQA — Text-to-Pandas Financial QA

Source code dự thi cho bài toán trả lời câu hỏi tài chính tiếng Việt từ báo cáo OCR. Hệ thống trích xuất bảng thành CSV, phân tích câu hỏi, truy hồi evidence, dựng biểu thức Pandas, replay đáp án và đóng gói `submission.zip`.

## Cấu trúc

```text
Road-to-AI/
├── src/
│   ├── download_data.py          # tải ViFinQA từ Hugging Face
│   ├── data_extractor.py         # OCR TXT -> CSV + manifest
│   ├── build_line_index.py       # doc/table -> dòng nguồn
│   ├── question_planner.py       # entity, metric, phép toán
│   ├── metric_registry.py        # chuẩn hóa chỉ tiêu tài chính
│   ├── retriever.py              # hard filter + BM25 retrieval
│   ├── fallback.py               # solver luật cho câu đơn
│   ├── complex_solver.py         # solver câu nhiều bước
│   ├── agent.py                  # backend Transformers/Ollama
│   ├── query_formatter.py        # parse và thực thi query an toàn
│   ├── semantic_validation.py    # gate ngữ nghĩa và unit
│   ├── units.py                  # chuẩn hóa đơn vị
│   ├── pipeline.py               # pipeline và packaging chính
│   ├── submission_contract.py    # schema, closure, ZIP replay
│   └── evaluator.py              # evaluator local
├── tests/                        # unit, integration, regression
├── data/
│   ├── raw_vifinqa/              # dataset OCR và questions từ BTC
│   ├── processed_csv/            # CSV, manifest đã trích xuất
│   ├── quality_reports/          # báo cáo chất lượng extractor
│   └── table_line_map.json       # provenance doc/table -> dòng nguồn
├── run_gpu_vifinqa.ipynb         # quy trình GPU đầy đủ
├── validate_submission.py        # validator JSON/ZIP độc lập
├── server.py                     # FastAPI demo tùy chọn
└── environment.yml               # môi trường Conda cơ bản
```

Repo commit toàn bộ `data/` để có thể chạy trực tiếp. Script download/extract vẫn được giữ để tái tạo hoặc cập nhật dataset. Log, cache và submission output không được commit.

## Yêu cầu và cài đặt

- Python 3.11
- GPU CUDA khuyến nghị cho backend Transformers
- Model mã nguồn mở không quá 14B theo luật cuộc thi

```bash
conda env create -f environment.yml
conda activate financial
python -m pip install -U pytest huggingface_hub rank-bm25
```

Cài GPU/LLM và notebook khi cần:

```bash
python -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -U transformers accelerate sentencepiece protobuf tqdm ipywidgets
```

Demo API cần thêm `fastapi uvicorn`.

Từ root repo, đặt module source vào import path:

```powershell
# PowerShell
$env:PYTHONPATH=(Get-Location).Path
```

```bash
# Bash
export PYTHONPATH="$PWD"
```

## Chuẩn bị dữ liệu

Tải dataset:

```bash
python src/download_data.py
```

Dữ liệu được đặt tại `data/raw_vifinqa/`.

Trích xuất thử 10 báo cáo:

```bash
python src/data_extractor.py --limit-files 10 --verbose
```

Trích xuất toàn bộ và làm sạch output cũ:

```bash
python src/data_extractor.py --clean
```

Output nằm tại `data/processed_csv/`, gồm CSV theo ticker và `_manifest.jsonl`. Có thể chạy hẹp:

```bash
python src/data_extractor.py --ticker VJC --year 2018 --clean
```

Tạo provenance line index:

```bash
python src/build_line_index.py
```

Output: `data/table_line_map.json`.

## Chạy pipeline

Chạy toàn bộ câu hỏi:

```bash
python src/pipeline.py
```

Smoke test tối đa 100 câu:

```bash
python src/pipeline.py 100
```

Pipeline mặc định đọc `data/raw_vifinqa/questions.jsonl`, dùng `data/processed_csv/` và tạo:

```text
submission.json
submission.failures.json
submission.quality.json
submission.zip
```

ZIP chỉ được tạo khi toàn bộ ID được account, schema hợp lệ và không còn failure. `run_gpu_vifinqa.ipynb` cung cấp luồng clone, cài dependency, test, chạy full và đóng gói trên Jupyter GPU.

## Kiểm submission

Kiểm JSON/schema/accounting:

```bash
python validate_submission.py submission.json \
  --questions data/raw_vifinqa/questions.jsonl \
  --failures submission.failures.json \
  --require-complete
```

Kiểm thêm ZIP-only replay:

```bash
python validate_submission.py submission.json --zip submission.zip
```

ZIP gate chỉ dùng byte trong archive và kiểm:

- đúng một `submission.json`;
- chỉ có `submission.json` và `data/<flat-name>.csv`;
- không duplicate entry;
- evidence/CSV closure hai chiều;
- query chỉ dùng evidence variables đã khai báo;
- provenance docs/tables/evidence nhất quán;
- query replay thành công và khớp `answer`.

CLI trả exit code khác 0 khi gate fail.

## Chạy test

```bash
pytest -q --disable-warnings --maxfail=1
```

Kiểm nhanh contract:

```bash
python -m py_compile src/submission_contract.py validate_submission.py
pytest -q tests/test_submission_contract.py tests/test_submission_eval.py tests/test_query_eval.py
```

## Demo API

Sau khi tạo `data/processed_csv/`:

```bash
python -m pip install -U fastapi uvicorn
python server.py
```

API mặc định tại `http://localhost:8000`; health check: `GET /api/health`.

## Submission contract

Mỗi item có đúng bảy field:

```text
id, question, answer, relevant_docs, relevant_tables, evidence, pandas_query
```

`evidence[].csv_path` có dạng `data/<flat-name>.csv`. `pandas_query` phải tái tạo chính xác `answer` từ evidence trong ZIP. Query chạy được chỉ chứng minh consistency nội bộ; điểm benchmark còn phụ thuộc retrieval và ground truth.
