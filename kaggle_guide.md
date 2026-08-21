# Hướng Dẫn Chạy Pipeline Trên Kaggle

## Bước 1: Tạo Kaggle Dataset

1. Vào [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. Đặt tên: `vifinqa-data`
3. Upload **2 thư mục** từ project:
   - `data/` (chứa `processed_csv/`, `raw_vifinqa/`)
   - `src/` (chứa `agent.py`, `retriever.py`, `pipeline.py`)
4. Nhấn **Create**

> **Lưu ý:** Thư mục `data/processed_csv/` có ~79k file (~197MB).
> Nên zip thành `processed_csv.zip` rồi upload, sau đó giải nén trên Kaggle.

## Bước 2: Tạo Kaggle Notebook

1. **New Notebook**
2. Cột phải → **Add Data** → tìm `vifinqa-data` → Add
3. **Settings:**
   - Accelerator: **GPU T4 x2** (16GB VRAM mỗi card)
   - Internet: **ON**
   - Persistence: **Files only**

## Bước 3: Copy các Cell sau vào Notebook

### Cell 1: Cài thư viện

```python
!pip install -q transformers accelerate rank-bm25
```

### Cell 2: Setup đường dẫn

```python
import os, sys, shutil

DATASET = 'vifinqa-data'  # ← sửa nếu đặt tên khác
INPUT = f'/kaggle/input/{DATASET}'
WORK = '/kaggle/working'

# Symlink data/
if not os.path.exists(f'{WORK}/data'):
    os.symlink(f'{INPUT}/data', f'{WORK}/data')

# Copy src/ (cần writable)
if not os.path.exists(f'{WORK}/src'):
    shutil.copytree(f'{INPUT}/src', f'{WORK}/src')

sys.path.insert(0, f'{WORK}/src')
os.chdir(WORK)

print('CSV files:', len(os.listdir('data/processed_csv')))
print('questions.jsonl:', os.path.exists('data/raw_vifinqa/questions.jsonl'))
```

### Cell 3: Kiểm tra GPU

```python
import torch
print(f'CUDA: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, '
          f'{props.total_mem / 1e9:.1f} GB VRAM')
```

### Cell 4: Test 10 câu đầu

```python
from pipeline import run_full_pipeline

run_full_pipeline(
    questions_file='data/raw_vifinqa/questions.jsonl',
    output_json='submission.json',
    output_zip='submission.zip',
    max_questions=10,
    checkpoint_interval=5,
)
```

### Cell 5: Kiểm tra kết quả

```python
import json

with open('submission.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

print(f'Total results: {len(results)}')
for r in results[:5]:
    print(f"  ID={r['id']} | answer={r['answer']}")
    print(f"    Q: {r['question'][:80]}")
    ev = r.get('evidence', [])
    print(f"    evidence: {ev}")
    print()
```

### Cell 6: Chạy FULL 1012 câu

```python
from pipeline import run_full_pipeline

run_full_pipeline(
    questions_file='data/raw_vifinqa/questions.jsonl',
    output_json='submission.json',
    output_zip='submission.zip',
    max_questions=None,
    checkpoint_interval=20,
)
```

### Cell 7: Download kết quả

```python
from IPython.display import FileLink
display(FileLink('submission.zip'))
```

## Tốc Độ Ước Tính

| Môi trường | VRAM | Tốc độ/câu | 1012 câu |
|---|---|---|---|
| Local Quadro T2000 (4GB) | 2.3GB on GPU | 5-10 phút | 3-7 ngày |
| **Kaggle T4 x2 (32GB)** | **Full on GPU** | **15-30 giây** | **4-8 giờ** |

## Lưu Ý

- Kaggle session tối đa **12 giờ** → đủ chạy 1012 câu.
- Pipeline có **checkpoint**: nếu bị ngắt, chạy lại Cell 6 sẽ tiếp tục từ chỗ dừng.
- Checkpoint chỉ giữ kết quả thật (có evidence + pandas_query), bỏ qua bản ghi dummy.
- Model `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` tải lần đầu ~9GB, lần sau dùng cache.
- `submission.zip` chứa `submission.json` + thư mục `data/` CSV → nộp trực tiếp cho BTC.
