# Hướng Dẫn Chạy ViFinQA Pipeline Trên JupyterLab GPU

Dùng cho môi trường thuê GPU như RunPod, Vast.ai, Lambda Labs, Paperspace hoặc server riêng có JupyterLab.

Cấu hình khuyên dùng hiện tại:

- **GPU**: NVIDIA RTX A6000 48GB VRAM
- **Model**: `Qwen/Qwen2.5-Coder-14B-Instruct`
- **Backend**: `transformers`
- **Precision**: `bfloat16`
- **Không cần quantize** 4-bit/8-bit

Repo:

- **GitHub**: `https://github.com/huunghiac/100PERCENT_ROADTOAI.git`
- **Hugging Face Dataset mirror**: `https://huggingface.co/datasets/huunghiac/vifinqa`

> Lưu ý: `_manifest.jsonl` được lưu bằng Git LFS. Cần `git lfs pull` sau khi clone.

---

## 1. Mở JupyterLab

Sau khi thuê GPU, mở link JupyterLab nhà cung cấp đưa.

Trong JupyterLab:

1. Chọn **File**.
2. Chọn **New**.
3. Chọn **Terminal**.
4. Chạy lệnh setup ở phần dưới.

---

## 2. Setup trong Terminal JupyterLab

Chạy toàn bộ block này trong Terminal:

```bash
cd /workspace 2>/dev/null || cd ~

apt-get update && apt-get install -y git-lfs

git lfs install

if [ ! -d "Road-to-AI" ]; then
  git clone https://github.com/huunghiac/100PERCENT_ROADTOAI.git Road-to-AI
fi

cd Road-to-AI

git pull

git lfs pull

python -m pip install --upgrade pip
python -m pip install -U transformers accelerate rank-bm25 pandas sentencepiece protobuf tqdm
python -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Kiểm tra nhanh:

```bash
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, round(p.total_memory / 1024**3, 2), 'GiB')
PY
```

Kỳ vọng với A6000:

```text
CUDA available: True
GPU count: 1
0 NVIDIA RTX A6000 47.xx GiB
```

---

## 3. Tạo Notebook

Trong JupyterLab:

1. Mở thư mục `Road-to-AI`.
2. Tạo notebook mới: **File** -> **New** -> **Notebook**.
3. Chọn kernel Python.
4. Lưu tên ví dụ: `run_qwen14b.ipynb`.

Notebook phải chạy từ thư mục repo `Road-to-AI`. Nếu không chắc, Cell 1 bên dưới tự `chdir`.

---

## 4. Cell 1 - Thiết lập thư mục làm việc và kiểm tra GPU

```python
import os
import sys
import torch

candidates = [
    os.getcwd(),
    '/workspace/Road-to-AI',
    os.path.expanduser('~/Road-to-AI'),
]

ROOT = None
for path in candidates:
    if os.path.exists(os.path.join(path, 'src')) and os.path.exists(os.path.join(path, 'data')):
        ROOT = path
        break

if ROOT is None:
    raise RuntimeError('Không tìm thấy repo Road-to-AI. Hãy mở notebook trong thư mục Road-to-AI.')

os.chdir(ROOT)
sys.path.insert(0, os.path.abspath('src'))

print('ROOT:', os.getcwd())
print('CUDA Available:', torch.cuda.is_available())
print('Device Count:', torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name} | VRAM: {p.total_memory / 1024**3:.2f} GiB')

print('questions.jsonl exists:', os.path.exists('data/raw_vifinqa/questions.jsonl'))
print('processed_csv exists:', os.path.exists('data/processed_csv'))
```

---

## 5. Cell 2 - Load model Qwen2.5-Coder-14B bằng Transformers

Model tải tự động từ Hugging Face:

`https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct`

Dung lượng tải lần đầu khoảng 28-30GB. Cache mặc định nằm ở `~/.cache/huggingface/hub`.

Nếu disk `/workspace` lớn hơn home, đặt cache vào `/workspace/hf_cache`:

```python
import os
import torch
from agent import PandasAgent

os.environ['HF_HOME'] = '/workspace/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/hf_cache'

agent = PandasAgent(
    model_name='Qwen/Qwen2.5-Coder-14B-Instruct',
    backend='transformers',
    torch_dtype=torch.bfloat16,
    max_length=6144,
)

print('PandasAgent loaded successfully!')
```

Nếu gặp lỗi `bfloat16` trên GPU khác, đổi sang:

```python
torch_dtype=torch.float16
```

Với RTX A6000, `torch.bfloat16` chạy được.

---

## 6. Cell 3 - Chạy thử 5 câu đầu

```python
from pipeline import run_full_pipeline

run_full_pipeline(
    questions_file='data/raw_vifinqa/questions.jsonl',
    output_json='submission_test5.json',
    output_zip='submission_test5.zip',
    max_questions=5,
    checkpoint_interval=1,
    agent=agent,
)
```

---

## 7. Cell 4 - Validate file test

```python
!python validate_submission.py submission_test5.json
```

Nếu validate OK, chạy full.

---

## 8. Cell 5 - Chạy toàn bộ bộ câu hỏi

```python
from pipeline import run_full_pipeline

run_full_pipeline(
    questions_file='data/raw_vifinqa/questions.jsonl',
    output_json='submission.json',
    output_zip='submission.zip',
    max_questions=None,
    checkpoint_interval=20,
    agent=agent,
)
```

---

## 9. Cell 6 - Validate submission cuối

```python
!python validate_submission.py submission.json
```

---

## 10. Cell 7 - Thống kê nhanh kết quả

```python
import json

with open('submission.json', 'r', encoding='utf-8') as f:
    rows = json.load(f)

total = len(rows)
nonzero = sum(1 for x in rows if float(x.get('answer', 0) or 0) != 0)
zero_ids = [x['id'] for x in rows if float(x.get('answer', 0) or 0) == 0]
gen_failed = [x['id'] for x in rows if 'GENERATION_FAILED' in x.get('pandas_query', '')]
fallback = [x['id'] for x in rows if 'FALLBACK' in x.get('pandas_query', '') or 'df.iloc' in x.get('pandas_query', '')]

print('TOTAL:', total)
print('NONZERO:', nonzero)
print('ZERO:', total - nonzero)
print('ZERO IDS:', zero_ids)
print('GENERATION_FAILED:', len(gen_failed), gen_failed)
print('FALLBACK-like:', len(fallback), fallback[:50])
```

---

## 11. Cell 8 - Tải file kết quả về máy

Trong JupyterLab sidebar:

1. Tìm `submission.zip`.
2. Chuột phải.
3. Chọn **Download**.

Hoặc link trong notebook:

```python
from IPython.display import FileLink

display(FileLink('submission.zip'))
display(FileLink('submission.json'))
```

---

## 12. Ước tính tài nguyên

| GPU | VRAM | Model | Precision | Kỳ vọng |
|---|---:|---|---|---|
| RTX A6000 | 48GB | Qwen2.5-Coder-14B | BF16 | Rất phù hợp |
| A100 40GB | 40GB | Qwen2.5-Coder-14B | BF16 | Phù hợp |
| RTX 4090 | 24GB | Qwen2.5-Coder-14B | 8-bit/4-bit | Cần quantize |
| T4 16GB | 16GB | Qwen2.5-Coder-14B | 4-bit | Không khuyên dùng |

A6000 48GB chạy model 14B full BF16 thoải mái:

- Weights: khoảng 28-30GB
- KV cache + runtime: khoảng 4-8GB
- Tổng thực tế: khoảng 32-38GB
- Còn dư VRAM: khoảng 10GB+

---

## 13. Lỗi thường gặp

### Lỗi 1: `CUDA out of memory`

Giảm context:

```python
max_length=4096
```

Hoặc restart kernel rồi load lại model.

### Lỗi 2: `ModuleNotFoundError`

Chạy lại trong Terminal:

```bash
cd /workspace/Road-to-AI
python -m pip install -U transformers accelerate rank-bm25 pandas sentencepiece protobuf tqdm
```

### Lỗi 3: Không thấy `data/raw_vifinqa/questions.jsonl`

Chạy lại:

```bash
cd /workspace/Road-to-AI
git lfs pull
find data -maxdepth 3 -type f | head
```

### Lỗi 4: Tải model Hugging Face chậm hoặc timeout

Tải trước bằng Terminal:

```bash
cd /workspace/Road-to-AI
export HF_HOME=/workspace/hf_cache
huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct
```
