# Hướng Dẫn Chạy ViFinQA Pipeline Trên JupyterLab GPU

Dùng cho môi trường thuê GPU như RunPod, Vast.ai, Lambda Labs, Paperspace hoặc server riêng có JupyterLab.

## Mục tiêu

Chạy pipeline sau các fix mới:

- Strip submission chỉ còn 7 field BTC
- Fix nhầm multi-ticker do ticker trong ngoặc đơn
- Fix unit mismatch do detect nhầm "cổ phần", ratio/currency
- Bổ sung 10 metric definitions
- Prompt guard chống `.iloc[0]` trên DataFrame rỗng
- Fallback rule-based cho `missing_metric_facts`/`no_mapped_evidence`

Output cần gửi lại để đánh giá:

```text
submission.json
submission.zip
submission.failures.json
submission.quality.json
pipeline_full.log
pipeline_smoke100.log
gpu_run_artifacts.zip
```

---

## 1. Cấu hình khuyên dùng

| Thành phần | Khuyên dùng |
|---|---|
| GPU | NVIDIA A40 48GB, RTX A6000 48GB, A100 40GB+ |
| Model | `Qwen/Qwen2.5-Coder-14B-Instruct` |
| Backend | `transformers` |
| Precision | `float16` hoặc `bfloat16` |
| Quantize | Không cần với 40GB+ VRAM |

Repo:

```text
https://github.com/huunghiac/100PERCENT_ROADTOAI.git
```

Lưu ý: `data/processed_csv/_manifest.jsonl` dùng Git LFS. Phải chạy `git lfs pull`.

---

## 2. Cách nhanh nhất

Upload/chạy notebook có sẵn:

```text
run_gpu_vifinqa.ipynb
```

Trong notebook chỉ cần sửa nếu cần:

```python
REPO_URL = "https://github.com/huunghiac/100PERCENT_ROADTOAI.git"
BRANCH = "main"
ROOT = "/workspace/Road-to-AI"
MODEL_NAME = "Qwen/Qwen2.5-Coder-14B-Instruct"
RUN_SMOKE_100 = True
RUN_FULL = True
```

Sau đó chạy tuần tự tất cả cells.

---

## 3. Nếu setup bằng Terminal JupyterLab

```bash
cd /workspace 2>/dev/null || cd ~
apt-get update && apt-get install -y git-lfs

git lfs install

if [ ! -d "Road-to-AI" ]; then
  git clone https://github.com/huunghiac/100PERCENT_ROADTOAI.git Road-to-AI
fi

cd Road-to-AI
git fetch origin
git checkout main
git pull origin main
git lfs pull

python -m pip install --upgrade pip
python -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -U transformers accelerate rank-bm25 pandas sentencepiece protobuf tqdm pytest ipywidgets huggingface_hub
```

Nếu cài torch xong kernel lỗi, restart kernel/Jupyter rồi chạy tiếp.

---

## 4. Kiểm tra GPU

```bash
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, round(p.total_memory / 1024**3, 2), 'GiB')
PY
nvidia-smi
```

Kỳ vọng:

```text
CUDA available: True
GPU count: 1
0 NVIDIA RTX A6000 47.xx GiB
```

---

## 5. Kiểm tra code trước khi chạy

Trong repo:

```bash
python -m pytest tests/ -x -q
```

Kỳ vọng local hiện tại:

```text
202 passed
```

Nếu fail, dừng lại. Gửi log lỗi.

---

## 6. Không dùng checkpoint cũ

Checkpoint cũ có thể chứa field nội bộ hoặc kết quả trước fix. Backup trước khi chạy:

```bash
python - <<'PY'
from pathlib import Path
from datetime import datetime
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
for name in ['submission.json','submission.zip','submission.failures.json','submission.quality.json','pipeline_full.log','pipeline_smoke100.log']:
    p = Path(name)
    if p.exists():
        new = p.with_name(f'{p.stem}.before_{stamp}{p.suffix}')
        p.rename(new)
        print('backup', p, '->', new)
PY
```

---

## 7. Chạy smoke 100 câu

Trong notebook dùng `agent` đã load. Nếu chạy script trực tiếp:

```bash
python src/pipeline.py 100 2>&1 | tee pipeline_smoke100.log
```

Khuyên dùng notebook vì notebook truyền sẵn `PandasAgent` 14B vào `run_full_pipeline()`.

Sau smoke, kiểm tra:

```bash
python validate_submission.py submission_smoke100.json
```

Kiểm tra 7 field BTC:

```bash
python - <<'PY'
import json
allowed = {'id','question','answer','relevant_docs','relevant_tables','evidence','pandas_query'}
with open('submission_smoke100.json', encoding='utf-8') as f:
    rows = json.load(f)
bad = [(x.get('id'), sorted(set(x)-allowed)) for x in rows if set(x)-allowed]
print('saved:', len(rows))
print('invalid fields:', bad[:10])
assert not bad
PY
```

---

## 8. Chạy full 1012 câu

Trong notebook:

```python
run_full_pipeline(
    questions_file='data/raw_vifinqa/questions.jsonl',
    output_json='submission.json',
    output_zip='submission.zip',
    max_questions=None,
    checkpoint_interval=20,
    agent=agent,
)
```

Hoặc CLI:

```bash
python src/pipeline.py 2>&1 | tee pipeline_full.log
```

Notebook tốt hơn vì giữ model loaded trong memory và truyền `agent` vào pipeline.

---

## 9. Validate output cuối

```bash
python validate_submission.py submission.json
```

Kiểm tra 7 field:

```bash
python - <<'PY'
import json
allowed = {'id','question','answer','relevant_docs','relevant_tables','evidence','pandas_query'}
with open('submission.json', encoding='utf-8') as f:
    rows = json.load(f)
bad = [(x.get('id'), sorted(set(x)-allowed)) for x in rows if set(x)-allowed]
print('saved:', len(rows))
print('invalid fields:', bad[:10])
assert not bad
PY
```

Thống kê failures:

```bash
python - <<'PY'
import json, collections
p = 'submission.failures.json'
with open(p, encoding='utf-8') as f:
    rows = json.load(f)
print('failures:', len(rows))
print(collections.Counter(x.get('code') or x.get('stage') for x in rows).most_common(30))
PY
```

---

## 10. Package artifact gửi lại

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
files = [
    'submission.json', 'submission.zip', 'submission.failures.json', 'submission.quality.json',
    'pipeline_full.log', 'pipeline_smoke100.log',
    'submission_smoke100.json', 'submission_smoke100.zip',
    'submission_smoke100.failures.json', 'submission_smoke100.quality.json',
]
with ZipFile('gpu_run_artifacts.zip', 'w', ZIP_DEFLATED) as z:
    for name in files:
        p = Path(name)
        if p.exists():
            z.write(p, p.name)
            print('add', p)
print('created gpu_run_artifacts.zip')
PY
```

Gửi lại:

```text
gpu_run_artifacts.zip
submission.json
submission.failures.json
submission.quality.json
pipeline_full.log
```

---

## 11. Lỗi thường gặp

### `CUDA out of memory`

Dùng GPU 40GB+ cho 14B. Nếu vẫn OOM:

- Restart kernel
- Đảm bảo không load model nhiều lần
- Giảm `prompt_token_budget` còn `4096`
- Dùng model 7B nếu bắt buộc:

```python
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
```

### Tải model Hugging Face chậm

```bash
export HF_HOME=/workspace/hf_cache
huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct
```

### Không thấy data

```bash
cd /workspace/Road-to-AI
git lfs pull
find data -maxdepth 3 -type f | head
```

### `ModuleNotFoundError`

```bash
python -m pip install -U transformers accelerate rank-bm25 pandas sentencepiece protobuf tqdm pytest
```

### Notebook không ở đúng folder

Cell config tự `chdir(ROOT)`. Nếu repo clone nơi khác, sửa:

```python
ROOT = "/path/to/Road-to-AI"
```

---

## 12. Sau khi chạy xong

Đưa lại artifact. Sẽ phân tích tiếp:

- Saved count mới
- `missing_target_metric` giảm bao nhiêu
- `missing_metric_facts` fallback cứu được bao nhiêu
- `incompatible_target_unit` còn bao nhiêu
- IndexError còn không
- Query execution format đúng không
- Có regression answer không
