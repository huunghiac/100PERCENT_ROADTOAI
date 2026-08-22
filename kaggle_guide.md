# Hướng Dẫn Chạy ViFinQA Pipeline Trên Kaggle

Repo GitHub và Hugging Face Dataset đồng bộ tự động qua GitHub Actions. Clone từ nguồn nào cũng được (code + data giống nhau).

- **GitHub**: `https://github.com/huunghiac/100PERCENT_ROADTOAI.git`
- **Hugging Face**: `https://huggingface.co/datasets/huunghiac/vifinqa`

> **Lưu ý:** `_manifest.jsonl` (105MB) được lưu bằng Git LFS. Cần `git lfs pull` sau khi clone.

---

## Cấu Hình Môi Trường Kaggle

1. Tạo **New Notebook** trên Kaggle.
2. Cột cấu hình bên phải (Notebook Settings):
   - **Accelerator**: `GPU T4 x2` (Tổng 30GB VRAM) hoặc `GPU P100` (16GB VRAM)
   - **Internet**: `ON` (bắt buộc để tải model và kéo dữ liệu)
   - **Persistence**: `Files only` (khuyên dùng để giữ file tạm)

---

## Các Cell Cần Chạy Trong Notebook

### Cell 1: Cài đặt thư viện phụ thuộc

```python
!pip install -q transformers accelerate rank-bm25
```

---

### Cell 2: Clone repo và thiết lập đường dẫn

```python
import os, sys

WORK = '/kaggle/working'
os.chdir(WORK)

# Clone repo (chứa cả code lẫn data)
if not os.path.exists(f'{WORK}/vifinqa'):
    !git clone https://huggingface.co/datasets/huunghiac/vifinqa {WORK}/vifinqa
    !cd {WORK}/vifinqa && git lfs pull

# Thiết lập đường dẫn làm việc
if not os.path.exists(f'{WORK}/data'):
    os.symlink(f'{WORK}/vifinqa/data', f'{WORK}/data')

sys.path.insert(0, f'{WORK}/vifinqa/src')
os.chdir(WORK)

# Kiểm tra dữ liệu
print('=== Kiểm tra Setup ===')
print('Ticker directories:', len([d for d in os.listdir('data/processed_csv') if os.path.isdir(f'data/processed_csv/{d}')]) )
print('questions.jsonl exists:', os.path.exists('data/raw_vifinqa/questions.jsonl'))
print('manifest size:', round(os.path.getsize('data/processed_csv/_manifest.jsonl') / 1024 / 1024, 2), 'MB')
```

---

### Cell 3: Kiểm tra phần cứng GPU

```python
import torch

print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'Device Count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} | VRAM: {p.total_memory / 1e9:.2f} GB')
```

---

### Cell 4: Tải và khởi tạo mô hình DeepSeek

Model ~9GB, tải lần đầu mất ~2-3 phút, lưu vào GPU memory tự động:

```python
from agent import PandasAgent

agent = PandasAgent(
    model_name='deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
    backend='transformers',
    torch_dtype=torch.float16
)
print('PandasAgent loaded successfully!')
```

---

### Cell 5: Chạy thử nghiệm 5 câu đầu

```python
from pipeline import run_full_pipeline

run_full_pipeline(
    questions_file='data/raw_vifinqa/questions.jsonl',
    output_json='submission.json',
    output_zip='submission.zip',
    max_questions=5,
    checkpoint_interval=2
)
```

---

### Cell 6: Kiểm tra tính hợp lệ (Schema BTC)

```python
import sys
sys.path.insert(0, f'{WORK}/vifinqa')
!python {WORK}/vifinqa/validate_submission.py submission.json
```

---

### Cell 7: Chạy toàn bộ (1,012 câu hỏi)

```python
from pipeline import run_full_pipeline

run_full_pipeline(
    questions_file='data/raw_vifinqa/questions.jsonl',
    output_json='submission.json',
    output_zip='submission.zip',
    max_questions=None,
    checkpoint_interval=20
)
```

---

### Cell 8: Tải file submission.zip về máy

```python
from IPython.display import FileLink
display(FileLink('submission.zip'))
```

---

## Cách Biến Thành Kaggle Dataset Vĩnh Viễn (Tùy chọn)

Nếu không muốn mỗi lần bật session phải `git clone` lại ~4-5 phút:

1. Chạy **Cell 1 + Cell 2** xong.
2. Bấm nút **Save Version** (góc trên bên phải) -> Chọn **Save & Run All (Commit)**.
3. Chờ notebook chạy xong -> Vào [kaggle.com/datasets](https://www.kaggle.com/datasets) -> Bấm **+ New Dataset**.
4. Chọn nguồn **Notebook Output** -> Chọn Notebook vừa chạy.
5. Đặt tên (vd: `vifinqa-dataset`) -> Bấm **Create**.
6. Ở các notebook sau, chỉ cần bấm **+ Add Input** -> chọn Dataset. Dữ liệu sẽ nằm sẵn ở `/kaggle/input/vifinqa-dataset/` và load trong 0 giây!

---

## Bảng Ước Tính Thời Gian & Tài Nguyên

| Phần Cứng | Cấu Hình VRAM | Tốc Độ Dự Kiến | Thời Gian (1012 câu) |
|---|---|---|---|
| Local RTX 3050/T2000 (4GB) | Offload RAM nặng | 5-10 phút / câu | 3-7 ngày |
| **Kaggle GPU T4 x2 (30GB)** | **100% On GPU** | **12-25 giây / câu** | **~3.5 - 5 giờ** |
| Kaggle GPU P100 (16GB) | 100% On GPU (fp16) | 15-30 giây / câu | ~4 - 6 giờ |

*(Kaggle hỗ trợ tối đa 12 giờ chạy liên tục, đủ sức hoàn thành trong 1 lần chạy)*
