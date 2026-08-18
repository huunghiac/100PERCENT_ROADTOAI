"""
HƯỚNG DẪN CHẠY TRÊN KAGGLE
============================
1. Tạo Kaggle Dataset tên "vifinqa-data", upload 2 thứ:
   - Thư mục data/ (chứa processed_csv/, raw_vifinqa/)
   - Thư mục src/ (chứa agent.py, retriever.py, pipeline.py)

2. Tạo Kaggle Notebook:
   - Accelerator: GPU T4 x2
   - Internet: ON
   - Add dataset "vifinqa-data"

3. Copy từng cell bên dưới vào notebook, Run All.
"""

# ===================== CELL 1 =====================
# !pip install -q transformers accelerate rank-bm25

# ===================== CELL 2 =====================
"""
import os, sys, shutil

DATASET = 'vifinqa-data'   # <-- sửa đúng tên dataset bạn upload
INPUT = f'/kaggle/input/{DATASET}'
WORK  = '/kaggle/working'

# Symlink data
if not os.path.exists(f'{WORK}/data'):
    os.symlink(f'{INPUT}/data', f'{WORK}/data')

# Copy src
if not os.path.exists(f'{WORK}/src'):
    shutil.copytree(f'{INPUT}/src', f'{WORK}/src')

sys.path.insert(0, f'{WORK}/src')
os.chdir(WORK)

print('CSV count:', len(os.listdir('data/processed_csv')))
print('questions.jsonl:', os.path.exists('data/raw_vifinqa/questions.jsonl'))
"""

# ===================== CELL 3 =====================
"""
import torch
print(f'CUDA: {torch.cuda.is_available()}')
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, {mem:.1f} GB')
"""

# ===================== CELL 4: Test 10 câu =====================
"""
from pipeline import run_full_pipeline

run_full_pipeline(
    questions_file='data/raw_vifinqa/questions.jsonl',
    output_json='submission.json',
    output_zip='submission.zip',
    max_questions=10,
    checkpoint_interval=5,
)
"""

# ===================== CELL 5: Xem kết quả =====================
"""
import json
with open('submission.json') as f:
    results = json.load(f)
print(f'Total: {len(results)}')
for r in results[:5]:
    print(f"  ID={r['id']} answer={r['answer']}")
    print(f"    Q: {r['question'][:80]}")
"""

# ===================== CELL 6: Full 1012 câu =====================
"""
run_full_pipeline(
    questions_file='data/raw_vifinqa/questions.jsonl',
    output_json='submission.json',
    output_zip='submission.zip',
    max_questions=None,
    checkpoint_interval=20,
)
"""

# ===================== CELL 7: Download =====================
"""
from IPython.display import FileLink
FileLink('submission.zip')
"""
