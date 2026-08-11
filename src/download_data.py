import os
from huggingface_hub import snapshot_download

def download_dataset():
    print("Downloading dataset from HuggingFace...")
    # Thư mục đích lưu dữ liệu
    target_dir = os.path.join(os.getcwd(), "data", "raw_vifinqa")
    os.makedirs(target_dir, exist_ok=True)
    
    # Tải snapshot repository từ huggingface
    snapshot_download(
        repo_id="AIGuruTinix/ViFinQA",
        repo_type="dataset",
        local_dir=target_dir,
        # bỏ qua thư mục .git để giảm size tải
        ignore_patterns=["*.git*"]
    )
    print(f"Dataset downloaded to {target_dir}")

if __name__ == "__main__":
    download_dataset()
