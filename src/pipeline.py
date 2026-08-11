import json
import zipfile
import os
import sys
from retriever import TableRetriever
from agent import PandasAgent

def run_full_pipeline(questions_file="questions.jsonl", 
                      output_json="submission.json", 
                      output_zip="submission.zip", 
                      max_questions=None,
                      checkpoint_interval=10):
    """
    Pipeline chính kết nối các thành phần với cơ chế Checkpointing:
    1. Tải checkpoint cũ từ output_json nếu tồn tại để tiếp tục xử lý.
    2. Đọc câu hỏi từ questions.jsonl
    3. Gọi TableRetriever tìm file CSV căn cứ
    4. Gọi PandasAgent (DeepSeek-R1:14b) sinh mã, lọc <think>, thực thi & tự sửa lỗi
    5. Ghi nhận dạng checkpoint sau mỗi checkpoint_interval câu hỏi.
    6. Ghi file submission.json và đóng gói submission.zip theo đúng quy định BTC.
    """
    print("=== Khởi động ViFinQA Pipeline ===")
    
    # Checkpoint loading
    results_map = {}
    used_csv_paths = set()
    
    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                for item in existing_data:
                    results_map[item["id"]] = item
                    for ev in item.get("evidence", []):
                        if "csv_path" in ev:
                            used_csv_paths.add(ev["csv_path"])
            print(f"[Checkpoint] Tìm thấy checkpoint cũ với {len(results_map)} câu hỏi đã hoàn thành.")
        except Exception as e:
            print(f"[Checkpoint] Lỗi khi đọc file checkpoint: {e}. Sẽ bắt đầu từ đầu.")

    # Khởi tạo mô-đun
    retriever = TableRetriever(csv_dir="data/processed_csv")
    agent = PandasAgent(model_name="deepseek-r1:14b", base_url="http://localhost:11434")
    
    if not os.path.exists(questions_file):
        print(f"Lỗi: Không tìm thấy file {questions_file}")
        return

    with open(questions_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    if max_questions:
        lines = lines[:max_questions]
        print(f"[Pipeline Config] Đang chạy thử nghiệm trên {max_questions} câu hỏi đầu tiên.")
    else:
        print(f"[Pipeline Config] Đang chạy trên toàn bộ {len(lines)} câu hỏi.")

    processed_count = 0
    for idx, line in enumerate(lines, 1):
        if not line.strip():
            continue
            
        q_data = json.loads(line)
        q_id = q_data["id"]
        question = q_data["question"]
        
        # Bỏ qua nếu đã có trong checkpoint
        if q_id in results_map:
            print(f"--- [{idx}/{len(lines)}] Question ID {q_id}: Đã xử lý (Skip) ---")
            continue

        print(f"\n--- [{idx}/{len(lines)}] Question ID {q_id}: {question[:60]}... ---")
        
        # 1. Retrieval Stage
        csv_paths = retriever.retrieve(question, top_k=3)
        print(f" -> Found evidence CSVs: {csv_paths}")
        
        # Lưu lại đường dẫn CSV để đóng gói vào ZIP
        for p in csv_paths:
            used_csv_paths.add(p)
            
        # 2. Agent Execution Stage
        ans, err = agent.run_agent(question, csv_paths, max_retries=3)
        print(f" -> Result: {ans}")
        if err:
            print(f" -> Error Notice: {err}")

        # 3. Format result item according to BTC schema
        evidence = [{"csv_path": p} for p in csv_paths]
        
        # Ép kiểu answer nếu có thể thành float/int hoặc giữ nguyên string
        formatted_ans = ans
        try:
            if "." in str(ans):
                formatted_ans = float(ans)
            else:
                formatted_ans = int(ans)
        except ValueError:
            formatted_ans = str(ans)

        results_map[q_id] = {
            "id": q_id,
            "answer": formatted_ans,
            "evidence": evidence
        }
        processed_count += 1

        # Save checkpoint periodically
        if processed_count % checkpoint_interval == 0:
            print(f"[Checkpoint] Tự động lưu checkpoint sau {processed_count} câu mới...")
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(list(results_map.values()), f, ensure_ascii=False, indent=2)

    # 4. Ghi submission.json hoàn chỉnh
    print(f"\n[Ghi File] Đang lưu tổng cộng {len(results_map)} kết quả ra {output_json}...")
    final_results = list(results_map.values())
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    # 5. Đóng gói submission.zip
    print(f"[Nén File] Đang tạo file {output_zip}...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # Thêm submission.json vào root của zip
        zf.write(output_json, arcname=os.path.basename(output_json))
        
        # Thêm các file CSV được tham chiếu vào thư mục data/ của zip
        for csv_path in used_csv_paths:
            real_file_path = csv_path if os.path.exists(csv_path) else csv_path.replace("data/", "", 1)
            if os.path.exists(real_file_path):
                arc_name = csv_path if csv_path.startswith("data/") else f"data/{os.path.basename(csv_path)}"
                zf.write(real_file_path, arcname=arc_name)
            else:
                print(f"[Cảnh báo Zip] File {real_file_path} không tồn tại trên đĩa, bỏ qua.")

    print(f"=== Hoàn thành! File {output_zip} sẵn sàng để nộp bài ===")

if __name__ == "__main__":
    max_q = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_full_pipeline(max_questions=max_q)

