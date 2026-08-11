import os
import glob
import csv
import re

def extract_tables_from_txt(txt_path, output_dir, company, year, report_type):
    """
    Nhiệm vụ: Trích xuất dữ liệu bảng HTML-like từ file .txt OCR (dùng RegEx thay vì BeautifulSoup để chạy ngay).
    Lưu dưới dạng .csv
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Tìm các đoạn văn bản trong thẻ <table>...</table>
    tables = re.findall(r'<table.*?>(.*?)</table>', content, re.DOTALL | re.IGNORECASE)
    
    for i, table_html in enumerate(tables):
        rows = re.findall(r'<tr.*?>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
        table_data = []
        for row in rows:
            # Lấy nội dung trong thẻ <td> hoặc <th>
            cells = re.findall(r'<t[dh].*?>(.*?)</t[dh]>', row, re.DOTALL | re.IGNORECASE)
            # Làm sạch thẻ HTML con bề trong cell
            clean_cells = [re.sub(r'<.*?>', '', cell).strip() for cell in cells]
            # Loại bỏ dòng rỗng hoàn toàn
            if any(c != "" for c in clean_cells):
                table_data.append(clean_cells)
        
        if table_data:
            csv_filename = f"{company}_{year}_{report_type}_Table{i+1}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(table_data)

def process_all_reports(raw_dir, processed_dir, max_companies=2):
    """
    Xử lý các báo cáo. Giới hạn 2 công ty để làm baseline cho Data Engineer.
    Cấu trúc raw: AAA/2015/...
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    companies = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    
    for company in companies[:max_companies]:
        comp_dir = os.path.join(raw_dir, company)
        years = [d for d in os.listdir(comp_dir) if os.path.isdir(os.path.join(comp_dir, d))]
        
        for year in years:
            year_dir = os.path.join(comp_dir, year)
            txt_files = glob.glob(f"{year_dir}/**/*.txt", recursive=True)
            for txt_file in txt_files:
                # Phân loại báo cáo hợp nhất hay riêng lẻ từ tên file
                report_type = "consolidated" if "consolidated" in txt_file.lower() else "separate"
                extract_tables_from_txt(txt_file, processed_dir, company, year, report_type)
    
    print(f"Extraction completed. Checked {max_companies} companies.")

if __name__ == "__main__":
    raw_dir = os.path.join("data", "raw_vifinqa", "financial_statements")
    processed_dir = os.path.join("data", "processed_csv")
    process_all_reports(raw_dir, processed_dir)

