import { Check, Circle, LoaderCircle, Search } from 'lucide-react';

export default function AssistantLoading() {
  return <div className="assistant-loading" role="status" aria-label="Đang phân tích câu hỏi">
    <div className="loading-orb"><Search size={18} /></div>
    <div><strong>Đang xây dựng câu trả lời có kiểm chứng</strong>
      <ol>
        <li className="done"><Check size={14} /> Nhận diện doanh nghiệp và kỳ báo cáo</li>
        <li className="active"><LoaderCircle size={14} /> Đang truy hồi bảng dữ liệu</li>
        <li><Circle size={10} /> Sinh truy vấn Pandas</li>
        <li><Circle size={10} /> Kiểm chứng kết quả</li>
      </ol>
    </div>
  </div>;
}
