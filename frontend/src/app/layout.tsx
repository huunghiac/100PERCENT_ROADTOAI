import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'ViFinQA | AI Financial Research Assistant',
  description: 'Tra cứu, so sánh và kiểm chứng số liệu báo cáo tài chính Việt Nam bằng AI.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="vi"><body>{children}</body></html>;
}
