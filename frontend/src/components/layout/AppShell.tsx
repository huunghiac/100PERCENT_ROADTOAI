'use client';

import { useEffect, useState, type ReactNode } from 'react';
import { Bot, ChartNoAxesCombined, ChevronLeft, Database, GitCompareArrows, History, Menu, MessageSquareText, PanelLeftOpen, X } from 'lucide-react';
import type { HealthResponse, WorkspaceView } from '@/types/chat';
import { readSidebarCollapsed, saveSidebarCollapsed } from '@/lib/storage';

const navigation: { id: WorkspaceView; label: string; icon: typeof Bot }[] = [
  { id: 'assistant', label: 'Trợ lý AI', icon: MessageSquareText },
  { id: 'compare', label: 'So sánh', icon: GitCompareArrows },
  { id: 'dashboard', label: 'Phân tích', icon: ChartNoAxesCombined },
  { id: 'data', label: 'Kho dữ liệu', icon: Database },
  { id: 'history', label: 'Lịch sử', icon: History },
];

interface Props {
  active: WorkspaceView;
  onNavigate: (view: WorkspaceView) => void;
  health: HealthResponse | null;
  online: boolean | null;
  children: ReactNode;
}

export default function AppShell({ active, onNavigate, health, online, children }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  useEffect(() => setCollapsed(readSidebarCollapsed()), []);

  function toggleCollapsed() {
    setCollapsed(value => { saveSidebarCollapsed(!value); return !value; });
  }

  function navigate(view: WorkspaceView) { onNavigate(view); setMobileOpen(false); }

  return <div className={`workspace ${collapsed ? 'sidebar-collapsed' : ''}`}>
    <aside className={`sidebar ${mobileOpen ? 'mobile-open' : ''}`} aria-label="Điều hướng chính">
      <div className="sidebar-brand">
        <div className="logo-mark"><Bot size={21} /></div>
        <div className="brand-copy"><strong>ViFinQA</strong><span>AI Financial Research</span></div>
        <button className="icon-button mobile-close" onClick={() => setMobileOpen(false)} aria-label="Đóng menu"><X size={19} /></button>
      </div>
      <nav className="sidebar-nav">
        <span className="nav-section-label">Workspace</span>
        {navigation.map(item => { const Icon = item.icon; return <button key={item.id} className={active === item.id ? 'nav-item active' : 'nav-item'} onClick={() => navigate(item.id)} title={collapsed ? item.label : undefined}>
          <Icon size={19} /><span>{item.label}</span>{active === item.id && <i className="active-rail" />}
        </button>; })}
      </nav>
      <div className="sidebar-system">
        <div className={`system-card ${online === false ? 'offline' : ''}`}>
          <div className="system-card-top"><span className="status-pulse" /><strong>{online === null ? 'Đang kiểm tra' : online ? 'Hệ thống Online' : 'Backend Offline'}</strong></div>
          <p>{online ? `${health?.available_tickers ?? 0} doanh nghiệp · ${health?.manifest_count.toLocaleString('vi-VN') ?? 0} bảng` : 'Có thể sử dụng dữ liệu minh hoạ.'}</p>
        </div>
        <button className="collapse-button" onClick={toggleCollapsed} aria-label={collapsed ? 'Mở rộng sidebar' : 'Thu gọn sidebar'}>
          {collapsed ? <PanelLeftOpen size={17} /> : <ChevronLeft size={17} />}<span>Thu gọn</span>
        </button>
      </div>
    </aside>
    {mobileOpen && <button className="sidebar-backdrop" onClick={() => setMobileOpen(false)} aria-label="Đóng menu" />}
    <div className="workspace-main">
      <button className="mobile-menu-button" onClick={() => setMobileOpen(true)} aria-label="Mở menu"><Menu size={20} /><span>ViFinQA</span></button>
      {children}
    </div>
    <nav className="mobile-nav" aria-label="Điều hướng di động">
      {navigation.map(item => { const Icon = item.icon; return <button key={item.id} className={active === item.id ? 'active' : ''} onClick={() => navigate(item.id)} aria-label={item.label}><Icon size={18} /><span>{item.label}</span></button>; })}
    </nav>
  </div>;
}
