import React, { useEffect } from 'react';

interface ShortcutsModalProps {
  onClose: () => void;
}

const SHORTCUTS = [
  { group: '편집', items: [
    { key: 'Ctrl + Z',  desc: '실행 취소 (Undo)' },
    { key: 'Ctrl + Y',  desc: '다시 실행 (Redo)' },
    { key: 'Ctrl + C',  desc: '모듈 복사' },
    { key: 'Ctrl + V',  desc: '모듈 붙여넣기' },
    { key: 'Ctrl + X',  desc: '모듈 잘라내기' },
    { key: 'Delete',    desc: '선택 모듈 삭제' },
  ]},
  { group: '선택', items: [
    { key: 'Ctrl + A',  desc: '모든 모듈 선택' },
    { key: 'Click',     desc: '단일 모듈 선택' },
    { key: 'Ctrl + Click', desc: '다중 선택 추가/제거' },
    { key: '드래그',    desc: '범위 선택 (빈 공간에서)' },
  ]},
  { group: '뷰 & 탐색', items: [
    { key: 'Ctrl + =',  desc: '확대' },
    { key: 'Ctrl + -',  desc: '축소' },
    { key: 'Ctrl + 0',  desc: '전체 화면 맞춤' },
    { key: 'Space + 드래그', desc: '캔버스 이동 (패닝)' },
    { key: '스크롤',    desc: '확대/축소' },
  ]},
  { group: '파일', items: [
    { key: 'Ctrl + S',  desc: '파이프라인 저장' },
    { key: '드래그 & 드롭', desc: '.ins/.json 파일 캔버스에 로드' },
  ]},
  { group: '연결선', items: [
    { key: '포트 드래그', desc: '모듈 연결 생성' },
    { key: '더블 클릭',  desc: '연결선 삭제' },
    { key: '⚡ 클릭',   desc: '연결 데이터 전체 보기' },
    { key: '⚡ 호버',   desc: '연결 데이터 요약 미리보기' },
  ]},
];

export const ShortcutsModal: React.FC<ShortcutsModalProps> = ({ onClose }) => {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-xl shadow-2xl w-[520px] max-h-[85vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
          <h2 className="text-sm font-bold text-gray-900 dark:text-white">키보드 단축키</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors text-xl leading-none"
          >
            ×
          </button>
        </div>

        {/* Body */}
        <div className="overflow-y-auto flex-1 p-5 grid grid-cols-2 gap-5">
          {SHORTCUTS.map(group => (
            <div key={group.group}>
              <p className="text-[10px] font-bold text-gray-400 dark:text-gray-500 uppercase tracking-widest mb-2">
                {group.group}
              </p>
              <div className="space-y-1.5">
                {group.items.map(item => (
                  <div key={item.key} className="flex items-center gap-3">
                    <kbd className="flex-shrink-0 px-2 py-0.5 text-[10px] font-mono bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded text-gray-700 dark:text-gray-300 whitespace-nowrap">
                      {item.key}
                    </kbd>
                    <span className="text-xs text-gray-600 dark:text-gray-400">{item.desc}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="px-5 py-3 border-t border-gray-200 dark:border-gray-700 flex-shrink-0">
          <p className="text-[10px] text-gray-400">클릭하거나 Esc로 닫기</p>
        </div>
      </div>
    </div>
  );
};
