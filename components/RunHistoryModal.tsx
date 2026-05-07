import React, { useState } from 'react';
import { ModuleStatus } from '../types';

export interface RunHistorySession {
  id: string;
  timestamp: number;
  results: Array<{
    name: string;
    type: string;
    status: ModuleStatus;
    executionTime?: number;
  }>;
}

interface RunHistoryModalProps {
  sessions: RunHistorySession[];
  onClose: () => void;
}

function fmtTime(ms?: number): string {
  if (ms === undefined) return '—';
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

function fmtDate(ts: number): string {
  const d = new Date(ts);
  return d.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function statusColor(status: ModuleStatus): string {
  switch (status) {
    case ModuleStatus.Success: return 'text-green-400';
    case ModuleStatus.Error:   return 'text-red-400';
    case ModuleStatus.Running: return 'text-yellow-400';
    default: return 'text-gray-500';
  }
}

function statusIcon(status: ModuleStatus): string {
  switch (status) {
    case ModuleStatus.Success: return '✓';
    case ModuleStatus.Error:   return '✕';
    case ModuleStatus.Running: return '▶';
    default: return '○';
  }
}

function timeDiff(a?: number, b?: number): { label: string; cls: string } {
  if (a === undefined || b === undefined) return { label: '', cls: '' };
  const diff = a - b;
  if (diff === 0) return { label: '±0', cls: 'text-gray-400' };
  const sign = diff > 0 ? '+' : '';
  const abs = Math.abs(diff);
  const label = sign + (abs < 1000 ? `${diff}ms` : `${(diff / 1000).toFixed(1)}s`);
  return { label, cls: diff > 0 ? 'text-red-400' : 'text-green-400' };
}

// Build a unified module list from two sessions (by name)
function buildCompareRows(a: RunHistorySession, b: RunHistorySession) {
  const names = Array.from(new Set([...a.results.map(r => r.name), ...b.results.map(r => r.name)]));
  return names.map(name => ({
    name,
    a: a.results.find(r => r.name === name),
    b: b.results.find(r => r.name === name),
  }));
}

export const RunHistoryModal: React.FC<RunHistoryModalProps> = ({ sessions, onClose }) => {
  const [expandedId, setExpandedId] = useState<string | null>(sessions[0]?.id ?? null);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [compareMode, setCompareMode] = useState(false);

  const toggleCompare = (id: string) => {
    setCompareIds(prev => {
      if (prev.includes(id)) return prev.filter(x => x !== id);
      if (prev.length >= 2) return [...prev.slice(1), id]; // 최대 2개
      return [...prev, id];
    });
  };

  const enterCompare = () => {
    if (compareIds.length === 2) setCompareMode(true);
  };

  const exitCompare = () => {
    setCompareMode(false);
    setCompareIds([]);
  };

  // 비교 모드 렌더링
  if (compareMode && compareIds.length === 2) {
    const sessA = sessions.find(s => s.id === compareIds[0])!;
    const sessB = sessions.find(s => s.id === compareIds[1])!;
    const idxA = sessions.findIndex(s => s.id === sessA.id);
    const idxB = sessions.findIndex(s => s.id === sessB.id);
    const rows = buildCompareRows(sessA, sessB);

    const totalA = sessA.results.reduce((acc, r) => acc + (r.executionTime ?? 0), 0);
    const totalB = sessB.results.reduce((acc, r) => acc + (r.executionTime ?? 0), 0);
    const totalDiff = timeDiff(totalA, totalB);

    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
        <div
          className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-xl shadow-2xl w-[600px] max-h-[85vh] flex flex-col"
          onClick={e => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
            <button
              onClick={exitCompare}
              className="text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors text-sm"
              title="목록으로"
            >
              ← 목록
            </button>
            <h2 className="text-sm font-bold text-gray-900 dark:text-white flex-1">
              세션 비교: #{sessions.length - idxA} vs #{sessions.length - idxB}
            </h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors text-lg leading-none"
            >
              ×
            </button>
          </div>

          {/* 비교 테이블 헤더 */}
          <div className="flex-shrink-0 grid grid-cols-[1fr_120px_120px_70px] gap-0 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 px-3 py-2">
            <span className="text-[10px] font-semibold text-gray-500 dark:text-gray-400">모듈</span>
            <div className="text-center">
              <div className="text-[10px] font-semibold text-blue-500">#{sessions.length - idxA}</div>
              <div className="text-[9px] text-gray-400">{fmtDate(sessA.timestamp)}</div>
              <div className="text-[9px] text-gray-500">{fmtTime(totalA)}</div>
            </div>
            <div className="text-center">
              <div className="text-[10px] font-semibold text-purple-500">#{sessions.length - idxB}</div>
              <div className="text-[9px] text-gray-400">{fmtDate(sessB.timestamp)}</div>
              <div className="text-[9px] text-gray-500">{fmtTime(totalB)}</div>
            </div>
            <div className="text-center">
              <div className="text-[10px] font-semibold text-gray-500">차이</div>
              <div className={`text-[10px] font-bold ${totalDiff.cls}`}>{totalDiff.label}</div>
            </div>
          </div>

          {/* 비교 행 */}
          <div className="overflow-y-auto flex-1">
            {rows.map((row, i) => {
              const diff = timeDiff(row.a?.executionTime, row.b?.executionTime);
              const statusChanged = row.a?.status !== row.b?.status;
              return (
                <div
                  key={i}
                  className={`grid grid-cols-[1fr_120px_120px_70px] gap-0 px-3 py-2 border-b border-gray-100 dark:border-gray-700 text-xs ${
                    statusChanged ? 'bg-yellow-50 dark:bg-yellow-900/10' : ''
                  }`}
                >
                  <span className="text-gray-700 dark:text-gray-300 truncate" title={row.name}>
                    {row.name}
                  </span>
                  {/* 세션 A */}
                  <div className="text-center">
                    {row.a ? (
                      <span className={`${statusColor(row.a.status)} font-semibold`}>
                        {statusIcon(row.a.status)} {fmtTime(row.a.executionTime)}
                      </span>
                    ) : (
                      <span className="text-gray-400">—</span>
                    )}
                  </div>
                  {/* 세션 B */}
                  <div className="text-center">
                    {row.b ? (
                      <span className={`${statusColor(row.b.status)} font-semibold`}>
                        {statusIcon(row.b.status)} {fmtTime(row.b.executionTime)}
                      </span>
                    ) : (
                      <span className="text-gray-400">—</span>
                    )}
                  </div>
                  {/* 시간 차이 */}
                  <div className="text-center">
                    {diff.label ? (
                      <span className={`${diff.cls} text-[10px] font-semibold`}>{diff.label}</span>
                    ) : (
                      <span className="text-gray-400 text-[10px]">—</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="px-4 py-2 border-t border-gray-200 dark:border-gray-700 flex-shrink-0">
            <p className="text-[10px] text-gray-400">차이: 양수(+)는 첫 번째 세션이 더 느림, 음수(-)는 더 빠름. 노란 행은 상태가 다른 모듈.</p>
          </div>
        </div>
      </div>
    );
  }

  // 목록 모드 렌더링
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-xl shadow-2xl w-[480px] max-h-[80vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
          <h2 className="text-sm font-bold text-gray-900 dark:text-white">실행 히스토리</h2>
          <div className="flex items-center gap-2">
            {compareIds.length > 0 && (
              <span className="text-[10px] text-gray-500">
                {compareIds.length === 1 ? '비교할 세션을 1개 더 선택하세요' : ''}
              </span>
            )}
            {compareIds.length === 2 && (
              <button
                onClick={enterCompare}
                className="text-xs font-semibold px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors"
              >
                비교하기 →
              </button>
            )}
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors text-lg leading-none"
            >
              ×
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="overflow-y-auto flex-1 p-3 space-y-2">
          {sessions.length === 0 ? (
            <p className="text-sm text-gray-400 text-center py-8">아직 Run All 실행 기록이 없습니다.</p>
          ) : (
            sessions.map((session, idx) => {
              const successCount = session.results.filter(r => r.status === ModuleStatus.Success).length;
              const errorCount   = session.results.filter(r => r.status === ModuleStatus.Error).length;
              const totalTime    = session.results.reduce((acc, r) => acc + (r.executionTime ?? 0), 0);
              const isExpanded   = expandedId === session.id;
              const isSelected   = compareIds.includes(session.id);

              return (
                <div
                  key={session.id}
                  className={`border rounded-lg overflow-hidden transition-colors ${
                    isSelected
                      ? 'border-blue-400 dark:border-blue-500'
                      : 'border-gray-200 dark:border-gray-700'
                  }`}
                >
                  <div className="flex items-center bg-gray-50 dark:bg-gray-750 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                    {/* 비교 선택 체크박스 */}
                    <label
                      className="flex items-center justify-center w-8 h-full cursor-pointer flex-shrink-0"
                      title="비교에 추가"
                      onClick={e => { e.stopPropagation(); toggleCompare(session.id); }}
                    >
                      <div className={`w-3.5 h-3.5 rounded border-2 flex items-center justify-center transition-colors ${
                        isSelected
                          ? 'bg-blue-500 border-blue-500'
                          : 'border-gray-400 dark:border-gray-500'
                      }`}>
                        {isSelected && <span className="text-white text-[8px] font-bold leading-none">✓</span>}
                      </div>
                    </label>

                    <button
                      className="flex-1 flex items-center gap-3 py-2 pr-3 text-left"
                      onClick={() => setExpandedId(isExpanded ? null : session.id)}
                    >
                      <span className="text-[10px] font-bold text-gray-400 dark:text-gray-500 flex-shrink-0">
                        #{sessions.length - idx}
                      </span>
                      <span className="text-xs font-medium text-gray-700 dark:text-gray-300 flex-shrink-0">
                        {fmtDate(session.timestamp)}
                      </span>
                      <span className="flex-1" />
                      <span className="text-[10px] text-green-500">{successCount}✓</span>
                      {errorCount > 0 && <span className="text-[10px] text-red-400">{errorCount}✕</span>}
                      <span className="text-[10px] text-gray-400">{fmtTime(totalTime)}</span>
                      <span className="text-gray-400 text-xs ml-1">{isExpanded ? '▲' : '▼'}</span>
                    </button>
                  </div>

                  {isExpanded && (
                    <div className="divide-y divide-gray-100 dark:divide-gray-700">
                      {session.results.map((result, i) => (
                        <div key={i} className="flex items-center gap-2 px-3 py-1.5">
                          <span className={`text-[11px] font-bold ${statusColor(result.status)} flex-shrink-0 w-3 text-center`}>
                            {statusIcon(result.status)}
                          </span>
                          <span className="text-xs text-gray-700 dark:text-gray-300 flex-1 truncate" title={result.name}>
                            {result.name}
                          </span>
                          <span className="text-[10px] text-gray-400 flex-shrink-0">
                            {fmtTime(result.executionTime)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>

        <div className="px-4 py-2 border-t border-gray-200 dark:border-gray-700 flex-shrink-0">
          <p className="text-[10px] text-gray-400">세션 2개를 선택하면 비교하기 버튼이 활성화됩니다. 최근 10회 기록.</p>
        </div>
      </div>
    </div>
  );
};
