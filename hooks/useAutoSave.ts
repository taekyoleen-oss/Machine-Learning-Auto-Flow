import { useEffect, useRef } from 'react';
import { CanvasModule, Connection } from '../types';

const AUTO_SAVE_KEY = 'mlAutoFlow_pipeline_autosave';
const TTL_SECONDS = 3600;

interface UseAutoSaveOptions {
  modules: CanvasModule[];
  connections: Connection[];
  projectName: string;
  onRestore: (modules: CanvasModule[], connections: Connection[], projectName?: string) => void;
  onRestoreLog: (message: string) => void;
}

export function useAutoSave({
  modules,
  connections,
  projectName,
  onRestore,
  onRestoreLog,
}: UseAutoSaveOptions) {
  // Save whenever pipeline state changes
  useEffect(() => {
    if (modules.length === 0 && connections.length === 0) return;
    try {
      const saveable = {
        modules: modules.map((m) => ({
          ...m,
          outputData: undefined,
          executionTime: undefined,
          parameters: {
            ...m.parameters,
            fileContent: m.parameters?.fileContent ? '__FILE_LOADED__' : undefined,
          },
        })),
        connections,
        projectName,
        savedAt: Date.now(),
      };
      sessionStorage.setItem(AUTO_SAVE_KEY, JSON.stringify(saveable));
    } catch (_) {
      // sessionStorage 용량 초과 등 실패 시 무시
    }
  }, [modules, connections, projectName]);

  // Restore once on mount
  const restoredRef = useRef(false);
  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;
    try {
      const saved = sessionStorage.getItem(AUTO_SAVE_KEY);
      if (!saved) return;
      const parsed = JSON.parse(saved);
      const ageSec = (Date.now() - (parsed.savedAt || 0)) / 1000;
      if (ageSec > TTL_SECONDS) {
        sessionStorage.removeItem(AUTO_SAVE_KEY);
        return;
      }
      if (parsed.modules?.length > 0) {
        onRestore(parsed.modules, parsed.connections || [], parsed.projectName);
        onRestoreLog(
          `자동 저장된 파이프라인을 복원했습니다 (${parsed.modules.length}개 모듈). LoadData 모듈은 파일을 다시 선택해주세요.`
        );
      }
    } catch (_) {
      // 복원 실패 시 무시
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}
