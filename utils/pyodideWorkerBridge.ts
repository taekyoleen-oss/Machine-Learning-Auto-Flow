/**
 * Bridge to the Pyodide Web Worker.
 * Provides the same runPythonWithOutput interface as pyodideRunner.ts
 * but executes in a background worker thread — UI stays responsive.
 */

type StatusCb = (status: string, progress: number) => void;

let worker: Worker | null = null;
let isReady = false;
let initStarted = false;
let statusCb: StatusCb | null = null;
let callId = 0;

const pending = new Map<number, {
  resolve: (r: { stdout: string; error: string | null }) => void;
  reject: (e: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}>();

const readyWaiters: Array<{ resolve: () => void; reject: (e: Error) => void }> = [];

function handleMessage(event: MessageEvent) {
  const { type, id, status, progress, stdout, error } = event.data;

  if (type === 'ready') {
    isReady = true;
    readyWaiters.splice(0).forEach((w) => w.resolve());
    setTimeout(() => statusCb?.('', 0), 1500);
  } else if (type === 'progress') {
    statusCb?.(status, progress);
  } else if (type === 'result') {
    const cb = pending.get(id);
    if (cb) {
      clearTimeout(cb.timer);
      cb.resolve({ stdout, error });
      pending.delete(id);
    }
  } else if (type === 'init_error') {
    initStarted = false;
    const err = new Error(error || 'Worker 초기화 실패');
    readyWaiters.splice(0).forEach((w) => w.reject(err));
  }
}

function createAndInitWorker() {
  worker = new Worker('/pyodideWorker.js');
  worker.onmessage = handleMessage;
  worker.onerror = (e) => {
    const err = new Error(e.message || 'Worker 오류');
    readyWaiters.splice(0).forEach((w) => w.reject(err));
    for (const [, cb] of pending) { clearTimeout(cb.timer); cb.reject(err); }
    pending.clear();
  };
  worker.postMessage({ type: 'init' });
}

function ensureWorkerReady(): Promise<void> {
  if (isReady) return Promise.resolve();
  return new Promise((resolve, reject) => {
    readyWaiters.push({ resolve, reject });
    if (!initStarted) {
      initStarted = true;
      createAndInitWorker();
    }
  });
}

export function setWorkerStatusCallback(cb: StatusCb | null): void {
  statusCb = cb;
}

export function getWorkerLoadingStatus(): { isLoading: boolean; isPyodideReady: boolean } {
  return { isLoading: initStarted && !isReady, isPyodideReady: isReady };
}

export async function runPythonWithOutputWorker(
  code: string,
  timeoutMs = 90000
): Promise<{ stdout: string; error: string | null }> {
  await ensureWorkerReady();
  const id = ++callId;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(new Error('실행 타임아웃 (90초 초과)'));
    }, timeoutMs);
    pending.set(id, { resolve, reject, timer });
    worker!.postMessage({ type: 'run', id, code });
  });
}

/**
 * Terminate the worker to cancel any running execution.
 * The next call to runPythonWithOutputWorker will restart it (re-init ~30s).
 */
export function cancelWorkerRun(): void {
  if (worker) {
    worker.terminate();
    worker = null;
  }
  isReady = false;
  initStarted = false;
  for (const [, cb] of pending) {
    clearTimeout(cb.timer);
    cb.reject(new Error('실행이 취소되었습니다.'));
  }
  pending.clear();
  readyWaiters.splice(0).forEach((w) => w.reject(new Error('취소됨')));
}
