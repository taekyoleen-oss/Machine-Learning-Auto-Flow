/**
 * Pyodide Web Worker — runs Python in a background thread so the UI stays responsive.
 * Loaded as a classic worker (importScripts).
 */
importScripts('https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js');

let pyodide = null;
let isReady = false;

const PACKAGES = ['numpy', 'scipy', 'pandas', 'scikit-learn'];
const PKG_PROGRESS = { numpy: 55, scipy: 65, pandas: 75, 'scikit-learn': 88 };

function report(status, progress) {
  self.postMessage({ type: 'progress', status, progress });
}

self.onmessage = async (event) => {
  const { type, id, code } = event.data;

  if (type === 'init') {
    try {
      report('Python 환경(Pyodide) 초기화 중...', 10);
      pyodide = await loadPyodide({ indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/' });

      let lastPkgIdx = -1;
      const pkgCallback = (msg) => {
        const match = msg.match(/(?:Loading|Installing|Loaded)\s+(\S+)/i);
        if (!match) return;
        const name = match[1].toLowerCase().replace(/[^a-z0-9-]/g, '');
        PACKAGES.forEach((pkg, idx) => {
          if (idx > lastPkgIdx && (name.startsWith(pkg.replace('-', '')) || name.includes(pkg))) {
            lastPkgIdx = idx;
            report(`패키지 설치 중... (${idx + 1}/${PACKAGES.length}) ${pkg}`, PKG_PROGRESS[pkg] || 70);
          }
        });
      };

      await pyodide.loadPackage(PACKAGES, pkgCallback);
      report('Python 환경 준비 완료!', 100);
      isReady = true;
      self.postMessage({ type: 'ready' });
    } catch (err) {
      self.postMessage({ type: 'init_error', error: err.message });
    }

  } else if (type === 'run') {
    if (!isReady) {
      self.postMessage({ type: 'result', id, stdout: '', error: 'Pyodide가 초기화되지 않았습니다.' });
      return;
    }

    // Clear previous globals
    try {
      await pyodide.runPythonAsync(`
import gc as _gc
_user_vars = [k for k in list(globals().keys()) if not k.startswith('_')]
for _k in _user_vars:
    try: del globals()[_k]
    except: pass
_gc.collect()
`);
    } catch (_) { /* ignore cleanup errors */ }

    // Execute with stdout capture
    try {
      const indented = code.split('\n').map(l => '    ' + l).join('\n');
      const wrapped = `
import io as _io, sys as _sys, traceback as _tb
_buf = _io.StringIO()
_old = _sys.stdout
_sys.stdout = _buf
_err = None
try:
${indented}
except Exception as _e:
    _err = _tb.format_exc()
finally:
    _sys.stdout = _old
(_buf.getvalue(), _err)
`;
      const result = await pyodide.runPythonAsync(wrapped);
      const js = result && typeof result.toJs === 'function' ? result.toJs() : result;
      self.postMessage({ type: 'result', id, stdout: js?.[0] || '', error: js?.[1] || null });
    } catch (err) {
      self.postMessage({ type: 'result', id, stdout: '', error: err.message });
    }
  }
};
