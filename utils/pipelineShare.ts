import { CanvasModule, Connection } from '../types';

interface SharePayload {
  version: 1;
  projectName: string;
  modules: Array<Omit<CanvasModule, 'outputData' | 'status' | 'executionTime'>>;
  connections: Connection[];
}

/** Encode pipeline state to a base64 URL hash string. */
export function encodePipeline(
  modules: CanvasModule[],
  connections: Connection[],
  projectName: string
): string {
  const stripped = modules.map(({ outputData: _od, status: _s, executionTime: _et, ...rest }) => rest);
  const payload: SharePayload = { version: 1, projectName, modules: stripped as any, connections };
  const json = JSON.stringify(payload);
  return btoa(encodeURIComponent(json));
}

/** Decode a base64 hash string back to pipeline state. Returns null on failure. */
export function decodePipeline(hash: string): { modules: CanvasModule[]; connections: Connection[]; projectName: string } | null {
  try {
    const json = decodeURIComponent(atob(hash));
    const payload: SharePayload = JSON.parse(json);
    if (payload.version !== 1 || !Array.isArray(payload.modules)) return null;
    return {
      modules: payload.modules as CanvasModule[],
      connections: payload.connections,
      projectName: payload.projectName || 'Shared Pipeline',
    };
  } catch {
    return null;
  }
}

/** Copy a shareable URL with the pipeline encoded in the hash to the clipboard. */
export async function copyShareLink(
  modules: CanvasModule[],
  connections: Connection[],
  projectName: string
): Promise<void> {
  const hash = encodePipeline(modules, connections, projectName);
  const url = `${window.location.origin}${window.location.pathname}#share=${hash}`;
  await navigator.clipboard.writeText(url);
}

/** Check the current URL hash for a shared pipeline. Returns payload or null. */
export function detectSharedPipeline(): ReturnType<typeof decodePipeline> {
  const hash = window.location.hash;
  const match = hash.match(/[#&]share=([^&]+)/);
  if (!match) return null;
  return decodePipeline(match[1]);
}
