/**
 * In-memory cache for LoadData file content shared across pipeline tabs.
 * Keyed by filename. Cleared only when the page refreshes.
 *
 * Usage:
 *   cacheFileContent('data.csv', base64String)   // when user loads a file
 *   getCachedFileContent('data.csv')              // when restoring a tab
 */

const cache = new Map<string, string>(); // filename → base64 content

export function cacheFileContent(filename: string, content: string): void {
  if (filename && content) cache.set(filename, content);
}

export function getCachedFileContent(filename: string): string | null {
  return cache.get(filename) ?? null;
}

export function clearFileCache(): void {
  cache.clear();
}

export function listCachedFiles(): string[] {
  return Array.from(cache.keys());
}
