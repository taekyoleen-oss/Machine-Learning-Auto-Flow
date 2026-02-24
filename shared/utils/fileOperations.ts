/**
 * File operations using File System Access API
 */

export interface SaveOptions {
  extension: string;
  description: string;
  onSuccess?: (fileName: string) => void;
  onError?: (error: Error) => void;
}

/** Save pipeline — 항상 로컬 다운로드로 저장 */
export async function savePipeline(
  data: any,
  options: SaveOptions
): Promise<void> {
  try {
    const fileName = `${data.projectName && data.projectName.trim() ? data.projectName.trim() : 'pipeline'}${options.extension}`;
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    if (options.onSuccess) {
      options.onSuccess(fileName);
    }
  } catch (error: any) {
    if (options.onError) {
      options.onError(error instanceof Error ? error : new Error(String(error)));
    } else {
      throw error;
    }
  }
}

export async function loadPipeline(): Promise<any> {
  try {
    if (!('showOpenFilePicker' in window)) {
      throw new Error(
        'File System Access API is not supported in this browser.'
      );
    }

    const [fileHandle] = await (window as any).showOpenFilePicker({
      types: [
        {
          description: 'ML Pipeline Files',
          accept: {
            'application/json': ['.ins', '.json'],
          },
        },
      ],
    });

    const file = await fileHandle.getFile();
    const content = await file.text();
    return JSON.parse(content);
  } catch (error: any) {
    if (error.name === 'AbortError') {
      // User cancelled the open dialog
      return null;
    }
    throw error;
  }
}

