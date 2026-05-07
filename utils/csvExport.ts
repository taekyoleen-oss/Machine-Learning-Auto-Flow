import { CanvasModule } from '../types';

interface CsvData {
  rows: Record<string, any>[];
  columns: Array<{ name: string; type?: string }>;
}

/** Extract downloadable tabular data from a module's outputData. Returns null if none available. */
export function getModuleCsvData(module: CanvasModule): CsvData | null {
  const od = module.outputData as any;
  if (!od) return null;

  // Direct DataPreview
  if (od.type === 'DataPreview' && Array.isArray(od.rows) && od.rows.length > 0) {
    return { rows: od.rows, columns: od.columns || [] };
  }

  // SplitDataOutput — return train set (largest)
  if (od.type === 'SplitDataOutput' && od.train?.rows?.length > 0) {
    return { rows: od.train.rows, columns: od.train.columns || [] };
  }

  // Wrapper outputs (MissingHandlerOutput, EncoderOutput, NormalizerOutput, etc.)
  if (od.data?.type === 'DataPreview' && Array.isArray(od.data.rows) && od.data.rows.length > 0) {
    return { rows: od.data.rows, columns: od.data.columns || [] };
  }

  // KMeansOutput / ClusteringDataOutput
  const clusterTarget = od.clusterAssignments || od.clusteredData;
  if (clusterTarget?.rows?.length > 0) {
    return { rows: clusterTarget.rows, columns: clusterTarget.columns || [] };
  }

  // PCAOutput
  if (od.transformedData?.rows?.length > 0) {
    return { rows: od.transformedData.rows, columns: od.transformedData.columns || [] };
  }

  return null;
}

/** Convert rows+columns to a CSV string. */
function toCsvString(data: CsvData): string {
  const colNames = data.columns.map(c => c.name);
  const escape = (v: any): string => {
    if (v === null || v === undefined) return '';
    const s = String(v);
    return s.includes(',') || s.includes('"') || s.includes('\n')
      ? `"${s.replace(/"/g, '""')}"`
      : s;
  };
  const header = colNames.map(escape).join(',');
  const body = data.rows.map(row => colNames.map(col => escape(row[col])).join(',')).join('\n');
  return `${header}\n${body}`;
}

/** Trigger a browser download of the module's output data as CSV. */
export function downloadModuleDataAsCsv(module: CanvasModule): void {
  const data = getModuleCsvData(module);
  if (!data) return;

  const csv = toCsvString(data);
  const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8;' }); // BOM for Excel
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${module.name.replace(/[^a-zA-Z0-9가-힣_-]/g, '_')}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
