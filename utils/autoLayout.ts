import { CanvasModule, Connection } from '../types';

const MODULE_W = 256;
const MODULE_H = 130;
const H_GAP = 80;  // horizontal gap between columns
const V_GAP = 40;  // vertical gap within a column

/** Assign (x, y) positions to modules using topological column layout. Returns updated modules. */
export function autoLayoutModules(
  modules: CanvasModule[],
  connections: Connection[]
): CanvasModule[] {
  if (modules.length === 0) return modules;

  const ids = modules.map(m => m.id);

  // Build adjacency: parent → children
  const children = new Map<string, string[]>(ids.map(id => [id, []]));
  const parentCount = new Map<string, number>(ids.map(id => [id, 0]));

  for (const c of connections) {
    if (children.has(c.from.moduleId) && parentCount.has(c.to.moduleId)) {
      const existing = children.get(c.from.moduleId)!;
      if (!existing.includes(c.to.moduleId)) {
        existing.push(c.to.moduleId);
        parentCount.set(c.to.moduleId, (parentCount.get(c.to.moduleId) ?? 0) + 1);
      }
    }
  }

  // Kahn's algorithm — assign column (level) to each module
  const col = new Map<string, number>(ids.map(id => [id, 0]));
  const queue = ids.filter(id => (parentCount.get(id) ?? 0) === 0);
  const remaining = new Map(parentCount);

  const processed = new Set<string>();
  while (queue.length > 0) {
    const id = queue.shift()!;
    processed.add(id);
    for (const child of children.get(id) ?? []) {
      col.set(child, Math.max(col.get(child) ?? 0, (col.get(id) ?? 0) + 1));
      remaining.set(child, (remaining.get(child) ?? 1) - 1);
      if ((remaining.get(child) ?? 0) <= 0) queue.push(child);
    }
  }

  // Modules not reached (isolated or in cycles) get col 0
  for (const id of ids) {
    if (!processed.has(id)) col.set(id, 0);
  }

  // Group by column and sort within each column by original y (preserve relative order)
  const colGroups = new Map<number, CanvasModule[]>();
  for (const m of modules) {
    const c = col.get(m.id) ?? 0;
    if (!colGroups.has(c)) colGroups.set(c, []);
    colGroups.get(c)!.push(m);
  }
  // Sort each column by original y position to maintain relative order
  for (const [, group] of colGroups) {
    group.sort((a, b) => a.position.y - b.position.y);
  }

  // Assign positions: start at (80, 80) from top-left
  const START_X = 80;
  const START_Y = 80;

  const posMap = new Map<string, { x: number; y: number }>();
  const sortedCols = Array.from(colGroups.keys()).sort((a, b) => a - b);

  for (const c of sortedCols) {
    const group = colGroups.get(c)!;
    const x = START_X + c * (MODULE_W + H_GAP);

    // Center the column vertically around the tallest column
    const totalH = group.length * MODULE_H + (group.length - 1) * V_GAP;
    let y = START_Y;

    for (const m of group) {
      posMap.set(m.id, { x, y });
      y += MODULE_H + V_GAP;
    }
  }

  return modules.map(m => ({
    ...m,
    position: posMap.get(m.id) ?? m.position,
  }));
}
