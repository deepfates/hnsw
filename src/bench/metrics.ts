export type Neighbor = { id: number; score: number };

export type RecallResult = {
  recall: number;
  hits: number;
  total: number;
};

export function bruteForceKNN(
  query: Float32Array | number[],
  vectors: Array<{ id: number; vector: Float32Array | number[] }>,
  similarity: (a: Float32Array | number[], b: Float32Array | number[]) => number,
  k: number,
): Neighbor[] {
  if (k <= 0) return [];

  const scored: Neighbor[] = vectors.map((entry) => ({
    id: entry.id,
    score: similarity(query, entry.vector),
  }));

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, Math.min(k, scored.length));
}

export function recallAtK(approx: Neighbor[], exact: Neighbor[], k: number): RecallResult {
  if (k <= 0) {
    return { recall: 0, hits: 0, total: 0 };
  }

  const topApprox = new Set(approx.slice(0, k).map((item) => item.id));
  const topExact = exact.slice(0, k);

  let hits = 0;
  for (const item of topExact) {
    if (topApprox.has(item.id)) {
      hits++;
    }
  }

  const total = topExact.length;
  const recall = total === 0 ? 0 : hits / total;
  return { recall, hits, total };
}

export function average(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const clamped = Math.max(0, Math.min(100, p));
  const index = Math.floor((clamped / 100) * (sorted.length - 1));
  return sorted[index];
}

export function summarizeLatencies(latenciesMs: number[]) {
  return {
    count: latenciesMs.length,
    avg: average(latenciesMs),
    p50: percentile(latenciesMs, 50),
    p90: percentile(latenciesMs, 90),
    p95: percentile(latenciesMs, 95),
    p99: percentile(latenciesMs, 99),
  };
}

export function meanRecall(results: RecallResult[]): number {
  if (results.length === 0) return 0;
  const totalHits = results.reduce((sum, result) => sum + result.hits, 0);
  const total = results.reduce((sum, result) => sum + result.total, 0);
  return total === 0 ? 0 : totalHits / total;
}
