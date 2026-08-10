// Reproducible benchmark for the cosine norm-caching optimization.
//
// Usage (after `npm run build`):
//
//   node dist/bench/norms.js [count] [dims] [queries]
//
// Defaults: 10000 vectors x 384 dims, 1000 queries, M=16, efConstruction=200,
// efSearch=64, k=8. Vectors and level selection are seeded, so the same graph
// is built on every run and across implementations. Run it on the baseline
// commit and on this branch (same machine, same Node) to get before/after
// numbers. Note this harness does not exist at the baseline commit (8542a17):
// to benchmark the baseline, copy this file into the baseline checkout
// (`git worktree add <dir> 8542a17 && cp src/bench/norms.ts <dir>/src/bench/`),
// build there, and run it.
//
// Verification digests (sha256) cover the FULL serialized graph and EVERY
// ordered search result — not an aggregate — so a baseline/branch match means
// identical graphs and identical result lists, bit for bit.

import { createHash } from 'crypto';
import { HNSW } from '../main';

function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

async function run() {
  const count = Number(process.argv[2] ?? 10000);
  const dims = Number(process.argv[3] ?? 384);
  const numQueries = Number(process.argv[4] ?? 1000);
  const k = 8;

  const rng = mulberry32(42);
  const data = Array.from({ length: count }, (_, i) => ({
    id: i,
    vector: Float32Array.from({ length: dims }, () => rng() * 2 - 1),
  }));
  const queries = Array.from({ length: numQueries }, () => Float32Array.from({ length: dims }, () => rng() * 2 - 1));

  const hnsw = new HNSW(16, 200, dims, 'cosine', 64);

  // Seed level selection so every run builds the same graph.
  const originalRandom = Math.random;
  Math.random = mulberry32(7);
  const buildStart = performance.now();
  await hnsw.buildIndex(data);
  const buildMs = performance.now() - buildStart;
  Math.random = originalRandom;

  // Warm up, then measure.
  for (let i = 0; i < 100; i++) {
    hnsw.searchKNN(queries[i % numQueries], k);
  }
  const allResults: { id: number; score: number }[][] = [];
  const queryStart = performance.now();
  for (const query of queries) {
    allResults.push(hnsw.searchKNN(query, k));
  }
  const queryMs = performance.now() - queryStart;

  const graphDigest = createHash('sha256').update(JSON.stringify(hnsw.toJSON())).digest('hex');
  const resultsDigest = createHash('sha256').update(JSON.stringify(allResults)).digest('hex');

  process.stdout.write(
    [
      `hnsw norm-caching benchmark: ${count} vectors x ${dims} dims, ` +
        `M=16 efConstruction=200 efSearch=64 k=${k} queries=${numQueries}`,
      `build: ${(buildMs / 1000).toFixed(2)} s`,
      `query: ${(queryMs / 1000).toFixed(2)} s (${((queryMs / numQueries) * 1000).toFixed(0)} us/query)`,
      `total: ${((buildMs + queryMs) / 1000).toFixed(2)} s`,
      `graph digest (sha256 of full toJSON, compare across implementations): ${graphDigest}`,
      `results digest (sha256 of all ordered search results): ${resultsDigest}`,
      '',
    ].join('\n'),
  );
}

run().catch((error) => {
  process.stderr.write(`${error?.stack ?? error}\n`);
  process.exit(1);
});
