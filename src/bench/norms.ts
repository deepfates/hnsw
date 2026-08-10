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
// numbers.

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
  let checksum = 0;
  const queryStart = performance.now();
  for (const query of queries) {
    const results = hnsw.searchKNN(query, k);
    checksum += results[0].id + results[0].score;
  }
  const queryMs = performance.now() - queryStart;

  process.stdout.write(
    [
      `hnsw norm-caching benchmark: ${count} vectors x ${dims} dims, ` +
        `M=16 efConstruction=200 efSearch=64 k=${k} queries=${numQueries}`,
      `build: ${(buildMs / 1000).toFixed(2)} s`,
      `query: ${(queryMs / 1000).toFixed(2)} s (${((queryMs / numQueries) * 1000).toFixed(0)} us/query)`,
      `total: ${((buildMs + queryMs) / 1000).toFixed(2)} s`,
      `checksum (compare across implementations): ${checksum}`,
      '',
    ].join('\n'),
  );
}

run().catch((error) => {
  process.stderr.write(`${error?.stack ?? error}\n`);
  process.exit(1);
});
