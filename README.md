# HNSW

This is a small Typescript package that implements the Hierarchical Navigable Small Worlds algorithm for approximate nearest neighbor search.

I wrote this package because I wanted to do efficient vector search directly in the client browser. All the other implementations I found for TS were either bindings for libraries written in other languages, or dealt with WASM compilation complexity.

This is not the fastest, most fully featured, or most memory efficient implementation of HNSW. It is, however, a simple and easy to use implementation that is fast enough for many use cases.

Included is a simple persistent storage layer that uses IndexedDB to store the graph.

## Installation
    
```bash
npm install hnsw
```

## Usage

Ephemeral index in-memory:
```typescript
import { HNSW } from 'hnsw';

// Simple example
const hnsw = new HNSW(16, 200, 5, 'cosine', 50);

// Make some data
const data = [
{id: 1, vector: [1, 2, 3, 4, 5]},
{id: 2, vector: [2, 3, 4, 5, 6]},
{id: 3, vector: [3, 4, 5, 6, 7]},
{id: 4, vector: [4, 5, 6, 7, 8]},
{id: 5, vector: [5, 6, 7, 8, 9]}
]

// Build the index
await hnsw.buildIndex(data);

// Search for nearest neighbors
const results = hnsw.searchKNN([6, 7, 8, 9, 10], 2);
const resultsWithEf = hnsw.searchKNN([6, 7, 8, 9, 10], 2, { efSearch: 100 });
console.log(results);
```

Persistent index using IndexedDB:
```typescript
import { HNSWWithDB } from 'hnsw';

// With persistence
const index = await HNSWWithDB.create(16, 200, 'my-index', 50);

// Make some data
const data = [
{id: 1, vector: [1, 2, 3, 4, 5]},
{id: 2, vector: [2, 3, 4, 5, 6]},
{id: 3, vector: [3, 4, 5, 6, 7]},
{id: 4, vector: [4, 5, 6, 7, 8]},
{id: 5, vector: [5, 6, 7, 8, 9]}
]

// Build the index
await index.buildIndex(data);
await index.saveIndex();

// Load the index
const index2 = await HNSWWithDB.create(16, 200, 'my-index-2', 50);
await index2.loadIndex();

// Search for nearest neighbors
const results2 = index2.searchKNN([6, 7, 8, 9, 10], 2);
console.log(results2);

// Delete the index
await index2.deleteIndex();
```

Notes:
- The `metric` determines how scores are computed: `cosine` uses cosine similarity and `euclidean` uses an inverse-distance similarity (higher is better in both cases).
- `efSearch` controls query-time exploration and should be at least `k` for best recall.

## Benchmarks

A lightweight benchmark harness is available to validate recall/latency tradeoffs and the impact of parameters like `efSearch`, `M`, and `efConstruction`.

Build the project first:
```/dev/null/build.sh#L1-1
npm run build
```

Download SIFT small (10k) dataset:
```/dev/null/download-siftsmall.sh#L1-2
node dist/bench/download.js --extract
```

Synthetic dataset (fast sanity check):
```/dev/null/synthetic.sh#L1-2
node dist/bench/run.js --mode synthetic --count 10000 --dim 64 --metric cosine
```

FVECS dataset (SIFT/GloVe-style):
```/dev/null/fvecs.sh#L1-2
node dist/bench/run.js --mode fvecs --base bench/datasets/siftsmall_base.fvecs --query bench/datasets/siftsmall_query.fvecs --metric euclidean --limit 10000 --query-limit 100
```

Compare results (baseline vs changes):
```/dev/null/report.sh#L1-2
node dist/bench/report.js --base bench/outputs/baseline.json --candidate bench/outputs/changes.json --format csv --output bench/outputs/compare.csv
```

One-shot compare (runs baseline + candidate + report in one command):
```/dev/null/compare.sh#L1-2
node dist/bench/compare.js --base-ref HEAD~1 --candidate-ref HEAD --mode fvecs --base bench/datasets/siftsmall_base.fvecs --query bench/datasets/siftsmall_query.fvecs --metric euclidean --limit 10000 --query-limit 100
```

For more details, see `bench/README.md`.
