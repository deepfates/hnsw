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
import { HNSW } from '../src/hnsw';

// Simple example
const hnsw = new HNSW(200, 16, 5, 'cosine');

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
console.log(results);
```

Persistent index using IndexedDB:
```typescript
import { HNSWWithDB } from 'hnsw';

// With persistence
const index = await HNSWWithDB.create(200, 16, 'my-index');

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
const index2 = await HNSWWithDB.create(200, 16, 'my-index-2');
await index2.loadIndex();

// Search for nearest neighbors
const results2 = index2.searchKNN([6, 7, 8, 9, 10], 2);
console.log(results2);

// Delete the index
await index2.deleteIndex();
```

