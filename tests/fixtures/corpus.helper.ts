// Shared deterministic corpora for the cached-norm regression suite.
//
// This module is imported both by tests/regression.test.ts and by the fixture
// generator (tests/fixtures/generate.test.ts, run with GENERATE_FIXTURES=1 on
// the pre-change commit), so the data and level-selection randomness are
// guaranteed to be identical on both sides.

import { HNSW } from '../../src';

export type Vector = number[] | Float32Array;
export type Corpus = { name: string; dims: number; data: { id: number; vector: Vector }[]; queries: Vector[] };

// Deterministic PRNG so vectors and level selection are reproducible.
export function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Three dimensionalities; each corpus includes a zero vector, Float32Array
// inputs, an exact duplicate and a near-tie pair, plus zero / Float32Array /
// exact-match queries.
export function makeCorpora(): Corpus[] {
  const corpora: Corpus[] = [];
  for (const dims of [4, 16, 64]) {
    const rng = mulberry32(1000 + dims);
    const data: { id: number; vector: Vector }[] = [];
    for (let i = 0; i < 160; i++) {
      let vector: Vector = Array.from({ length: dims }, () => rng() * 2 - 1);
      if (i === 3) {
        vector = new Array(dims).fill(0); // zero vector
      } else if (i === 40) {
        vector = Array.from(data[39].vector, Number); // exact duplicate of id 39 (tie)
      } else if (i === 41) {
        const nearTie = Array.from(data[39].vector, Number);
        nearTie[0] += 1e-12; // near-tie with ids 39/40
        vector = nearTie;
      } else if (i % 7 === 0) {
        vector = Float32Array.from(vector as number[]); // Float32Array input
      }
      data.push({ id: i, vector });
    }
    const queries: Vector[] = [];
    for (let q = 0; q < 8; q++) {
      queries.push(Array.from({ length: dims }, () => rng() * 2 - 1));
    }
    queries.push(new Array(dims).fill(0)); // zero query
    queries.push(Float32Array.from(queries[0] as number[])); // Float32Array query
    queries.push(Array.from(data[39].vector, Number)); // exact-match query -> near-tie ordering
    corpora.push({ name: `d${dims}`, dims, data, queries });
  }
  return corpora;
}

// Level selection goes through Math.random; seed it so both implementations
// build the exact same graph topology.
export async function buildWithSeededLevels(hnsw: HNSW, data: { id: number; vector: Vector }[], seed = 7) {
  const original = Math.random;
  Math.random = mulberry32(seed);
  try {
    await hnsw.buildIndex(data);
  } finally {
    Math.random = original;
  }
}

// Custom similarity functions used by the fixture generator and the test.
// Deliberately asymmetric, so any change in operand orientation shows up.
export const asymmetricSimilarity = (a: Vector, b: Vector): number => a[0] - b[0];
export const l1Similarity = (a: Vector, b: Vector): number => {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  return -sum;
};

export const K = 10;
export const FIXTURE_FILE = 'regression-v2.json';
