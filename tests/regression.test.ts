import * as fs from 'fs';
import * as path from 'path';
import { HNSW } from '../src';

const expected: Record<string, { id: number; score: number }[][]> = JSON.parse(
  fs.readFileSync(path.join(__dirname, 'fixtures', 'searchResults.json'), 'utf8'),
);

// Deterministic PRNG so vectors and level selection are reproducible.
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const dims = 16;
const rngVec = mulberry32(42);
const data = Array.from({ length: 300 }, (_, i) => ({
  id: i,
  vector: Array.from({ length: dims }, () => rngVec() * 2 - 1),
}));
const queries = Array.from({ length: 10 }, () => Array.from({ length: dims }, () => rngVec() * 2 - 1));

// The fixture was generated with the pre-norm-caching implementation
// (three dot products per cosine call). Cached norms must produce
// identical graphs and identical search results.
describe('search results are unchanged by cached-norm optimization', () => {
  it.each(['cosine', 'euclidean'] as const)('%s metric matches the pre-change fixture exactly', async (metric) => {
    const rngLevel = mulberry32(7);
    const mockRandom = jest.spyOn(Math, 'random').mockImplementation(rngLevel);
    const hnsw = new HNSW(16, 64, dims, metric, 64);
    await hnsw.buildIndex(data);
    mockRandom.mockRestore();

    const results = queries.map((q) => hnsw.searchKNN(q, 10));
    expect(results).toEqual(expected[metric]);
  });
});
