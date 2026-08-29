import { deserializeHNSW, HNSW, serializeHNSW } from '../src';

describe('binary persistence', () => {
  function seededRandom(seed: number): () => number {
    return () => {
      seed |= 0;
      seed = (seed + 0x6d2b79f5) | 0;
      let value = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
      return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
  }

  async function example(): Promise<HNSW> {
    const index = new HNSW(8, 50, 3, 'cosine', 20);
    await index.buildIndex([
      { id: 10, vector: [1, 0, 0] },
      { id: 20, vector: [0.9, 0.1, 0] },
      { id: 30, vector: [0, 1, 0] },
    ]);
    return index;
  }

  it.each(['float32', 'int16'] as const)('round-trips a %s index', async (vectorEncoding) => {
    const index = await example();
    const restored = deserializeHNSW(serializeHNSW(index, { vectorEncoding }));
    expect(restored.searchKNN([1, 0, 0], 3).map((result) => result.id)).toEqual([10, 20, 30]);
    expect(restored.metric).toBe('cosine');
    expect(restored.efSearch).toBe(20);
  });

  it('rejects compact encoding when it would change euclidean meaning', async () => {
    const index = new HNSW(8, 50, 2, 'euclidean');
    await index.buildIndex([{ id: 1, vector: [1, 2] }]);
    expect(() => serializeHNSW(index, { vectorEncoding: 'int16' })).toThrow('only valid for cosine');
  });

  it('rejects corrupt, truncated, and trailing input', async () => {
    const binary = serializeHNSW(await example());
    expect(() => deserializeHNSW(binary.subarray(0, binary.length - 1))).toThrow();
    expect(() => deserializeHNSW(new Uint8Array(binary.length))).toThrow('magic');
    const trailing = new Uint8Array(binary.length + 1);
    trailing.set(binary);
    expect(() => deserializeHNSW(trailing)).toThrow('trailing');
  });

  it('does not allocate from an impossible untrusted node count', async () => {
    const binary = serializeHNSW(await example());
    const corrupt = binary.slice();
    new DataView(corrupt.buffer).setUint32(32, 0xffffffff, true);
    expect(() => deserializeHNSW(corrupt)).toThrow('node count');
  });

  it('can build reproducible static indexes from a seeded random source', async () => {
    const data = Array.from({ length: 50 }, (_, id) => ({
      id,
      vector: [Math.sin(id), Math.cos(id), id / 50],
    }));
    const first = new HNSW(8, 50, 3, 'cosine', 20, seededRandom(42));
    const second = new HNSW(8, 50, 3, 'cosine', 20, seededRandom(42));
    await first.buildIndex(data);
    await second.buildIndex(data);
    expect(serializeHNSW(first)).toEqual(serializeHNSW(second));
  });
});
