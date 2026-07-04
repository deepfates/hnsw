import { cosineSimilarity, euclideanSimilarity } from '../src/similarity';

describe('cosineSimilarity', () => {
  it('returns 0 instead of NaN when both vectors are zero', () => {
    const result = cosineSimilarity([0, 0, 0], [0, 0, 0]);
    expect(result).toBe(0);
    expect(Number.isNaN(result)).toBe(false);
  });

  it('returns 0 instead of NaN when only one vector is zero', () => {
    const result = cosineSimilarity([0, 0, 0], [1, 2, 3]);
    expect(result).toBe(0);
    expect(Number.isNaN(result)).toBe(false);
  });

  it('still computes normal cosine similarity for non-zero vectors', () => {
    expect(cosineSimilarity([1, 0], [1, 0])).toBeCloseTo(1, 5);
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0, 5);
  });
});

describe('euclideanSimilarity', () => {
  it('returns 1 for identical vectors including zero vectors', () => {
    expect(euclideanSimilarity([0, 0, 0], [0, 0, 0])).toBeCloseTo(1, 5);
  });
});
