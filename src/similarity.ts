// Note: Similarity functions
export function dotProduct(a: Float32Array | number[], b: Float32Array | number[]): number {
  let dP = 0.0;
  for (let i = 0; i < a.length; i++) {
    dP += a[i] * b[i];
  }
  return dP;
}

export function norm(a: Float32Array | number[]): number {
  return Math.sqrt(dotProduct(a, a));
}

export function cosineSimilarity(a: Float32Array | number[], b: Float32Array | number[]): number {
  return cosineSimilarityFromNorms(a, b, norm(a), norm(b));
}

// Cosine similarity when the vector norms are already known: a single dot
// product instead of three. The hot path inside HNSW uses this with norms
// computed at most once per vector within a single operation.
export function cosineSimilarityFromNorms(
  a: Float32Array | number[],
  b: Float32Array | number[],
  normA: number,
  normB: number,
): number {
  const denominator = normA * normB;
  if (denominator === 0) {
    return 0;
  }
  return dotProduct(a, b) / denominator;
}

function euclideanDistance(a: Float32Array | number[], b: Float32Array | number[]): number {
  let sum = 0.0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] - b[i]) ** 2;
  }
  return Math.sqrt(sum);
}

export function euclideanSimilarity(a: Float32Array | number[], b: Float32Array | number[]): number {
  return 1 / (1 + euclideanDistance(a, b));
}
