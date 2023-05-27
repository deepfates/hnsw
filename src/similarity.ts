// Note: Similarity functions
function dotProduct(a: Float32Array | number[], b: Float32Array | number[]): number {
  let dP = 0.0;
  for (let i = 0; i < a.length; i++) {
    dP += a[i] * b[i];
  }
  return dP;
}

export function cosineSimilarity(a: Float32Array | number[], b: Float32Array | number[]): number {
  return dotProduct(a, b) / (Math.sqrt(dotProduct(a, a)) * Math.sqrt(dotProduct(b, b)));
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
