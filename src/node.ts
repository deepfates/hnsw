import { norm } from './similarity';

export class Node {
  id: number;
  level: number;
  vector: Float32Array | number[];
  neighbors: number[][];
  // Cached L2 norm, computed on first use and reused for every subsequent
  // cosine comparison. Valid because vectors are immutable once inserted (see
  // README "Vector immutability"): mutating a vector after insertion leaves
  // this cache stale by design. Lazy (rather than eager in the constructor) so
  // euclidean and custom-similarity indexes never pay for a norm they never
  // read. -1 is a safe "unset" sentinel: real norms are always >= 0. Nodes
  // rebuilt by fromJSON get a fresh instance, so norms are always recomputed
  // from the deserialized vector.
  private cachedNorm = -1;

  constructor(id: number, vector: Float32Array | number[], level: number) {
    this.id = id;
    this.vector = vector;
    this.level = level;
    this.neighbors = Array.from({ length: level + 1 }, () => [] as number[]);
  }

  get norm(): number {
    if (this.cachedNorm < 0) {
      this.cachedNorm = norm(this.vector);
    }
    return this.cachedNorm;
  }
}
