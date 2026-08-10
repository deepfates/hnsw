import { norm } from './similarity';

// Cached L2 norms, computed on first cosine use and reused for every later
// comparison. Held in a module-level WeakMap rather than on Node so Node's
// public structural shape is untouched (`HNSW.nodes` is a public
// Map<number, Node>; callers may insert structurally compatible plain
// objects, and any added field or getter would break them). Valid because
// vectors are immutable once inserted (see README "Vector immutability"):
// mutating a vector after insertion leaves the cache stale by design. Lazy so
// euclidean and custom-similarity indexes never pay for a norm they never
// read. Entries die with their nodes; nodes rebuilt by fromJSON are fresh
// keys, so norms are always recomputed from the deserialized vector.
const norms = new WeakMap<Node, number>();

export function nodeNorm(node: Node): number {
  let n = norms.get(node);
  if (n === undefined) {
    n = norm(node.vector);
    norms.set(node, n);
  }
  return n;
}

export class Node {
  id: number;
  level: number;
  vector: Float32Array | number[];
  neighbors: number[][];

  constructor(id: number, vector: Float32Array | number[], level: number) {
    this.id = id;
    this.vector = vector;
    this.level = level;
    this.neighbors = Array.from({ length: level + 1 }, () => [] as number[]);
  }
}
