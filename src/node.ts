import { norm } from './similarity';

export class Node {
  id: number;
  level: number;
  vector: Float32Array | number[];
  norm: number; // Cached L2 norm, so cosine needs one dot product per comparison
  neighbors: number[][];

  constructor(id: number, vector: Float32Array | number[], level: number) {
    this.id = id;
    this.vector = vector;
    this.norm = norm(vector);
    this.level = level;
    this.neighbors = Array.from({ length: level + 1 }, () => [] as number[]);
  }
}
