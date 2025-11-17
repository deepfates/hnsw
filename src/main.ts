import { Node } from './node';
import { cosineSimilarity, euclideanSimilarity } from './similarity';

type Metric = 'cosine' | 'euclidean';
type SearchCandidate = { node: Node; score: number };

export class HNSW {
  metric: Metric; // Metric to use
  similarityFunction: (a: number[] | Float32Array, b: number[] | Float32Array) => number;
  d: number | null = null; // Dimension of the vectors
  M: number; // Max number of neighbors
  efConstruction: number; // Max number of nodes to visit during construction
  levelMax: number; // Max level of the graph
  entryPointId: number; // Id of the entry point
  nodes: Map<number, Node>; // Map of nodes
  probs: number[]; // Probabilities for the levels

  constructor(M = 16, efConstruction = 200, d: number | null = null, metric = 'cosine') {
    this.metric = metric as Metric;
    this.d = d;
    this.M = M;
    this.efConstruction = efConstruction;
    this.entryPointId = -1;
    this.nodes = new Map<number, Node>();
    this.probs = this.set_probs(M, 1 / Math.log(M));
    this.levelMax = -1;
    this.similarityFunction = this.getMetric(metric as Metric);
  }

  private getMetric(metric: Metric): (a: number[] | Float32Array, b: number[] | Float32Array) => number {
    if (metric === 'cosine') {
      return cosineSimilarity;
    } else if (metric === 'euclidean') {
      return euclideanSimilarity;
    } else {
      throw new Error('Invalid metric');
    }
  }

  private set_probs(M: number, levelMult: number): number[] {
    let level = 0;
    const probs = [];
    while (true) {
      const prob = Math.exp(-level / levelMult) * (1 - Math.exp(-1 / levelMult));
      if (prob < 1e-9) break;
      probs.push(prob);
      level++;
    }
    return probs;
  }

  private selectLevel(): number {
    let r = Math.random();
    for (let i = 0; i < this.probs.length; i++) {
      const p = this.probs[i];
      if (r < p) {
        return i;
      }
      r -= p;
    }
    return this.probs.length - 1;
  }

  private insertCandidate(list: SearchCandidate[], candidate: SearchCandidate, limit?: number) {
    let index = 0;
    while (index < list.length && list[index].score >= candidate.score) {
      index++;
    }
    list.splice(index, 0, candidate);
    if (typeof limit === 'number' && list.length > limit) {
      list.length = limit;
    }
  }

  private greedySearch(query: Float32Array | number[], entryNode: Node, level: number): Node {
    let bestNode = entryNode;
    let bestScore = this.similarityFunction(query, entryNode.vector);
    let improved = true;

    while (improved) {
      improved = false;
      const neighbors = bestNode.neighbors[level] ?? [];
      for (const neighborId of neighbors) {
        const neighborNode = this.nodes.get(neighborId)!;
        const similarity = this.similarityFunction(query, neighborNode.vector);
        if (similarity > bestScore) {
          bestScore = similarity;
          bestNode = neighborNode;
          improved = true;
        }
      }
    }

    return bestNode;
  }

  private searchLayer(query: Float32Array | number[], entryNode: Node, level: number, ef: number): Node[] {
    const visited = new Set<number>([entryNode.id]);
    const queue: SearchCandidate[] = [];
    const best: SearchCandidate[] = [];

    const entryScore = this.similarityFunction(query, entryNode.vector);
    this.insertCandidate(queue, { node: entryNode, score: entryScore });
    this.insertCandidate(best, { node: entryNode, score: entryScore }, ef);

    while (queue.length > 0) {
      const current = queue.shift()!;
      const neighbors = current.node.neighbors[level] ?? [];
      for (const neighborId of neighbors) {
        if (visited.has(neighborId)) continue;
        visited.add(neighborId);
        const neighborNode = this.nodes.get(neighborId)!;
        const score = this.similarityFunction(query, neighborNode.vector);
        if (best.length < ef || score > best[best.length - 1].score) {
          this.insertCandidate(queue, { node: neighborNode, score });
          this.insertCandidate(best, { node: neighborNode, score }, ef);
        }
      }
    }

    return best.map((entry) => entry.node);
  }

  private connectNodeAtLevel(node: Node, candidates: Node[], level: number) {
    const selected: Node[] = [];
    const seen = new Set<number>();

    for (const candidate of candidates) {
      if (candidate.id === node.id || seen.has(candidate.id)) continue;
      seen.add(candidate.id);
      selected.push(candidate);
      if (selected.length === this.M) {
        break;
      }
    }

    for (const neighbor of selected) {
      this.addBidirectionalConnection(node, neighbor, level);
    }
  }

  private addBidirectionalConnection(node: Node, other: Node, level: number) {
    this.insertNeighbor(node, other.id, level);
    this.insertNeighbor(other, node.id, level);
  }

  private insertNeighbor(node: Node, neighborId: number, level: number) {
    if (!node.neighbors[level]) {
      node.neighbors[level] = [];
    }

    const neighborList = node.neighbors[level].filter((id) => id !== neighborId);
    neighborList.push(neighborId);
    neighborList.sort((a, b) => {
      const simB = this.similarityFunction(node.vector, this.nodes.get(b)!.vector);
      const simA = this.similarityFunction(node.vector, this.nodes.get(a)!.vector);
      return simB - simA;
    });

    if (neighborList.length > this.M) {
      neighborList.length = this.M;
    }

    node.neighbors[level] = neighborList;
  }

  private async addNodeToGraph(node: Node) {
    if (this.entryPointId === -1) {
      this.entryPointId = node.id;
      this.levelMax = node.level;
      return;
    }

    const currentMaxLevel = this.levelMax;
    let entryNode = this.nodes.get(this.entryPointId)!;

    for (let level = currentMaxLevel; level > node.level; level--) {
      entryNode = this.greedySearch(node.vector, entryNode, level);
    }

    const targetLevel = Math.min(node.level, currentMaxLevel);
    for (let level = targetLevel; level >= 0; level--) {
      const neighbors = this.searchLayer(node.vector, entryNode, level, this.efConstruction);
      this.connectNodeAtLevel(node, neighbors, level);
      if (neighbors.length > 0) {
        entryNode = neighbors[0];
      }
    }

    if (node.level > this.levelMax) {
      this.entryPointId = node.id;
      this.levelMax = node.level;
    }
  }

  async addPoint(id: number, vector: Float32Array | number[]) {
    if (this.d !== null && vector.length !== this.d) {
      throw new Error('All vectors must be of the same dimension');
    }
    this.d = vector.length;

    this.nodes.set(id, new Node(id, vector, this.selectLevel()));
    const node = this.nodes.get(id)!;

    await this.addNodeToGraph(node);
  }

  searchKNN(query: Float32Array | number[], k: number): { id: number; score: number }[] {
    if (this.entryPointId === -1 || this.nodes.size === 0) {
      return [];
    }

    let entryNode = this.nodes.get(this.entryPointId)!;
    for (let level = this.levelMax; level > 0; level--) {
      entryNode = this.greedySearch(query, entryNode, level);
    }

    const ef = Math.max(k, this.efConstruction);
    const candidates = this.searchLayer(query, entryNode, 0, ef);
    const results: { id: number; score: number }[] = [];
    const seen = new Set<number>();

    for (const node of candidates) {
      if (seen.has(node.id)) continue;
      seen.add(node.id);
      const score = this.similarityFunction(node.vector, query);
      results.push({ id: node.id, score });
      if (results.length === k) {
        break;
      }
    }

    return results;
  }

  async buildIndex(data: { id: number; vector: Float32Array | number[] }[]) {
    // Clear existing index
    this.nodes.clear();
    this.levelMax = -1;
    this.entryPointId = -1;

    // Add points to the index
    for (const item of data) {
      await this.addPoint(item.id, item.vector);
    }
  }

  toJSON() {
    const entries = Array.from(this.nodes.entries());
    return {
      M: this.M,
      efConstruction: this.efConstruction,
      levelMax: this.levelMax,
      entryPointId: this.entryPointId,
      nodes: entries.map(([id, node]) => {
        return [
          id,
          {
            id: node.id,
            level: node.level,
            vector: Array.from(node.vector),
            neighbors: node.neighbors.map((level) => Array.from(level)),
          },
        ];
      }),
    };
  }

  static fromJSON(json: any): HNSW {
    const hnsw = new HNSW(json.M, json.efConstruction);
    hnsw.levelMax = json.levelMax;
    hnsw.entryPointId = json.entryPointId;
    hnsw.nodes = new Map(
      json.nodes.map(([id, node]: [number, any]) => {
        return [
          id,
          {
            ...node,
            vector: new Float32Array(node.vector),
          },
        ];
      }),
    );
    return hnsw;
  }
}
