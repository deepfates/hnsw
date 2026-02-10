import { BinaryHeap } from './heap';
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
  efSearch: number; // Max number of nodes to visit during search
  levelMax: number; // Max level of the graph
  entryPointId: number; // Id of the entry point
  nodes: Map<number, Node>; // Map of nodes
  probs: number[]; // Probabilities for the levels

  constructor(M = 16, efConstruction = 200, d: number | null = null, metric = 'cosine', efSearch = 50) {
    this.metric = metric as Metric;
    this.d = d;
    this.M = M;
    this.efConstruction = efConstruction;
    this.efSearch = efSearch;
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
    const candidates = new BinaryHeap<SearchCandidate>((a, b) => a.score - b.score);
    const best = new BinaryHeap<SearchCandidate>((a, b) => b.score - a.score);

    const entryScore = this.similarityFunction(query, entryNode.vector);
    candidates.push({ node: entryNode, score: entryScore });
    best.push({ node: entryNode, score: entryScore });

    while (candidates.size > 0) {
      const current = candidates.pop()!;
      const worstBest = best.peek();
      if (worstBest && best.size >= ef && current.score < worstBest.score) {
        break;
      }

      const neighbors = current.node.neighbors[level] ?? [];
      for (const neighborId of neighbors) {
        if (visited.has(neighborId)) continue;
        visited.add(neighborId);
        const neighborNode = this.nodes.get(neighborId)!;
        const score = this.similarityFunction(query, neighborNode.vector);
        if (best.size < ef || score > (best.peek()?.score ?? -Infinity)) {
          candidates.push({ node: neighborNode, score });
          best.push({ node: neighborNode, score });
          if (best.size > ef) {
            best.pop();
          }
        }
      }
    }

    return best
      .values()
      .sort((a, b) => b.score - a.score)
      .map((entry) => entry.node);
  }

  private connectNodeAtLevel(node: Node, candidates: Node[], level: number) {
    const selected = this.selectNeighborsHeuristic(node, candidates, this.M);

    for (const neighbor of selected) {
      this.addBidirectionalConnection(node, neighbor, level);
    }
  }

  private addBidirectionalConnection(node: Node, other: Node, level: number) {
    const removedFromNode = this.insertNeighbor(node, other.id, level);
    const removedFromOther = this.insertNeighbor(other, node.id, level);
    this.removeReciprocalLinks(node, removedFromNode, level);
    this.removeReciprocalLinks(other, removedFromOther, level);
  }

  private removeReciprocalLinks(node: Node, removedIds: number[], level: number) {
    for (const removedId of removedIds) {
      const removedNode = this.nodes.get(removedId);
      if (!removedNode) continue;
      removedNode.neighbors[level] = (removedNode.neighbors[level] ?? []).filter((id) => id !== node.id);
    }
  }

  private insertNeighbor(node: Node, neighborId: number, level: number): number[] {
    if (!node.neighbors[level]) {
      node.neighbors[level] = [];
    }

    const existingIds = node.neighbors[level].filter((id) => id !== neighborId);
    const candidateIds = [...existingIds, neighborId];
    const candidateNodes = candidateIds
      .map((id) => this.nodes.get(id))
      .filter((candidate): candidate is Node => Boolean(candidate));
    const selected = this.selectNeighborsHeuristic(node, candidateNodes, this.M);
    const selectedIds = selected.map((selectedNode) => selectedNode.id);
    const removedIds = existingIds.filter((id) => !selectedIds.includes(id));
    node.neighbors[level] = selectedIds;
    return removedIds;
  }

  private selectNeighborsHeuristic(node: Node, candidates: Node[], maxNeighbors: number): Node[] {
    const uniqueCandidates = new Map<number, Node>();
    for (const candidate of candidates) {
      if (candidate.id === node.id) continue;
      uniqueCandidates.set(candidate.id, candidate);
    }

    const scored = Array.from(uniqueCandidates.values())
      .map((candidate) => ({
        node: candidate,
        score: this.similarityFunction(node.vector, candidate.vector),
      }))
      .sort((a, b) => b.score - a.score);

    const selected: Node[] = [];
    for (const entry of scored) {
      if (selected.length >= maxNeighbors) {
        break;
      }
      const shouldSelect = selected.every((neighbor) => {
        const neighborSimilarity = this.similarityFunction(entry.node.vector, neighbor.vector);
        return neighborSimilarity <= entry.score;
      });
      if (shouldSelect) {
        selected.push(entry.node);
      }
    }

    return selected;
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

    if (this.nodes.has(id)) {
      throw new Error(`Node with id ${id} already exists`);
    }

    this.nodes.set(id, new Node(id, vector, this.selectLevel()));
    const node = this.nodes.get(id)!;

    await this.addNodeToGraph(node);
  }

  searchKNN(
    query: Float32Array | number[],
    k: number,
    options?: { efSearch?: number },
  ): { id: number; score: number }[] {
    if (this.entryPointId === -1 || this.nodes.size === 0 || k <= 0) {
      return [];
    }

    let entryNode = this.nodes.get(this.entryPointId)!;
    for (let level = this.levelMax; level > 0; level--) {
      entryNode = this.greedySearch(query, entryNode, level);
    }

    const ef = Math.max(k, options?.efSearch ?? this.efSearch);
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

  async buildIndex(
    data: { id: number; vector: Float32Array | number[] }[],
    options?: {
      onProgress?: (current: number, total: number) => void;
      progressInterval?: number;
    },
  ) {
    // Clear existing index
    this.nodes.clear();
    this.levelMax = -1;
    this.entryPointId = -1;
    this.d = null;

    const total = data.length;
    const interval = options?.progressInterval ?? 10000;
    const onProgress = options?.onProgress;

    // Add points to the index
    for (let i = 0; i < data.length; i++) {
      await this.addPoint(data[i].id, data[i].vector);
      if (onProgress && (i + 1) % interval === 0) {
        onProgress(i + 1, total);
      }
    }

    // Final progress call
    if (onProgress && total % interval !== 0) {
      onProgress(total, total);
    }
  }

  toJSON() {
    const entries = Array.from(this.nodes.entries());
    return {
      M: this.M,
      efConstruction: this.efConstruction,
      efSearch: this.efSearch,
      metric: this.metric,
      d: this.d,
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
    const hnsw = new HNSW(json.M, json.efConstruction, json.d ?? null, json.metric ?? 'cosine', json.efSearch ?? 50);
    hnsw.levelMax = json.levelMax;
    hnsw.entryPointId = json.entryPointId;
    hnsw.nodes = new Map(
      json.nodes.map(([id, node]: [number, any]) => {
        const restored = new Node(node.id, new Float32Array(node.vector), node.level);
        restored.neighbors = node.neighbors.map((level: number[]) => [...level]);
        return [id, restored];
      }),
    );
    return hnsw;
  }
}
