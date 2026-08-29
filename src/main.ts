import { BinaryHeap } from './heap.js';
import { Node, nodeNorm } from './node.js';
import { cosineSimilarity, cosineSimilarityFromNorms, euclideanSimilarity, norm } from './similarity.js';

type Metric = 'cosine' | 'euclidean';
type SearchCandidate = { node: Node; score: number };

// Cosine fast-path context: the norm of the vector being inserted or
// searched, computed once per operation. Stored-node norms are cached in a
// module-level WeakMap on first cosine use (see nodeNorm in node.ts), which is valid because vectors
// are immutable once inserted — see README "Vector immutability".
type CosineContext = { queryNorm: number };

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
  private readonly random: () => number;

  /**
   * Creates an in-memory HNSW index.
   */
  constructor(
    M = 16,
    efConstruction = 200,
    d: number | null = null,
    metric = 'cosine',
    efSearch?: number,
    random: () => number = () => Math.random(),
  ) {
    this.metric = metric as Metric;
    this.d = d;
    this.M = M;
    this.efConstruction = efConstruction;
    // Default efSearch to efConstruction for backward compatibility
    this.efSearch = efSearch ?? efConstruction;
    this.entryPointId = -1;
    this.nodes = new Map<number, Node>();
    this.probs = this.set_probs(M, 1 / Math.log(M));
    this.levelMax = -1;
    this.similarityFunction = this.getMetric(metric as Metric);
    this.random = random;
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

  // The fast path engages only while `similarityFunction` is still the stock
  // cosineSimilarity. Euclidean indexes and any user-assigned custom function
  // get the original per-comparison calls (same operand orientation) and never
  // pay for norm computation.
  private cosineContext(query: Float32Array | number[], queryNorm?: number): CosineContext | null {
    if (this.similarityFunction !== cosineSimilarity) {
      return null;
    }
    return { queryNorm: queryNorm ?? norm(query) };
  }

  // Traversal scoring; pre-optimization operand order was (query, node.vector).
  private scoreQueryToNode(query: Float32Array | number[], node: Node, ctx: CosineContext | null): number {
    if (ctx) {
      return cosineSimilarityFromNorms(query, node.vector, ctx.queryNorm, nodeNorm(node));
    }
    return this.similarityFunction(query, node.vector);
  }

  // Final result scoring; pre-optimization operand order was (node.vector, query).
  private scoreNodeToQuery(node: Node, query: Float32Array | number[], ctx: CosineContext | null): number {
    if (ctx) {
      return cosineSimilarityFromNorms(node.vector, query, nodeNorm(node), ctx.queryNorm);
    }
    return this.similarityFunction(node.vector, query);
  }

  // Neighbor-selection scoring; pre-optimization operand order was (a.vector, b.vector).
  private scoreNodeToNode(a: Node, b: Node, ctx: CosineContext | null): number {
    if (ctx) {
      return cosineSimilarityFromNorms(a.vector, b.vector, nodeNorm(a), nodeNorm(b));
    }
    return this.similarityFunction(a.vector, b.vector);
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
    let r = this.random();
    if (!Number.isFinite(r) || r < 0 || r >= 1) {
      throw new Error('Random source must return a finite number in [0, 1)');
    }
    for (let i = 0; i < this.probs.length; i++) {
      const p = this.probs[i];
      if (r < p) {
        return i;
      }
      r -= p;
    }
    return this.probs.length - 1;
  }

  private greedySearch(
    query: Float32Array | number[],
    entryNode: Node,
    level: number,
    ctx: CosineContext | null,
  ): Node {
    let bestNode = entryNode;
    let bestScore = this.scoreQueryToNode(query, entryNode, ctx);
    let improved = true;

    while (improved) {
      improved = false;
      const neighbors = bestNode.neighbors[level] ?? [];
      for (const neighborId of neighbors) {
        const neighborNode = this.nodes.get(neighborId)!;
        const similarity = this.scoreQueryToNode(query, neighborNode, ctx);
        if (similarity > bestScore) {
          bestScore = similarity;
          bestNode = neighborNode;
          improved = true;
        }
      }
    }

    return bestNode;
  }

  private searchLayer(
    query: Float32Array | number[],
    entryNode: Node,
    level: number,
    ef: number,
    ctx: CosineContext | null,
  ): Node[] {
    const visited = new Set<number>([entryNode.id]);
    const candidates = new BinaryHeap<SearchCandidate>((a, b) => a.score - b.score);
    const best = new BinaryHeap<SearchCandidate>((a, b) => b.score - a.score);

    const entryScore = this.scoreQueryToNode(query, entryNode, ctx);
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
        const score = this.scoreQueryToNode(query, neighborNode, ctx);
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

  private connectNodeAtLevel(node: Node, candidates: Node[], level: number, ctx: CosineContext | null) {
    const selected = this.selectNeighborsHeuristic(node, candidates, this.M, ctx);

    for (const neighbor of selected) {
      this.addBidirectionalConnection(node, neighbor, level, ctx);
    }
  }

  private addBidirectionalConnection(node: Node, other: Node, level: number, ctx: CosineContext | null) {
    const removedFromNode = this.insertNeighbor(node, other.id, level, ctx);
    const removedFromOther = this.insertNeighbor(other, node.id, level, ctx);
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

  private insertNeighbor(node: Node, neighborId: number, level: number, ctx: CosineContext | null): number[] {
    if (!node.neighbors[level]) {
      node.neighbors[level] = [];
    }

    const existingIds = node.neighbors[level].filter((id) => id !== neighborId);
    const candidateIds = [...existingIds, neighborId];
    const candidateNodes = candidateIds
      .map((id) => this.nodes.get(id))
      .filter((candidate): candidate is Node => Boolean(candidate));
    const selected = this.selectNeighborsHeuristic(node, candidateNodes, this.M, ctx);
    const selectedIds = selected.map((selectedNode) => selectedNode.id);
    const removedIds = existingIds.filter((id) => !selectedIds.includes(id));
    node.neighbors[level] = selectedIds;
    return removedIds;
  }

  private selectNeighborsHeuristic(
    node: Node,
    candidates: Node[],
    maxNeighbors: number,
    ctx: CosineContext | null,
  ): Node[] {
    const uniqueCandidates = new Map<number, Node>();
    for (const candidate of candidates) {
      if (candidate.id === node.id) continue;
      uniqueCandidates.set(candidate.id, candidate);
    }

    const scored = Array.from(uniqueCandidates.values())
      .map((candidate) => ({
        node: candidate,
        score: this.scoreNodeToNode(node, candidate, ctx),
      }))
      .sort((a, b) => b.score - a.score);

    const selected: Node[] = [];
    for (const entry of scored) {
      if (selected.length >= maxNeighbors) {
        break;
      }
      const shouldSelect = selected.every((neighbor) => {
        const neighborSimilarity = this.scoreNodeToNode(entry.node, neighbor, ctx);
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

    const ctx = this.cosineContext(node.vector, nodeNorm(node));
    const currentMaxLevel = this.levelMax;
    let entryNode = this.nodes.get(this.entryPointId)!;

    for (let level = currentMaxLevel; level > node.level; level--) {
      entryNode = this.greedySearch(node.vector, entryNode, level, ctx);
    }

    const targetLevel = Math.min(node.level, currentMaxLevel);
    for (let level = targetLevel; level >= 0; level--) {
      const neighbors = this.searchLayer(node.vector, entryNode, level, this.efConstruction, ctx);
      this.connectNodeAtLevel(node, neighbors, level, ctx);
      if (neighbors.length > 0) {
        entryNode = neighbors[0];
      }
    }

    if (node.level > this.levelMax) {
      this.entryPointId = node.id;
      this.levelMax = node.level;
    }
  }

  /**
   * Adds a single vector to the graph.
   *
   * The vector must not be mutated after insertion: the index stores it by
   * reference and caches derived values (its L2 norm) at insert time, so
   * later mutation yields stale — though stable — scores.
   */
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

  /**
   * Returns up to k nearest neighbors for the query vector.
   */
  searchKNN(
    query: Float32Array | number[],
    k: number,
    options?: { efSearch?: number },
  ): { id: number; score: number }[] {
    if (this.entryPointId === -1 || this.nodes.size === 0 || k <= 0) {
      return [];
    }

    if (this.d !== null && query.length !== this.d) {
      throw new Error(`Query vector must have dimension ${this.d}, got ${query.length}`);
    }

    const ctx = this.cosineContext(query);
    let entryNode = this.nodes.get(this.entryPointId)!;
    for (let level = this.levelMax; level > 0; level--) {
      entryNode = this.greedySearch(query, entryNode, level, ctx);
    }

    const ef = Math.max(k, options?.efSearch ?? this.efSearch);
    const candidates = this.searchLayer(query, entryNode, 0, ef, ctx);
    const results: { id: number; score: number }[] = [];
    const seen = new Set<number>();

    for (const node of candidates) {
      if (seen.has(node.id)) continue;
      seen.add(node.id);
      const score = this.scoreNodeToQuery(node, query, ctx);
      results.push({ id: node.id, score });
      if (results.length === k) {
        break;
      }
    }

    return results;
  }

  /**
   * Rebuilds the graph from the provided data.
   */
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

  /**
   * Serializes the current in-memory index.
   */
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

  /**
   * Restores an index from serialized JSON produced by toJSON().
   */
  static fromJSON(json: any): HNSW {
    // efSearch defaults to efConstruction if not present (backward compatibility)
    const hnsw = new HNSW(json.M, json.efConstruction, json.d ?? null, json.metric ?? 'cosine', json.efSearch);
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
