import { HNSW } from '../src';
import { Node } from '../src/node';

const createSequentialData = (count: number, dimensions = 5) =>
  Array.from({ length: count }, (_, index) => ({
    id: index + 1,
    vector: Array.from({ length: dimensions }, (_, dimIndex) => dimIndex + 1 + index),
  }));

const baseData = createSequentialData(5);

const buildBasicIndex = async (
  data = baseData,
  {
    levelSequence,
    metric = 'cosine',
    M = 16,
    efConstruction = 200,
    efSearch = 50,
  }: {
    levelSequence?: number[];
    metric?: 'cosine' | 'euclidean';
    M?: number;
    efConstruction?: number;
    efSearch?: number;
  } = {},
) => {
  const hnsw = new HNSW(M, efConstruction, data[0].vector.length, metric, efSearch);
  const sequence = levelSequence ? [...levelSequence] : undefined;
  const selectMock = sequence
    ? jest.spyOn(hnsw as any, 'selectLevel').mockImplementation(() => (sequence!.shift() as number) ?? 0)
    : null;
  await hnsw.buildIndex(data);
  selectMock?.mockRestore();
  return hnsw;
};

describe('HNSW', () => {
  it('performs a basic KNN search with deterministic ordering', async () => {
    const levelSequence = Array(baseData.length).fill(0);
    const hnsw = await buildBasicIndex(baseData, { levelSequence });
    const results = hnsw.searchKNN([3, 4, 5, 6, 7], 3);
    expect(results.map((result) => result.id)).toEqual([3, 4, 2]);
    expect(results[0].score).toBeCloseTo(1, 5);
  });

  it('selectLevel obeys the probability distribution derived from M', () => {
    const hnsw = new HNSW(4, 16);
    (hnsw as any).probs = [0.5, 0.3, 0.2];
    const mockRandom = jest.spyOn(Math, 'random');
    mockRandom
      .mockReturnValueOnce(0.2) // level 0
      .mockReturnValueOnce(0.6) // level 1
      .mockReturnValueOnce(0.95); // level 2

    const levels = [0, 1, 2].map(() => (hnsw as any).selectLevel());
    expect(levels).toEqual([0, 1, 2]);
    mockRandom.mockRestore();
  });

  it('promotes the highest-level node to the entry point', async () => {
    const hnsw = await buildBasicIndex(baseData, { levelSequence: [0, 3, 1, 0, 2] });
    expect((hnsw as any).entryPointId).toBe(2);
    expect((hnsw as any).levelMax).toBe(3);
  });

  it('uses efSearch for query-time exploration with overrides', async () => {
    const hnsw = await buildBasicIndex(baseData, { levelSequence: Array(baseData.length).fill(0), efSearch: 4 });
    const spy = jest.spyOn(hnsw as any, 'searchLayer');

    hnsw.searchKNN([3, 4, 5, 6, 7], 2);
    expect(spy).toHaveBeenNthCalledWith(1, [3, 4, 5, 6, 7], expect.anything(), 0, 4);

    hnsw.searchKNN([3, 4, 5, 6, 7], 2, { efSearch: 3 });
    expect(spy).toHaveBeenNthCalledWith(2, [3, 4, 5, 6, 7], expect.anything(), 0, 3);
    spy.mockRestore();
  });

  it('keeps only the M most similar neighbors per node', async () => {
    const linearData = [
      { id: 1, vector: [0, 0] },
      { id: 2, vector: [0, 1] },
      { id: 3, vector: [0, 2] },
      { id: 4, vector: [0, 3] },
    ];

    const hnsw = await buildBasicIndex(linearData, {
      levelSequence: Array(linearData.length).fill(0),
      metric: 'euclidean',
      M: 2,
      efConstruction: 16,
    });

    const node4 = (hnsw as any).nodes.get(4);
    expect(node4.neighbors[0]).toContain(3);
    expect(node4.neighbors[0]).toHaveLength(1);

    const node2 = (hnsw as any).nodes.get(2);
    expect(node2.neighbors[0].length).toBeLessThanOrEqual(2);
    expect(node2.neighbors[0]).toEqual(expect.arrayContaining([3, 1]));
  });

  it('selects diverse neighbors using the heuristic', async () => {
    const hnsw = new HNSW(2, 16, 2, 'euclidean');
    const center = new Node(1, [0, 0], 0);
    const candidateA = new Node(2, [1, 0], 0);
    const candidateB = new Node(3, [2, 0], 0);
    const candidateC = new Node(4, [0, 2], 0);

    const selected = (hnsw as any).selectNeighborsHeuristic(center, [candidateA, candidateB, candidateC], 2);
    const selectedIds = selected.map((node: Node) => node.id);
    expect(selectedIds).toEqual([2, 4]);
  });

  it('round-trips through JSON serialization without losing neighbors', async () => {
    const hnsw = await buildBasicIndex();
    const serialized = hnsw.toJSON();
    const restored = HNSW.fromJSON(serialized);

    const originalResults = hnsw.searchKNN([6, 7, 8, 9, 10], 2);
    const restoredResults = restored.searchKNN([6, 7, 8, 9, 10], 2);

    expect(restoredResults).toEqual(originalResults);
  });
});
