import { HNSW, HNSWWithDB } from '../src';

async function main(): Promise<{ id: number; score: number }[]> {
  // Simple example
  const hnsw = new HNSW(200, 16, 5, 'cosine');

  // Make some data
  const data = [
    { id: 1, vector: [1, 2, 3, 4, 5] },
    { id: 2, vector: [2, 3, 4, 5, 6] },
    { id: 3, vector: [3, 4, 5, 6, 7] },
    { id: 4, vector: [4, 5, 6, 7, 8] },
    { id: 5, vector: [5, 6, 7, 8, 9] },
  ];

  // Build the index
  await hnsw.buildIndex(data);

  // Search for nearest neighbors
  const results = hnsw.searchKNN([6, 7, 8, 9, 10], 2);
  console.log(results);
  return results;

  // // Persistence is hard to test without a real database, but here's an example
  // const index = await HNSWWithPersistence.create(200, 16, 'my-index');
  // await index.buildIndex(data);
  // await index.saveIndex();

  // // Load the index
  // const index2 = await HNSWWithPersistence.create(200, 16, 'my-index-2');
  // await index2.loadIndex();

  // // Search for nearest neighbors
  // const results2 = index2.searchKNN([6, 7, 8, 9, 10], 2);
  // console.log(results2);
}

test('HNSW', async () => {
  const results = await main();
  expect(results).toEqual([
    { id: 1, score: 0.9649505047327671 },
    { id: 2, score: 0.9864400504156211 },
  ]);
});
