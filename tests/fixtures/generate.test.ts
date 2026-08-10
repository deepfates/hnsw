// Fixture generator for the cached-norm regression suite.
//
// Normally skipped. To (re)generate tests/fixtures/regression-v2.json from a
// reference implementation, check out the reference commit (the fixtures in
// git were produced on pre-cached-norm main, 8542a17), copy this file plus
// corpus.helper.ts over, and run:
//
//   GENERATE_FIXTURES=1 npx jest --config jestconfig.json tests/fixtures/generate.test.ts
//
// The fixture records, per corpus and metric: searchKNN results, the full
// toJSON() graph, and searchKNN results after a toJSON -> fromJSON round trip.
// It also records results with custom similarity functions assigned before and
// after construction.

import * as fs from 'fs';
import * as path from 'path';
import { HNSW } from '../../src';
import {
  asymmetricSimilarity,
  buildWithSeededLevels,
  FIXTURE_FILE,
  K,
  l1Similarity,
  makeCorpora,
} from './corpus.helper';

const maybeDescribe = process.env.GENERATE_FIXTURES ? describe : describe.skip;

maybeDescribe('generate regression fixtures', () => {
  it('writes regression-v2.json', async () => {
    const fixture: any = { corpora: {}, customFn: {} };

    for (const corpus of makeCorpora()) {
      for (const metric of ['cosine', 'euclidean'] as const) {
        const hnsw = new HNSW(16, 64, corpus.dims, metric, 64);
        await buildWithSeededLevels(hnsw, corpus.data);
        const searchResults = corpus.queries.map((q) => hnsw.searchKNN(q, K));
        const graph = JSON.parse(JSON.stringify(hnsw.toJSON()));
        const restored = HNSW.fromJSON(JSON.parse(JSON.stringify(graph)));
        const restoredSearchResults = corpus.queries.map((q) => restored.searchKNN(q, K));
        fixture.corpora[`${corpus.name}-${metric}`] = { searchResults, graph, restoredSearchResults };
      }
    }

    // Custom similarity assigned after construction (cosine-built graph).
    const d16 = makeCorpora().find((c) => c.name === 'd16')!;
    const postBuild = new HNSW(16, 64, d16.dims, 'cosine', 64);
    await buildWithSeededLevels(postBuild, d16.data);
    postBuild.similarityFunction = asymmetricSimilarity;
    fixture.customFn.postBuildAsymmetric = d16.queries.map((q) => postBuild.searchKNN(q, K));

    // Custom similarity assigned before construction (affects the graph too).
    const preBuild = new HNSW(16, 64, d16.dims, 'cosine', 64);
    preBuild.similarityFunction = l1Similarity;
    await buildWithSeededLevels(preBuild, d16.data);
    fixture.customFn.preBuildL1 = d16.queries.map((q) => preBuild.searchKNN(q, K));

    fs.writeFileSync(path.join(__dirname, FIXTURE_FILE), JSON.stringify(fixture));
  });
});
