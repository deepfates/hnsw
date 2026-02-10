import { mkdir, writeFile } from 'fs/promises';
import { basename, resolve } from 'path';
import { performance } from 'perf_hooks';
import { HNSW } from '../main';
import { cosineSimilarity, euclideanSimilarity } from '../similarity';
import { generateSyntheticDataset, loadFvecsDataset, Dataset, VectorRecord } from './dataset';
import { bruteForceKNN, meanRecall, summarizeLatencies, recallAtK } from './metrics';

type Metric = 'cosine' | 'euclidean';

type RunConfig = {
  mode: 'synthetic' | 'fvecs';
  metric: Metric;
  basePath?: string;
  queryPath?: string;
  limit?: number;
  queryLimit?: number;
  count?: number;
  dimension?: number;
  seed?: number;
  distribution?: 'uniform' | 'gaussian';
  k: number;
  efSearchList: number[];
  mList: number[];
  efConstructionList: number[];
  outputDir: string;
};

type RunResult = {
  dataset: {
    name: string;
    metric: Metric;
    dimension: number;
    count: number;
    queries: number;
  };
  params: {
    M: number;
    efConstruction: number;
    efSearch: number;
    k: number;
  };
  buildMs: number;
  searchLatencyMs: {
    count: number;
    avg: number;
    p50: number;
    p90: number;
    p95: number;
    p99: number;
  };
  recallAtK: number;
};

const DEFAULT_OUTPUT_DIR = 'bench/outputs';

function parseArgs(argv: string[]): RunConfig {
  const get = (flag: string): string | undefined => {
    const idx = argv.indexOf(flag);
    if (idx === -1) return undefined;
    return argv[idx + 1];
  };

  const getList = (flag: string, fallback: number[]): number[] => {
    const raw = get(flag);
    if (!raw) return fallback;
    return raw
      .split(',')
      .map((value) => Number(value.trim()))
      .filter((value) => Number.isFinite(value));
  };

  const mode = (get('--mode') as RunConfig['mode']) ?? 'synthetic';
  const metric = (get('--metric') as Metric) ?? 'cosine';

  const config: RunConfig = {
    mode,
    metric,
    basePath: get('--base'),
    queryPath: get('--query'),
    limit: get('--limit') ? Number(get('--limit')) : undefined,
    queryLimit: get('--query-limit') ? Number(get('--query-limit')) : undefined,
    count: get('--count') ? Number(get('--count')) : undefined,
    dimension: get('--dim') ? Number(get('--dim')) : undefined,
    seed: get('--seed') ? Number(get('--seed')) : 42,
    distribution: (get('--distribution') as 'uniform' | 'gaussian') ?? 'uniform',
    k: get('--k') ? Number(get('--k')) : 10,
    efSearchList: getList('--efSearch', [10, 20, 50, 100, 200]),
    mList: getList('--M', [8, 16, 32]),
    efConstructionList: getList('--efConstruction', [100, 200, 400]),
    outputDir: get('--output') ?? DEFAULT_OUTPUT_DIR,
  };

  if (config.mode === 'synthetic') {
    if (!config.count || !config.dimension) {
      throw new Error('Synthetic mode requires --count and --dim.');
    }
  } else {
    if (!config.basePath) {
      throw new Error('Vector file mode requires --base path.');
    }
  }

  return config;
}

function getSimilarity(metric: Metric) {
  return metric === 'cosine' ? cosineSimilarity : euclideanSimilarity;
}

async function loadDataset(config: RunConfig): Promise<{ base: Dataset; queries: VectorRecord[] }> {
  if (config.mode === 'synthetic') {
    const base = generateSyntheticDataset({
      count: config.count!,
      dimension: config.dimension!,
      metric: config.metric,
      seed: config.seed,
      distribution: config.distribution,
    });

    const queryCount = Math.min(100, base.vectors.length);
    const queries = base.vectors.slice(0, queryCount);
    return { base, queries };
  }

  const base = await loadFvecsDataset(config.basePath!, config.metric, {
    limit: config.limit,
  });

  const queries = config.queryPath
    ? (
        await loadFvecsDataset(config.queryPath, config.metric, {
          limit: config.queryLimit,
        })
      ).vectors
    : base.vectors.slice(0, Math.min(100, base.vectors.length));

  return { base, queries };
}

function asDataRecords(vectors: VectorRecord[]): Array<{ id: number; vector: Float32Array }> {
  return vectors.map((record) => ({ id: record.id, vector: record.vector }));
}

async function runBenchmark(config: RunConfig) {
  const { base, queries } = await loadDataset(config);
  if (base.vectors.length === 0) {
    throw new Error('Dataset contains zero vectors.');
  }
  if (queries.length === 0) {
    throw new Error('Query set is empty.');
  }

  const similarity = getSimilarity(config.metric);
  const baseData = asDataRecords(base.vectors);

  const results: RunResult[] = [];

  for (const M of config.mList) {
    for (const efConstruction of config.efConstructionList) {
      const hnsw = new HNSW(M, efConstruction, base.dimension, config.metric);
      console.log(`Building index: M=${M}, efConstruction=${efConstruction}, vectors=${base.vectors.length}`);
      const buildStart = performance.now();
      await hnsw.buildIndex(baseData, {
        progressInterval: Math.max(10000, Math.floor(base.vectors.length / 20)),
        onProgress: (current, total) => {
          const pct = ((current / total) * 100).toFixed(1);
          const elapsed = ((performance.now() - buildStart) / 1000).toFixed(1);
          const rate = (current / ((performance.now() - buildStart) / 1000)).toFixed(0);
          process.stdout.write(
            `\r  Progress: ${current.toLocaleString()}/${total.toLocaleString()} (${pct}%) - ${elapsed}s - ${rate} vec/s`,
          );
        },
      });
      const buildMs = performance.now() - buildStart;
      console.log(`\n  Build complete: ${(buildMs / 1000).toFixed(2)}s`);

      console.log(`  Computing ground truth (brute force on ${queries.length} queries)...`);
      const bruteStart = performance.now();
      const exact = queries.map((query, i) => {
        if ((i + 1) % 10 === 0 || i === queries.length - 1) {
          process.stdout.write(`\r  Brute force: ${i + 1}/${queries.length} queries`);
        }
        return bruteForceKNN(query.vector, baseData, similarity, config.k);
      });
      console.log(` - ${((performance.now() - bruteStart) / 1000).toFixed(2)}s`);

      for (const efSearch of config.efSearchList) {
        console.log(`  Searching with efSearch=${efSearch}...`);
        const latencies: number[] = [];
        const recalls = [];

        for (let i = 0; i < queries.length; i++) {
          const query = queries[i];
          const start = performance.now();
          const approx = hnsw.searchKNN(query.vector, config.k, { efSearch });
          latencies.push(performance.now() - start);
          recalls.push(recallAtK(approx, exact[i], config.k));
        }

        const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
        const recall = meanRecall(recalls);
        console.log(`    Recall: ${(recall * 100).toFixed(1)}%, Avg latency: ${avgLatency.toFixed(3)}ms`);

        results.push({
          dataset: {
            name: base.name,
            metric: config.metric,
            dimension: base.dimension,
            count: base.vectors.length,
            queries: queries.length,
          },
          params: {
            M,
            efConstruction,
            efSearch,
            k: config.k,
          },
          buildMs,
          searchLatencyMs: summarizeLatencies(latencies),
          recallAtK: meanRecall(recalls),
        });
      }
    }
  }

  return results;
}

async function writeResults(config: RunConfig, results: RunResult[]) {
  await mkdir(resolve(config.outputDir), { recursive: true });
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const datasetName =
    config.mode === 'synthetic' ? `synthetic-${config.count}x${config.dimension}` : basename(config.basePath!);
  const filePath = resolve(config.outputDir, `bench-${datasetName}-${timestamp}.json`);
  await writeFile(filePath, JSON.stringify({ config, results }, null, 2), 'utf8');
  return filePath;
}

function printUsage() {
  const message = `
Usage:
  node dist/bench/run.js --mode synthetic --count 10000 --dim 64 --metric cosine
  node dist/bench/run.js --mode fvecs --base ./bench/datasets/sift_base.fvecs --query ./bench/datasets/sift_query.fvecs --metric euclidean

Flags:
  --mode synthetic|fvecs
  --metric cosine|euclidean
  --base <path>           Base vectors file (fvecs)
  --query <path>          Query vectors file (optional)
  --limit <n>             Limit base vectors
  --query-limit <n>       Limit query vectors
  --count <n>             Synthetic count
  --dim <n>               Synthetic dimension
  --seed <n>              Synthetic seed
  --distribution uniform|gaussian
  --k <n>                 K for recall@k
  --efSearch <list>       Comma list, e.g. 10,50,100
  --M <list>              Comma list
  --efConstruction <list> Comma list
  --output <dir>          Output directory (default bench/outputs)
`.trim();
  console.log(message);
}

async function main() {
  const argv = process.argv.slice(2);
  if (argv.includes('--help')) {
    printUsage();
    return;
  }

  const config = parseArgs(argv);
  const results = await runBenchmark(config);
  const outputPath = await writeResults(config, results);

  console.log(`Saved results to ${outputPath}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  printUsage();
  process.exit(1);
});
