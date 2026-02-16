import { readFile, writeFile } from 'fs/promises';
import { basename, resolve } from 'path';

type BenchmarkFile = {
  config: Record<string, unknown>;
  results: BenchmarkResult[];
};

type BenchmarkResult = {
  dataset: {
    name: string;
    metric: 'cosine' | 'euclidean';
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

type ComparisonRow = {
  key: string;
  datasetName: string;
  metric: string;
  dimension: number;
  count: number;
  queries: number;
  M: number;
  efConstruction: number;
  efSearch: number;
  k: number;
  recallBase: number;
  recallCandidate: number;
  recallDelta: number;
  latencyAvgBase: number;
  latencyAvgCandidate: number;
  latencyAvgDelta: number;
  latencyP95Base: number;
  latencyP95Candidate: number;
  latencyP95Delta: number;
  buildMsBase: number;
  buildMsCandidate: number;
  buildMsDelta: number;
};

type CliConfig = {
  basePath?: string;
  candidatePath?: string;
  baseDir?: string;
  candidateDir?: string;
  output?: string;
  format: 'json' | 'csv';
  strict: boolean;
};

function parseArgs(argv: string[]): CliConfig {
  const get = (flag: string) => {
    const idx = argv.indexOf(flag);
    if (idx === -1) return undefined;
    return argv[idx + 1];
  };

  return {
    basePath: get('--base'),
    candidatePath: get('--candidate'),
    baseDir: get('--base-dir'),
    candidateDir: get('--candidate-dir'),
    output: get('--output'),
    format: (get('--format') as CliConfig['format']) ?? 'json',
    strict: argv.includes('--strict'),
  };
}

function makeKey(result: BenchmarkResult): string {
  const d = result.dataset;
  const p = result.params;
  return [d.name, d.metric, d.dimension, d.count, d.queries, p.M, p.efConstruction, p.efSearch, p.k].join('|');
}

function summarizeDelta(values: number[]) {
  if (values.length === 0) return { avg: 0, min: 0, max: 0 };
  const avg = values.reduce((sum, v) => sum + v, 0) / values.length;
  const min = Math.min(...values);
  const max = Math.max(...values);
  return { avg, min, max };
}

async function readBenchmarkFile(filePath: string): Promise<BenchmarkFile> {
  const raw = await readFile(filePath, 'utf8');
  const parsed = JSON.parse(raw);
  if (!parsed?.results || !Array.isArray(parsed.results)) {
    throw new Error(`Invalid benchmark file: ${filePath}`);
  }
  return parsed as BenchmarkFile;
}

async function loadFromDir(dirPath: string): Promise<BenchmarkResult[]> {
  const { readdir } = await import('fs/promises');
  const entries = await readdir(dirPath, { withFileTypes: true });
  const files = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith('.json'))
    .map((entry) => resolve(dirPath, entry.name));

  const results: BenchmarkResult[] = [];
  for (const file of files) {
    const data = await readBenchmarkFile(file);
    results.push(...data.results);
  }
  return results;
}

function compareResults(base: BenchmarkResult[], candidate: BenchmarkResult[], strict: boolean) {
  const baseMap = new Map<string, BenchmarkResult>();
  const candidateMap = new Map<string, BenchmarkResult>();

  for (const item of base) {
    baseMap.set(makeKey(item), item);
  }
  for (const item of candidate) {
    candidateMap.set(makeKey(item), item);
  }

  const allKeys = new Set<string>([...baseMap.keys(), ...candidateMap.keys()]);
  const rows: ComparisonRow[] = [];
  const missingInBase: string[] = [];
  const missingInCandidate: string[] = [];

  for (const key of allKeys) {
    const baseItem = baseMap.get(key);
    const candItem = candidateMap.get(key);

    if (!baseItem) {
      missingInBase.push(key);
      if (strict) continue;
    }
    if (!candItem) {
      missingInCandidate.push(key);
      if (strict) continue;
    }
    if (!baseItem || !candItem) continue;

    rows.push({
      key,
      datasetName: baseItem.dataset.name,
      metric: baseItem.dataset.metric,
      dimension: baseItem.dataset.dimension,
      count: baseItem.dataset.count,
      queries: baseItem.dataset.queries,
      M: baseItem.params.M,
      efConstruction: baseItem.params.efConstruction,
      efSearch: baseItem.params.efSearch,
      k: baseItem.params.k,
      recallBase: baseItem.recallAtK,
      recallCandidate: candItem.recallAtK,
      recallDelta: candItem.recallAtK - baseItem.recallAtK,
      latencyAvgBase: baseItem.searchLatencyMs.avg,
      latencyAvgCandidate: candItem.searchLatencyMs.avg,
      latencyAvgDelta: candItem.searchLatencyMs.avg - baseItem.searchLatencyMs.avg,
      latencyP95Base: baseItem.searchLatencyMs.p95,
      latencyP95Candidate: candItem.searchLatencyMs.p95,
      latencyP95Delta: candItem.searchLatencyMs.p95 - baseItem.searchLatencyMs.p95,
      buildMsBase: baseItem.buildMs,
      buildMsCandidate: candItem.buildMs,
      buildMsDelta: candItem.buildMs - baseItem.buildMs,
    });
  }

  return { rows, missingInBase, missingInCandidate };
}

function toCsv(rows: ComparisonRow[]): string {
  const header = [
    'datasetName',
    'metric',
    'dimension',
    'count',
    'queries',
    'M',
    'efConstruction',
    'efSearch',
    'k',
    'recallBase',
    'recallCandidate',
    'recallDelta',
    'latencyAvgBase',
    'latencyAvgCandidate',
    'latencyAvgDelta',
    'latencyP95Base',
    'latencyP95Candidate',
    'latencyP95Delta',
    'buildMsBase',
    'buildMsCandidate',
    'buildMsDelta',
  ];
  const lines = [header.join(',')];

  for (const row of rows) {
    lines.push(
      [
        row.datasetName,
        row.metric,
        row.dimension,
        row.count,
        row.queries,
        row.M,
        row.efConstruction,
        row.efSearch,
        row.k,
        row.recallBase.toFixed(6),
        row.recallCandidate.toFixed(6),
        row.recallDelta.toFixed(6),
        row.latencyAvgBase.toFixed(6),
        row.latencyAvgCandidate.toFixed(6),
        row.latencyAvgDelta.toFixed(6),
        row.latencyP95Base.toFixed(6),
        row.latencyP95Candidate.toFixed(6),
        row.latencyP95Delta.toFixed(6),
        row.buildMsBase.toFixed(3),
        row.buildMsCandidate.toFixed(3),
        row.buildMsDelta.toFixed(3),
      ].join(','),
    );
  }

  return lines.join('\n');
}

function printSummary(rows: ComparisonRow[]) {
  const recallDeltas = rows.map((r) => r.recallDelta);
  const latencyAvgDeltas = rows.map((r) => r.latencyAvgDelta);
  const latencyP95Deltas = rows.map((r) => r.latencyP95Delta);
  const buildDeltas = rows.map((r) => r.buildMsDelta);

  const recallStats = summarizeDelta(recallDeltas);
  const latencyAvgStats = summarizeDelta(latencyAvgDeltas);
  const latencyP95Stats = summarizeDelta(latencyP95Deltas);
  const buildStats = summarizeDelta(buildDeltas);

  console.log('Summary (candidate - base):');
  console.log(
    `  recall@k  avg=${recallStats.avg.toFixed(6)} min=${recallStats.min.toFixed(6)} max=${recallStats.max.toFixed(6)}`,
  );
  console.log(
    `  latencyAvg ms  avg=${latencyAvgStats.avg.toFixed(6)} min=${latencyAvgStats.min.toFixed(
      6,
    )} max=${latencyAvgStats.max.toFixed(6)}`,
  );
  console.log(
    `  latencyP95 ms  avg=${latencyP95Stats.avg.toFixed(6)} min=${latencyP95Stats.min.toFixed(
      6,
    )} max=${latencyP95Stats.max.toFixed(6)}`,
  );
  console.log(
    `  buildMs  avg=${buildStats.avg.toFixed(3)} min=${buildStats.min.toFixed(3)} max=${buildStats.max.toFixed(3)}`,
  );
}

function printUsage() {
  const message = `
Usage:
  node dist/bench/report.js --base <file> --candidate <file> [--format json|csv] [--output <path>]
  node dist/bench/report.js --base-dir <dir> --candidate-dir <dir> [--format json|csv] [--output <path>]

Options:
  --base <file>           Baseline JSON file from bench outputs
  --candidate <file>      Candidate JSON file from bench outputs
  --base-dir <dir>        Directory of baseline JSON files
  --candidate-dir <dir>   Directory of candidate JSON files
  --format json|csv       Output format (default json)
  --output <path>         Write report to file (prints summary to stdout)
  --strict                Only compare keys present in both base and candidate

Examples:
  node dist/bench/report.js --base bench/outputs/base.json --candidate bench/outputs/new.json --format csv --output bench/outputs/compare.csv
  node dist/bench/report.js --base-dir bench/outputs/base --candidate-dir bench/outputs/new
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

  let baseResults: BenchmarkResult[] = [];
  let candidateResults: BenchmarkResult[] = [];

  if (config.basePath && config.candidatePath) {
    const baseFile = await readBenchmarkFile(config.basePath);
    const candidateFile = await readBenchmarkFile(config.candidatePath);
    baseResults = baseFile.results;
    candidateResults = candidateFile.results;
  } else if (config.baseDir && config.candidateDir) {
    baseResults = await loadFromDir(config.baseDir);
    candidateResults = await loadFromDir(config.candidateDir);
  } else {
    printUsage();
    process.exit(1);
  }

  const { rows, missingInBase, missingInCandidate } = compareResults(baseResults, candidateResults, config.strict);

  printSummary(rows);

  if (missingInBase.length > 0) {
    console.log(`Missing in base (${missingInBase.length}):`);
    console.log(missingInBase.map((key) => `  ${key}`).join('\n'));
  }
  if (missingInCandidate.length > 0) {
    console.log(`Missing in candidate (${missingInCandidate.length}):`);
    console.log(missingInCandidate.map((key) => `  ${key}`).join('\n'));
  }

  if (config.output) {
    const outputPath = resolve(config.output);
    const payload =
      config.format === 'csv' ? toCsv(rows) : JSON.stringify({ rows, missingInBase, missingInCandidate }, null, 2);

    await writeFile(outputPath, payload, 'utf8');
    console.log(`Report written to ${outputPath}`);
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  printUsage();
  process.exit(1);
});
