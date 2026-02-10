import { spawnSync } from 'child_process';
import { existsSync } from 'fs';
import { copyFile, mkdir, readdir, rm } from 'fs/promises';
import { resolve } from 'path';

type CliConfig = {
  baseRef: string;
  candidateRef: string;
  baseOut: string;
  candidateOut: string;
  reportOut?: string;
  reportFormat: 'json' | 'csv';
  allowDirty: boolean;
  tag?: string;
  runArgs: string[];
  runScriptPath: string;
  reportScriptPath: string;
  buildIfMissing: boolean;
  injectSources: boolean;
  injectDir: string;
};

function parseArgs(argv: string[]): CliConfig {
  const get = (flag: string) => {
    const idx = argv.indexOf(flag);
    if (idx === -1) return undefined;
    return argv[idx + 1];
  };
  const has = (flag: string) => argv.includes(flag);

  const baseRef = get('--base-ref') ?? 'HEAD~1';
  const candidateRef = get('--candidate-ref') ?? 'HEAD';
  const tag = get('--tag');

  const baseOut = get('--base-out') ?? resolve('bench/outputs', `baseline${tag ? `-${tag}` : ''}`);
  const candidateOut = get('--candidate-out') ?? resolve('bench/outputs', `candidate${tag ? `-${tag}` : ''}`);
  const reportOut = get('--report-out') ?? resolve('bench/outputs', `compare${tag ? `-${tag}` : ''}.csv`);
  const reportFormat = (get('--report-format') as 'json' | 'csv') ?? 'csv';
  const runScriptPathRaw = get('--bench-runner');
  const reportScriptPathRaw = get('--bench-reporter');
  const runScriptPath = resolve(runScriptPathRaw ?? 'dist/bench/run.js');
  const reportScriptPath = resolve(reportScriptPathRaw ?? 'dist/bench/report.js');
  const buildIfMissing = !runScriptPathRaw || !reportScriptPathRaw;
  const injectSources = has('--inject-sources');
  const injectDir = resolve(get('--inject-dir') ?? 'bench/outputs/bench-sources');

  const passthroughFlags = [
    '--mode',
    '--metric',
    '--base',
    '--query',
    '--limit',
    '--query-limit',
    '--count',
    '--dim',
    '--seed',
    '--distribution',
    '--k',
    '--efSearch',
    '--M',
    '--efConstruction',
  ];

  const runArgs: string[] = [];
  for (const flag of passthroughFlags) {
    const value = get(flag);
    if (value !== undefined) {
      runArgs.push(flag, value);
    }
  }

  return {
    baseRef,
    candidateRef,
    baseOut,
    candidateOut,
    reportOut,
    reportFormat,
    allowDirty: has('--allow-dirty'),
    tag,
    runArgs,
    runScriptPath,
    reportScriptPath,
    buildIfMissing,
    injectSources,
    injectDir,
  };
}

function runCommand(command: string, args: string[], cwd?: string) {
  const result = spawnSync(command, args, {
    cwd,
    stdio: 'inherit',
    env: process.env,
  });
  if (result.status !== 0) {
    throw new Error(`Command failed: ${command} ${args.join(' ')}`);
  }
}

function runCommandCapture(command: string, args: string[]) {
  const result = spawnSync(command, args, {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
    env: process.env,
  });
  if (result.status !== 0) {
    throw new Error(`Command failed: ${command} ${args.join(' ')}`);
  }
  return (result.stdout ?? '').trim();
}

function ensureCleanWorkingTree(allowDirty: boolean) {
  const status = runCommandCapture('git', ['status', '--porcelain']);
  if (status.length > 0 && !allowDirty) {
    throw new Error('Working tree is dirty. Commit/stash changes or pass --allow-dirty.');
  }
}

function getCurrentRef(): string {
  const ref = runCommandCapture('git', ['rev-parse', '--abbrev-ref', 'HEAD']);
  if (ref === 'HEAD') {
    return runCommandCapture('git', ['rev-parse', 'HEAD']);
  }
  return ref;
}

function checkout(ref: string) {
  runCommand('git', ['checkout', ref]);
}

function build() {
  runCommand('npm', ['run', 'build']);
}

function runBench(runScriptPath: string, runArgs: string[], outputDir: string) {
  const args = [runScriptPath, ...runArgs, '--output', outputDir];
  runCommand('node', args);
}

function runReport(
  reportScriptPath: string,
  baseDir: string,
  candidateDir: string,
  outputPath: string | undefined,
  format: 'json' | 'csv',
) {
  const args = [reportScriptPath, '--base-dir', baseDir, '--candidate-dir', candidateDir, '--format', format];
  if (outputPath) {
    args.push('--output', outputPath);
  }
  runCommand('node', args);
}

function ensureDistExists(runPath: string, reportPath: string, buildIfMissing: boolean) {
  const downloadPath = resolve('dist/bench/download.js');

  if ((!existsSync(runPath) || !existsSync(reportPath)) && buildIfMissing) {
    build();
  }
  if (!existsSync(runPath) || !existsSync(reportPath)) {
    throw new Error(`Benchmark runner scripts not found: ${runPath} or ${reportPath}`);
  }
  if (!existsSync(downloadPath)) {
    // download script is optional for comparison flow
  }
}

function getTrackedBenchFiles(): Set<string> {
  const output = runCommandCapture('git', ['ls-files', 'src/bench']);
  if (!output) return new Set();
  return new Set(
    output
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean),
  );
}

async function exportBenchSources(sourceDir: string, injectDir: string) {
  await mkdir(injectDir, { recursive: true });
  const entries = await readdir(sourceDir, { withFileTypes: true });
  for (const entry of entries) {
    if (!entry.isFile()) continue;
    if (!entry.name.endsWith('.ts')) continue;
    if (entry.name === 'compare.ts') continue;
    await copyFile(resolve(sourceDir, entry.name), resolve(injectDir, entry.name));
  }
}

async function injectBenchSources(injectDir: string, targetDir: string): Promise<string[]> {
  await mkdir(targetDir, { recursive: true });
  const entries = await readdir(injectDir, { withFileTypes: true });
  const injected: string[] = [];
  for (const entry of entries) {
    if (!entry.isFile()) continue;
    if (!entry.name.endsWith('.ts')) continue;
    const dest = resolve(targetDir, entry.name);
    await copyFile(resolve(injectDir, entry.name), dest);
    injected.push(`src/bench/${entry.name}`);
  }
  return injected;
}

async function cleanupInjectedSources(injected: string[], tracked: Set<string>) {
  for (const relPath of injected) {
    if (tracked.has(relPath)) continue;
    await rm(resolve(relPath), { force: true });
  }
}

function printUsage() {
  const message = `
Usage:
  node dist/bench/compare.js --base-ref <ref> --candidate-ref <ref> [options]

Options:
  --base-ref <ref>         Git ref for baseline (default HEAD~1)
  --candidate-ref <ref>    Git ref for candidate (default HEAD)
  --base-out <dir>         Output dir for baseline (default bench/outputs/baseline[-tag])
  --candidate-out <dir>    Output dir for candidate (default bench/outputs/candidate[-tag])
  --report-out <path>      Report output path (default bench/outputs/compare[-tag].csv)
  --report-format csv|json Report format (default csv)
  --tag <label>            Suffix for output dirs and report name
  --allow-dirty            Allow dirty working tree
  --bench-runner <path>    Path to bench runner JS (useful for older refs)
  --bench-reporter <path>  Path to bench report JS (useful for older refs)
  --inject-sources         Copy bench TS sources when the target ref lacks them
  --no-inject-sources      Disable bench source injection (default)
  --inject-dir <path>      Destination directory for injected sources

Examples:
  node dist/bench/compare.js --base-ref HEAD~1 --candidate-ref HEAD --mode fvecs --base bench/datasets/siftsmall_base.fvecs --query bench/datasets/siftsmall_query.fvecs --metric euclidean --bench-runner dist/bench/run.js --bench-reporter dist/bench/report.js

Passthrough to bench runner:
  --mode synthetic|fvecs
  --metric cosine|euclidean
  --base <path>
  --query <path>
  --limit <n>
  --query-limit <n>
  --count <n>
  --dim <n>
  --seed <n>
  --distribution uniform|gaussian
  --k <n>
  --efSearch <list>
  --M <list>
  --efConstruction <list>
`.trim();
  console.log(message);
}

async function main() {
  const argv = process.argv.slice(2);
  if (argv.includes('--help')) {
    printUsage();
    return;
  }

  let config = parseArgs(argv);
  ensureCleanWorkingTree(config.allowDirty);

  const originalRef = getCurrentRef();

  if (config.injectSources) {
    await exportBenchSources(resolve('src/bench'), config.injectDir);
  }

  try {
    checkout(config.baseRef);
    if (config.injectSources) {
      const tracked = getTrackedBenchFiles();
      if (tracked.size === 0) {
        const injected = await injectBenchSources(config.injectDir, resolve('src/bench'));
        ensureDistExists(config.runScriptPath, config.reportScriptPath, config.buildIfMissing);
        runBench(config.runScriptPath, config.runArgs, config.baseOut);
        await cleanupInjectedSources(injected, tracked);
      } else {
        ensureDistExists(config.runScriptPath, config.reportScriptPath, config.buildIfMissing);
        runBench(config.runScriptPath, config.runArgs, config.baseOut);
      }
    } else {
      ensureDistExists(config.runScriptPath, config.reportScriptPath, config.buildIfMissing);
      runBench(config.runScriptPath, config.runArgs, config.baseOut);
    }

    checkout(config.candidateRef);
    if (config.injectSources) {
      const tracked = getTrackedBenchFiles();
      if (tracked.size === 0) {
        const injected = await injectBenchSources(config.injectDir, resolve('src/bench'));
        ensureDistExists(config.runScriptPath, config.reportScriptPath, config.buildIfMissing);
        runBench(config.runScriptPath, config.runArgs, config.candidateOut);
        await cleanupInjectedSources(injected, tracked);
      } else {
        ensureDistExists(config.runScriptPath, config.reportScriptPath, config.buildIfMissing);
        runBench(config.runScriptPath, config.runArgs, config.candidateOut);
      }
    } else {
      ensureDistExists(config.runScriptPath, config.reportScriptPath, config.buildIfMissing);
      runBench(config.runScriptPath, config.runArgs, config.candidateOut);
    }

    runReport(config.reportScriptPath, config.baseOut, config.candidateOut, config.reportOut, config.reportFormat);
  } finally {
    if (originalRef) {
      checkout(originalRef);
    }
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  printUsage();
  process.exit(1);
});
