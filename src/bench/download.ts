import { createWriteStream, existsSync, mkdirSync, statSync } from 'fs';
import { basename, resolve } from 'path';
import { pipeline } from 'stream/promises';
import { request } from 'https';
import { spawn } from 'child_process';

type DownloadOptions = {
  url: string;
  outDir: string;
  filename?: string;
  skipIfExists?: boolean;
  extract?: boolean;
  timeoutMs?: number;
};

const DEFAULT_URL = 'https://huggingface.co/datasets/vecdata/siftsmall/resolve/main/siftsmall.tar.gz?download=true';
const DEFAULT_OUT_DIR = 'bench/datasets';

function parseArgs(argv: string[]) {
  const get = (flag: string) => {
    const idx = argv.indexOf(flag);
    if (idx === -1) return undefined;
    return argv[idx + 1];
  };

  return {
    url: get('--url') ?? DEFAULT_URL,
    outDir: get('--out') ?? DEFAULT_OUT_DIR,
    filename: get('--name'),
    extract: argv.includes('--extract'),
    skipIfExists: argv.includes('--skip-if-exists'),
    timeoutMs: get('--timeout') ? Number(get('--timeout')) : 30000,
  };
}

async function downloadFile(options: DownloadOptions): Promise<string> {
  const { url, outDir, filename, skipIfExists = true, timeoutMs = 30000 } = options;

  mkdirSync(outDir, { recursive: true });

  const finalName = filename ?? (basename(new URL(url).pathname) || 'dataset.tar.gz');
  const outputPath = resolve(outDir, finalName);

  if (skipIfExists && existsSync(outputPath)) {
    const stats = statSync(outputPath);
    if (stats.size > 0) {
      console.log(`File already exists: ${outputPath}`);
      return outputPath;
    }
  }

  console.log(`Downloading ${url}`);
  console.log(`→ ${outputPath}`);

  await new Promise<void>((resolvePromise, rejectPromise) => {
    const req = request(url, { timeout: timeoutMs }, async (res) => {
      if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        // follow redirect - resolve relative URLs against the original
        res.resume();
        try {
          const redirectUrl = new URL(res.headers.location, url).toString();
          const redirected = await downloadFile({
            ...options,
            url: redirectUrl,
            filename: finalName,
          });
          resolvePromise();
        } catch (err) {
          rejectPromise(err);
        }
        return;
      }

      if (res.statusCode !== 200) {
        rejectPromise(new Error(`Download failed with status ${res.statusCode}`));
        res.resume();
        return;
      }

      const total = Number(res.headers['content-length'] ?? 0);
      let received = 0;

      res.on('data', (chunk: Buffer) => {
        received += chunk.length;
        if (total > 0) {
          const pct = ((received / total) * 100).toFixed(1);
          process.stdout.write(`\r${pct}% (${received}/${total} bytes)`);
        } else {
          process.stdout.write(`\r${received} bytes`);
        }
      });

      const fileStream = createWriteStream(outputPath);
      try {
        await pipeline(res, fileStream);
        process.stdout.write('\n');
        resolvePromise();
      } catch (err) {
        rejectPromise(err);
      }
    });

    req.on('error', rejectPromise);
    req.on('timeout', () => {
      req.destroy(new Error('Request timed out'));
    });
    req.end();
  });

  return outputPath;
}

async function extractTarGz(archivePath: string, outDir: string): Promise<void> {
  await new Promise<void>((resolvePromise, rejectPromise) => {
    const proc = spawn('tar', ['-xzf', archivePath, '-C', outDir], { stdio: 'inherit' });
    proc.on('error', rejectPromise);
    proc.on('close', (code) => {
      if (code === 0) resolvePromise();
      else rejectPromise(new Error(`tar exited with code ${code}`));
    });
  });
}

function printUsage() {
  const text = `
Usage:
  node dist/bench/download.js [--url <url>] [--out <dir>] [--name <filename>] [--extract] [--skip-if-exists]

Defaults:
  --url  ${DEFAULT_URL}
  --out  ${DEFAULT_OUT_DIR}

Examples:
  node dist/bench/download.js --extract
  node dist/bench/download.js --url ${DEFAULT_URL} --out bench/datasets --extract
`.trim();
  console.log(text);
}

async function main() {
  const argv = process.argv.slice(2);
  if (argv.includes('--help')) {
    printUsage();
    return;
  }

  const config = parseArgs(argv);
  const archivePath = await downloadFile({
    url: config.url,
    outDir: config.outDir,
    filename: config.filename,
    skipIfExists: config.skipIfExists,
    timeoutMs: config.timeoutMs,
  });

  if (config.extract) {
    console.log(`Extracting ${archivePath}`);
    await extractTarGz(archivePath, config.outDir);
  }
}

main().catch((err) => {
  console.error(err instanceof Error ? err.message : err);
  printUsage();
  process.exit(1);
});
