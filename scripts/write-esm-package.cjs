const { mkdirSync, writeFileSync } = require('node:fs');
const { join } = require('node:path');

const output = join(__dirname, '..', 'dist', 'esm');
mkdirSync(output, { recursive: true });
writeFileSync(join(output, 'package.json'), '{"type":"module"}\n');
