import { readFile } from 'fs/promises';
import { createReadStream } from 'fs';
import { basename } from 'path';

export type VectorRecord = { id: number; vector: Float32Array };
export type Dataset = {
  name: string;
  dimension: number;
  metric: 'cosine' | 'euclidean';
  vectors: VectorRecord[];
};

export type FVecsLoadOptions = {
  limit?: number;
  offset?: number;
  asFloat32?: boolean;
};

export type SyntheticOptions = {
  name?: string;
  count: number;
  dimension: number;
  metric: 'cosine' | 'euclidean';
  seed?: number;
  distribution?: 'uniform' | 'gaussian';
};

export function generateSyntheticDataset(options: SyntheticOptions): Dataset {
  const {
    name = `synthetic-${options.count}x${options.dimension}`,
    count,
    dimension,
    metric,
    seed = 42,
    distribution = 'uniform',
  } = options;

  const rng = mulberry32(seed);
  const vectors: VectorRecord[] = [];

  for (let i = 0; i < count; i++) {
    const vector = new Float32Array(dimension);
    for (let d = 0; d < dimension; d++) {
      vector[d] = distribution === 'gaussian' ? gaussian(rng) : rng() * 2 - 1;
    }
    vectors.push({ id: i, vector });
  }

  return { name, dimension, metric, vectors };
}

export async function loadFvecsDataset(
  filePath: string,
  metric: 'cosine' | 'euclidean',
  options: FVecsLoadOptions = {},
): Promise<Dataset> {
  const { limit, offset = 0, asFloat32 = true } = options;
  const name = basename(filePath).replace(/\.[^/.]+$/, '');
  const vectors = await readFvecs(filePath, { limit, offset, asFloat32 });
  const dimension = vectors.length > 0 ? vectors[0].vector.length : 0;

  return { name, dimension, metric, vectors };
}

export async function readFvecs(
  filePath: string,
  options: FVecsLoadOptions = {},
): Promise<VectorRecord[]> {
  const { limit, offset = 0, asFloat32 = true } = options;
  const data = await readFile(filePath);
  return decodeFvecsBuffer(data, { limit, offset, asFloat32 });
}

export function decodeFvecsBuffer(
  buffer: Buffer,
  options: FVecsLoadOptions = {},
): VectorRecord[] {
  const { limit, offset = 0, asFloat32 = true } = options;

  const vectors: VectorRecord[] = [];
  let cursor = 0;
  let index = 0;

  while (cursor + 4 <= buffer.length) {
    const dim = buffer.readInt32LE(cursor);
    cursor += 4;

    const byteCount = dim * 4;
    if (cursor + byteCount > buffer.length) {
      break;
    }

    const shouldTake = index >= offset && (limit === undefined || vectors.length < limit);
    if (shouldTake) {
      const vector = new Float32Array(dim);
      for (let i = 0; i < dim; i++) {
        vector[i] = buffer.readFloatLE(cursor + i * 4);
      }
      vectors.push({ id: index, vector: asFloat32 ? vector : new Float32Array(vector) });
    }

    cursor += byteCount;
    index++;
  }

  return vectors;
}

export async function loadIvecsAsFloatDataset(
  filePath: string,
  metric: 'cosine' | 'euclidean',
  options: FVecsLoadOptions = {},
): Promise<Dataset> {
  const { limit, offset = 0 } = options;
  const name = basename(filePath).replace(/\.[^/.]+$/, '');
  const vectors = await readIvecsAsFloat(filePath, { limit, offset });
  const dimension = vectors.length > 0 ? vectors[0].vector.length : 0;
  return { name, dimension, metric, vectors };
}

export async function readIvecsAsFloat(
  filePath: string,
  options: FVecsLoadOptions = {},
): Promise<VectorRecord[]> {
  const { limit, offset = 0 } = options;
  const data = await readFile(filePath);
  return decodeIvecsAsFloatBuffer(data, { limit, offset });
}

export function decodeIvecsAsFloatBuffer(
  buffer: Buffer,
  options: FVecsLoadOptions = {},
): VectorRecord[] {
  const { limit, offset = 0 } = options;

  const vectors: VectorRecord[] = [];
  let cursor = 0;
  let index = 0;

  while (cursor + 4 <= buffer.length) {
    const dim = buffer.readInt32LE(cursor);
    cursor += 4;

    const byteCount = dim * 4;
    if (cursor + byteCount > buffer.length) {
      break;
    }

    const shouldTake = index >= offset && (limit === undefined || vectors.length < limit);
    if (shouldTake) {
      const vector = new Float32Array(dim);
      for (let i = 0; i < dim; i++) {
        vector[i] = buffer.readInt32LE(cursor + i * 4);
      }
      vectors.push({ id: index, vector });
    }

    cursor += byteCount;
    index++;
  }

  return vectors;
}

export async function streamFvecs(
  filePath: string,
  onVector: (record: VectorRecord) => Promise<void> | void,
  options: FVecsLoadOptions = {},
): Promise<void> {
  const { limit, offset = 0 } = options;
  const stream = createReadStream(filePath);
  let buffer = Buffer.alloc(0);
  let index = 0;
  let emitted = 0;

  for await (const chunk of stream) {
    buffer = Buffer.concat([buffer, chunk as Buffer]);

    while (buffer.length >= 4) {
      const dim = buffer.readInt32LE(0);
      const byteCount = 4 + dim * 4;
      if (buffer.length < byteCount) {
        break;
      }

      const recordBuffer = buffer.subarray(0, byteCount);
      buffer = buffer.subarray(byteCount);

      const shouldTake = index >= offset && (limit === undefined || emitted < limit);
      if (shouldTake) {
        const vector = new Float32Array(dim);
        for (let i = 0; i < dim; i++) {
          vector[i] = recordBuffer.readFloatLE(4 + i * 4);
        }
        await onVector({ id: index, vector });
        emitted++;
      }

      index++;
      if (limit !== undefined && emitted >= limit) {
        stream.close();
        return;
      }
    }
  }
}

function mulberry32(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
