import { HNSW } from './main';
import { Node } from './node';
import { norm } from './similarity';

const MAGIC = [0x48, 0x4e, 0x53, 0x57] as const; // HNSW
const FORMAT_VERSION = 1;
const HEADER_BYTES = 36;
const MAX_DIMENSIONS = 1_000_000;
const MAX_NODES = 10_000_000;

export type VectorEncoding = 'float32' | 'int16';

export interface BinarySerializationOptions {
  /**
   * int16 is compact and preserves cosine direction. It is intentionally
   * unavailable for euclidean indexes because per-vector normalization would
   * change euclidean distances.
   */
  vectorEncoding?: VectorEncoding;
}

function assertInt32(value: number, label: string): void {
  if (!Number.isInteger(value) || value < -0x80000000 || value > 0x7fffffff) {
    throw new Error(`${label} must be a signed 32-bit integer`);
  }
}

function checkedSize(index: HNSW, bytesPerValue: number): number {
  if (!index.d || index.d < 1 || index.d > MAX_DIMENSIONS) {
    throw new Error(`Cannot serialize invalid vector dimension ${index.d}`);
  }
  if (index.nodes.size > MAX_NODES) {
    throw new Error(`Cannot serialize more than ${MAX_NODES} nodes`);
  }

  let size = HEADER_BYTES;
  for (const [id, node] of index.nodes) {
    assertInt32(id, 'Node id');
    if (!Number.isInteger(node.level) || node.level < 0 || node.level > 0xffff) {
      throw new Error(`Node ${id} has an invalid level`);
    }
    if (node.vector.length !== index.d) {
      throw new Error(`Node ${id} has ${node.vector.length} dimensions; expected ${index.d}`);
    }
    size += 8 + index.d * bytesPerValue;
    for (let level = 0; level <= node.level; level++) {
      const neighbors = node.neighbors[level] ?? [];
      size += 4 + neighbors.length * 4;
      for (const neighbor of neighbors) assertInt32(neighbor, `Neighbor id on node ${id}`);
    }
    if (!Number.isSafeInteger(size)) throw new Error('Serialized index is too large');
  }
  return size;
}

export function serializeHNSW(index: HNSW, options: BinarySerializationOptions = {}): Uint8Array {
  const encoding = options.vectorEncoding ?? 'float32';
  if (encoding === 'int16' && index.metric !== 'cosine') {
    throw new Error('int16 vector encoding is only valid for cosine indexes');
  }
  assertInt32(index.entryPointId, 'Entry point id');
  const bytesPerValue = encoding === 'int16' ? 2 : 4;
  const output = new Uint8Array(checkedSize(index, bytesPerValue));
  const view = new DataView(output.buffer);
  let offset = 0;

  for (const byte of MAGIC) output[offset++] = byte;
  view.setUint16(offset, FORMAT_VERSION, true);
  offset += 2;
  view.setUint8(offset++, index.metric === 'cosine' ? 0 : 1);
  view.setUint8(offset++, encoding === 'float32' ? 0 : 1);
  view.setUint32(offset, index.d!, true);
  offset += 4;
  view.setUint32(offset, index.M, true);
  offset += 4;
  view.setUint32(offset, index.efConstruction, true);
  offset += 4;
  view.setUint32(offset, index.efSearch, true);
  offset += 4;
  view.setInt32(offset, index.levelMax, true);
  offset += 4;
  view.setInt32(offset, index.entryPointId, true);
  offset += 4;
  view.setUint32(offset, index.nodes.size, true);
  offset += 4;

  for (const [id, node] of index.nodes) {
    view.setInt32(offset, id, true);
    offset += 4;
    view.setUint16(offset, node.level, true);
    offset += 2;
    view.setUint16(offset, 0, true);
    offset += 2;

    if (encoding === 'float32') {
      for (const value of node.vector) {
        view.setFloat32(offset, value, true);
        offset += 4;
      }
    } else {
      const magnitude = norm(node.vector);
      for (const value of node.vector) {
        const normalized = magnitude === 0 ? 0 : value / magnitude;
        view.setInt16(offset, Math.round(Math.max(-1, Math.min(1, normalized)) * 32767), true);
        offset += 2;
      }
    }

    for (let level = 0; level <= node.level; level++) {
      const neighbors = node.neighbors[level] ?? [];
      view.setUint32(offset, neighbors.length, true);
      offset += 4;
      for (const neighbor of neighbors) {
        view.setInt32(offset, neighbor, true);
        offset += 4;
      }
    }
  }

  return output;
}

class Reader {
  private offset = 0;
  private readonly view: DataView;

  constructor(private readonly bytes: Uint8Array) {
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }

  remaining(): number {
    return this.bytes.byteLength - this.offset;
  }

  readBytes(length: number): Uint8Array {
    this.ensure(length);
    const result = this.bytes.subarray(this.offset, this.offset + length);
    this.offset += length;
    return result;
  }

  u8(): number {
    this.ensure(1);
    return this.view.getUint8(this.offset++);
  }
  u16(): number {
    this.ensure(2);
    const value = this.view.getUint16(this.offset, true);
    this.offset += 2;
    return value;
  }
  i16(): number {
    this.ensure(2);
    const value = this.view.getInt16(this.offset, true);
    this.offset += 2;
    return value;
  }
  u32(): number {
    this.ensure(4);
    const value = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return value;
  }
  i32(): number {
    this.ensure(4);
    const value = this.view.getInt32(this.offset, true);
    this.offset += 4;
    return value;
  }
  f32(): number {
    this.ensure(4);
    const value = this.view.getFloat32(this.offset, true);
    this.offset += 4;
    return value;
  }

  private ensure(length: number): void {
    if (!Number.isSafeInteger(length) || length < 0 || length > this.remaining()) {
      throw new Error('Truncated or invalid HNSW binary index');
    }
  }
}

export function deserializeHNSW(data: ArrayBuffer | Uint8Array): HNSW {
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  const reader = new Reader(bytes);
  const magic = reader.readBytes(4);
  if (!MAGIC.every((byte, position) => magic[position] === byte)) {
    throw new Error('Invalid HNSW binary index magic');
  }
  const version = reader.u16();
  if (version !== FORMAT_VERSION) throw new Error(`Unsupported HNSW binary index version ${version}`);
  const metricCode = reader.u8();
  if (metricCode > 1) throw new Error('Invalid HNSW metric encoding');
  const vectorEncoding = reader.u8();
  if (vectorEncoding > 1) throw new Error('Invalid HNSW vector encoding');
  const dimensions = reader.u32();
  const M = reader.u32();
  const efConstruction = reader.u32();
  const efSearch = reader.u32();
  const levelMax = reader.i32();
  const entryPointId = reader.i32();
  const nodeCount = reader.u32();
  if (dimensions < 1 || dimensions > MAX_DIMENSIONS) throw new Error('Invalid HNSW vector dimensions');
  if (nodeCount > MAX_NODES) throw new Error('Invalid HNSW node count');
  const bytesPerValue = vectorEncoding === 0 ? 4 : 2;
  const minimumNodeBytes = 12 + dimensions * bytesPerValue;
  if (nodeCount > Math.floor(reader.remaining() / minimumNodeBytes)) {
    throw new Error('HNSW node count exceeds the available data');
  }

  const metric = metricCode === 0 ? 'cosine' : 'euclidean';
  if (metric === 'euclidean' && vectorEncoding === 1) {
    throw new Error('int16 vectors cannot be restored into a euclidean index');
  }
  const index = new HNSW(M, efConstruction, dimensions, metric, efSearch);
  index.levelMax = levelMax;
  index.entryPointId = entryPointId;
  index.nodes = new Map();

  for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
    const id = reader.i32();
    const level = reader.u16();
    reader.u16(); // reserved
    if (index.nodes.has(id)) throw new Error(`Duplicate HNSW node id ${id}`);
    if (level > nodeCount) throw new Error(`Invalid HNSW level on node ${id}`);
    const vector = new Float32Array(dimensions);
    for (let dimension = 0; dimension < dimensions; dimension++) {
      vector[dimension] = vectorEncoding === 0 ? reader.f32() : reader.i16() / 32767;
    }
    const node = new Node(id, vector, level);
    node.neighbors = [];
    for (let currentLevel = 0; currentLevel <= level; currentLevel++) {
      const count = reader.u32();
      if (count > Math.floor(reader.remaining() / 4)) throw new Error(`Invalid neighbor count on node ${id}`);
      const neighbors = new Array<number>(count);
      for (let neighborIndex = 0; neighborIndex < count; neighborIndex++) neighbors[neighborIndex] = reader.i32();
      node.neighbors.push(neighbors);
    }
    index.nodes.set(id, node);
  }

  if (reader.remaining() !== 0) throw new Error('Unexpected trailing data in HNSW binary index');
  if (nodeCount === 0) {
    if (entryPointId !== -1 || levelMax !== -1) throw new Error('Invalid empty HNSW index metadata');
  } else if (!index.nodes.has(entryPointId)) {
    throw new Error('HNSW entry point does not exist');
  }
  for (const [id, node] of index.nodes) {
    for (const neighbors of node.neighbors) {
      for (const neighbor of neighbors) {
        if (!index.nodes.has(neighbor)) throw new Error(`Node ${id} refers to missing neighbor ${neighbor}`);
      }
    }
  }
  return index;
}
