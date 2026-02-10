import { HNSW } from './main';
import { openDB, deleteDB, DBSchema, IDBPDatabase } from 'idb';

interface HNSWDB extends DBSchema {
  'hnsw-index': {
    key: string;
    value: any;
  };
}

export class HNSWWithDB extends HNSW {
  dbName: string;
  db: IDBPDatabase<HNSWDB> | null = null;

  private constructor(M: number, efConstruction: number, dbName: string, efSearch = 50) {
    super(M, efConstruction, null, 'cosine', efSearch);
    this.dbName = dbName;
  }

  static async create(M: number, efConstruction: number, dbName: string, efSearch = 50) {
    const instance = new HNSWWithDB(M, efConstruction, dbName, efSearch);
    await instance.initDB();
    return instance;
  }

  private async initDB() {
    this.db = await openDB<HNSWDB>(this.dbName, 1, {
      upgrade(db) {
        db.createObjectStore('hnsw-index');
      },
    });
  }

  async saveIndex() {
    if (!this.db) {
      // console.error('Database is not initialized');
      return;
    }

    await this.db.put('hnsw-index', this.toJSON(), 'hnsw');
  }

  async loadIndex() {
    if (!this.db) {
      // console.error('Database is not initialized');
      return;
    }

    const loadedHNSW: HNSW | undefined = await this.db.get('hnsw-index', 'hnsw');

    if (!loadedHNSW) {
      // console.error('No saved HNSW index found');
      return;
    }

    // Update this HNSW instance with loaded data
    const hnsw = HNSW.fromJSON(loadedHNSW);
    this.M = hnsw.M;
    this.efConstruction = hnsw.efConstruction;
    this.efSearch = hnsw.efSearch;
    this.metric = hnsw.metric;
    this.d = hnsw.d;
    this.similarityFunction = (this as any).getMetric(hnsw.metric);
    this.levelMax = hnsw.levelMax;
    this.entryPointId = hnsw.entryPointId;
    this.nodes = hnsw.nodes;
  }

  async deleteIndex() {
    if (!this.db) {
      // console.error('Database is not initialized');
      return;
    }

    try {
      await deleteDB(this.dbName);
      this.initDB();
    } catch (error) {
      // console.error('Failed to delete index:', error);
    }
  }
}
