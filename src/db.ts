import { HNSW } from './main';
import { openDB, deleteDB, DBSchema, IDBPDatabase } from 'idb';
import { cosineSimilarity, euclideanSimilarity } from './similarity';

type SerializedIndex = ReturnType<HNSW['toJSON']>;

interface HNSWDB extends DBSchema {
  'hnsw-index': {
    key: string;
    value: SerializedIndex;
  };
}

export class HNSWWithDB extends HNSW {
  dbName: string;
  db: IDBPDatabase<HNSWDB> | null = null;

  private constructor(M: number, efConstruction: number, dbName: string, efSearch = 50) {
    super(M, efConstruction, null, 'cosine', efSearch);
    this.dbName = dbName;
  }

  /**
   * Creates an IndexedDB-backed HNSW instance.
   */
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

  /**
   * Closes the current IndexedDB connection if open.
   */
  close() {
    if (!this.db) {
      return;
    }
    this.db.close();
    this.db = null;
  }

  /**
   * Persists the current graph to IndexedDB.
   */
  async saveIndex() {
    if (!this.db) {
      throw new Error('Database is not initialized');
    }

    await this.db.put('hnsw-index', this.toJSON(), 'hnsw');
  }

  /**
   * Loads a persisted graph from IndexedDB if present.
   */
  async loadIndex() {
    if (!this.db) {
      throw new Error('Database is not initialized');
    }

    const loadedHNSW = await this.db.get('hnsw-index', 'hnsw');

    if (!loadedHNSW) {
      return;
    }

    // Update this HNSW instance with loaded data
    const hnsw = HNSW.fromJSON(loadedHNSW);
    this.M = hnsw.M;
    this.efConstruction = hnsw.efConstruction;
    this.efSearch = hnsw.efSearch;
    this.metric = hnsw.metric;
    this.d = hnsw.d;
    this.similarityFunction = hnsw.metric === 'cosine' ? cosineSimilarity : euclideanSimilarity;
    this.levelMax = hnsw.levelMax;
    this.entryPointId = hnsw.entryPointId;
    this.nodes = hnsw.nodes;
  }

  /**
   * Deletes persisted graph data and re-initializes the backing DB.
   */
  async deleteIndex() {
    if (!this.db) {
      throw new Error('Database is not initialized');
    }

    this.close();
    await deleteDB(this.dbName);
    await this.initDB();
  }
}
