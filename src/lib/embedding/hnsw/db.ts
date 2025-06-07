
import { HNSW } from './main';

export class HNSWWithDB extends HNSW {
  private db: Map<number, any> = new Map();

  constructor(M = 16, efConstruction = 200, d: number | null = null, metric = 'cosine') {
    super(M, efConstruction, d, metric);
  }

  async addPointWithData(id: number, vector: Float32Array | number[], data: any) {
    await this.addPoint(id, vector);
    this.db.set(id, data);
  }

  getPointData(id: number): any {
    return this.db.get(id);
  }

  searchKNNWithData(query: Float32Array | number[], k: number): { id: number; score: number; data: any }[] {
    const results = this.searchKNN(query, k);
    return results.map(result => ({
      ...result,
      data: this.getPointData(result.id)
    }));
  }

  toJSON() {
    const baseJSON = super.toJSON();
    return {
      ...baseJSON,
      db: Array.from(this.db.entries())
    };
  }

  static fromJSON(json: any): HNSWWithDB {
    const hnsw = new HNSWWithDB(json.M, json.efConstruction);
    hnsw.levelMax = json.levelMax;
    hnsw.entryPointId = json.entryPointId;
    hnsw.nodes = new Map(
      json.nodes.map(([id, node]: [number, any]) => {
        return [
          id,
          {
            ...node,
            vector: new Float32Array(node.vector),
          },
        ];
      }),
    );
    if (json.db) {
      hnsw.db = new Map(json.db);
    }
    return hnsw;
  }
}
