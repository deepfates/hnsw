# Benchmarks

This folder contains a lightweight benchmark harness to validate HNSW behavior and quantify the impact of recent changes (heap-based search, `efSearch`, neighbor heuristic, and serialization updates).

## Goals

- Measure **recall@k vs latency** across `efSearch`, `M`, and `efConstruction`
- Validate **build time** and **query latency**
- Provide **repeatable results** for regression checks

## Structure

- `src/bench/`
  - `dataset.ts`: dataset loaders + synthetic generator
  - `metrics.ts`: brute-force KNN + recall computation
  - `run.ts`: CLI benchmark runner
- `bench/`
  - `datasets/`: place datasets here
  - `outputs/`: JSON benchmark outputs

## Running benchmarks

Build the project first:

```/dev/null/build.sh#L1-1
npm run build
```

### Download SIFT small (10k) dataset

```/dev/null/download-siftsmall.sh#L1-2
node dist/bench/download.js --extract
```

This will place `siftsmall_base.fvecs` and `siftsmall_query.fvecs` under `bench/datasets/`.

### Synthetic dataset (fast sanity check)

```/dev/null/synthetic.sh#L1-2
node dist/bench/run.js --mode synthetic --count 10000 --dim 64 --metric cosine
```

You can tune parameters:

```/dev/null/synthetic-advanced.sh#L1-2
node dist/bench/run.js --mode synthetic --count 20000 --dim 128 --metric euclidean --k 10 --M 8,16,32 --efConstruction 100,200 --efSearch 10,50,100
```

### FVECS dataset (SIFT / GloVe style)

Put `.fvecs` files in `bench/datasets/`, then run:

```/dev/null/fvecs.sh#L1-2
node dist/bench/run.js --mode fvecs --base bench/datasets/siftsmall_base.fvecs --query bench/datasets/siftsmall_query.fvecs --metric euclidean --limit 10000 --query-limit 100
```

## Output

Results are written to `bench/outputs/` as JSON:

```json
{
  "config": { ... },
  "results": [
    {
      "dataset": { "name": "...", "metric": "...", "dimension": 128, "count": 10000, "queries": 100 },
      "params": { "M": 16, "efConstruction": 200, "efSearch": 50, "k": 10 },
      "buildMs": 1234.56,
      "searchLatencyMs": { "count": 100, "avg": 2.3, "p50": 2.1, "p90": 3.5, "p95": 4.0, "p99": 6.2 },
      "recallAtK": 0.92
    }
  ]
}
```

## Comparing results (baseline vs changes)

Run the benchmark on your baseline code, then on your changes, and compare:

```/dev/null/report.sh#L1-4
node dist/bench/report.js --base bench/outputs/baseline.json --candidate bench/outputs/changes.json --format csv --output bench/outputs/compare.csv
node dist/bench/report.js --base-dir bench/outputs/baseline --candidate-dir bench/outputs/changes --format json --output bench/outputs/compare.json
```

Or run the one-shot compare runner (checks out refs, runs both, and produces the report):

```/dev/null/compare.sh#L1-2
node dist/bench/compare.js --base-ref HEAD~1 --candidate-ref HEAD --mode fvecs --base bench/datasets/siftsmall_base.fvecs --query bench/datasets/siftsmall_query.fvecs --metric euclidean --limit 10000 --query-limit 100 --tag sift-small
```

The report includes deltas for recall@k, latency (avg + p95), and build time.

## Recommended sanity sweep

A minimal sweep to verify improvements:

- `M`: 8, 16
- `efConstruction`: 100, 200
- `efSearch`: 10, 50, 100
- `k`: 10

This should show:
- Increasing `efSearch` improves recall at the cost of latency.
- Higher `M` improves recall/robustness but increases build time and memory.

## Notes

- The benchmark uses **brute-force KNN** for exact recall on the query set.
- Synthetic mode is great for quick regressions, but use real datasets for meaningful recall curves.
- If you change the similarity function or search logic, rerun a sweep to confirm recall/latency curves remain healthy.