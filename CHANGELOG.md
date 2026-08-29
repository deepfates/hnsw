# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

## [Unreleased]

### Added
- Versioned binary serialization for transporting or statically hosting in-memory indexes. The format is bounds-checked on restore and supports lossless `float32` vectors or compact normalized `int16` vectors for cosine indexes.
- An optional constructor random source for reproducible graph construction.
- An ES-module build for browser bundlers, alongside the existing CommonJS entry point.

### Changed
- Cosine comparisons now cost a single dot product: each node's L2 norm is cached lazily on first cosine use and the query norm is computed once per operation (~2.1x faster build+query measured on 10k x 384). This relies on a now-documented contract: vectors must not be mutated after insertion; the index caches derived values. Mutating a stored vector yields stale-but-stable scores. Euclidean and custom `similarityFunction` overrides are unaffected and take the original code path.

## [1.1.1] - 2026-02-16

### Changed
- npm package contents now use an explicit `files` whitelist to publish only runtime artifacts and top-level docs.

## [1.1.0] - 2026-02-16

### Added
- CI workflow for build, lint, test, and coverage.
- Persistence-focused test coverage for IndexedDB-backed index behavior.
- API reference and tuning guidance in README.

### Changed
- Lint gate now targets published library sources (`src/**`, excluding benchmark CLI code).
- README persistence example now loads from the same DB name it saved to.

### Fixed
- `HNSWWithDB.deleteIndex()` now awaits DB re-initialization.
- `HNSWWithDB` now surfaces initialization/load/delete errors instead of silently swallowing them.
